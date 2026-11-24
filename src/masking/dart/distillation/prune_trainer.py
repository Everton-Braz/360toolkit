"""
Self-distillation trainer for SAM3 block/sub-block pruning.

Teacher = full SAM3 vision backbone (frozen, FP16).
Student = same architecture with pruned blocks (trainable remaining layers).
Loss = weighted MSE on FPN features.

Supports two pruning granularities:
  - Sub-block masking: mask individual attn/mlp within a block (identity skip)
  - Full block removal: skip entire blocks via skip_blocks

After training, saves a pruned checkpoint with removed block weights stripped.
"""

import copy
import os
import time
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from sam3.distillation.distill_trainer import (
    ImageFolderDataset,
    FeatureDistillationLoss,
    dist_print,
    get_rank,
    get_world_size,
    is_dist_initialized,
    is_main_process,
    reduce_scalar,
)


def _log_vram(label: str = ""):
    """Log current GPU VRAM usage (rank 0 only)."""
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    prefix = f"  [{label}] " if label else "  "
    if is_main_process():
        print(f"{prefix}VRAM: {alloc:.1f}GB allocated, {reserved:.1f}GB reserved, {total:.0f}GB total")


def _freeze_masked_blocks(trunk, mask_blocks):
    """Freeze parameters of masked sub-blocks (they're identity, no need to train)."""
    frozen_params = 0
    for block_idx, sub_type in mask_blocks:
        block = trunk.blocks[block_idx]
        if sub_type == "attn":
            for name, p in block.named_parameters():
                if "attn" in name or "norm1" in name or "ls1" in name:
                    p.requires_grad = False
                    frozen_params += p.numel()
        elif sub_type == "mlp":
            for name, p in block.named_parameters():
                if "mlp" in name or "norm2" in name or "ls2" in name:
                    p.requires_grad = False
                    frozen_params += p.numel()
    return frozen_params


def _remove_masked_weights(state_dict, mask_blocks, skip_blocks=None):
    """Remove weights belonging to masked/skipped blocks from state_dict."""
    keys_to_remove = []

    # Sub-block masks: remove attn or mlp weights
    for block_idx, sub_type in mask_blocks:
        prefix = f"trunk.blocks.{block_idx}."
        for key in state_dict:
            if not key.startswith(prefix):
                continue
            if sub_type == "attn" and any(
                k in key for k in (".attn.", ".norm1.", ".ls1.")
            ):
                keys_to_remove.append(key)
            elif sub_type == "mlp" and any(
                k in key for k in (".mlp.", ".norm2.", ".ls2.")
            ):
                keys_to_remove.append(key)

    # Full block skips: remove all weights for the block
    if skip_blocks:
        for block_idx in skip_blocks:
            prefix = f"trunk.blocks.{block_idx}."
            for key in state_dict:
                if key.startswith(prefix):
                    keys_to_remove.append(key)

    keys_to_remove = list(set(keys_to_remove))  # deduplicate
    for key in keys_to_remove:
        del state_dict[key]
    return keys_to_remove


class PruneDistillTrainer:
    """Self-distillation trainer for sub-block pruning.

    Clones the SAM3 vision backbone, applies sub-block masks to the clone,
    and trains the remaining (unmasked) blocks + FPN convs to recover quality.
    """

    def __init__(
        self,
        teacher_model,
        mask_blocks: List[Tuple[int, str]],
        mask_blocks_str: str,
        data_dir: str,
        output_dir: str = "prune_checkpoints",
        lr: float = 1e-4,
        batch_size: int = 1,
        num_epochs: int = 5,
        resolution: int = 1008,
        level_weights: Optional[list] = None,
        num_workers: int = 4,
        save_every: int = 1,
        log_every: int = 50,
        device: str = "cuda",
        skip_blocks: Optional[List[int]] = None,
    ):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.save_every = save_every
        self.log_every = log_every
        self.mask_blocks = mask_blocks
        self.mask_blocks_str = mask_blocks_str
        self.skip_blocks = set(skip_blocks) if skip_blocks else set()

        # Distributed state
        self.distributed = is_dist_initialized()
        self.rank = get_rank()
        self.world_size = get_world_size()

        if self.distributed:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = torch.device(device)

        if is_main_process():
            os.makedirs(output_dir, exist_ok=True)

        # Teacher: full vision backbone, frozen FP16
        self.teacher = teacher_model.backbone.vision_backbone
        self.teacher.eval()
        self.teacher.to(self.device).half()
        for p in self.teacher.parameters():
            p.requires_grad = False

        # Scalp = 1 means drop last FPN level (matching SAM3VLBackbone)
        self.scalp = getattr(teacher_model.backbone, "scalp", 1)

        # Student: deep copy of vision backbone, with masks applied
        dist_print("  Deep-copying vision backbone for student...")
        self.student = copy.deepcopy(self.teacher)
        self.student.float()  # train in FP32
        for p in self.student.parameters():
            p.requires_grad = True

        # Apply sub-block masks to student trunk
        trunk = self.student.trunk
        for block_idx, sub_type in mask_blocks:
            block = trunk.blocks[block_idx]
            if sub_type == "attn":
                block.mask_attn = True
            elif sub_type == "mlp":
                block.mask_mlp = True

        # Apply full block skips
        if self.skip_blocks:
            trunk.skip_blocks = set(self.skip_blocks)
            dist_print(f"  Skipping entire blocks: {sorted(self.skip_blocks)}")
            # Freeze skipped block params (never executed, no gradients)
            for bi in self.skip_blocks:
                for p in trunk.blocks[bi].parameters():
                    p.requires_grad = False

        total_params = sum(p.numel() for p in self.student.parameters())
        trainable = sum(
            p.numel() for p in self.student.parameters() if p.requires_grad
        )
        dist_print(f"  Total student params: {total_params:,}")
        dist_print(f"  Trainable params: {trainable:,}")
        _log_vram("After model setup")

        # Wrap with DDP
        if self.distributed:
            self.student = DDP(
                self.student,
                device_ids=[self.device.index],
                output_device=self.device.index,
                find_unused_parameters=False,
            )
            self._student_unwrapped = self.student.module
        else:
            self._student_unwrapped = self.student

        # Optimizer
        train_params = [p for p in self.student.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(train_params, lr=lr, weight_decay=0.01)

        # Loss
        self.criterion = FeatureDistillationLoss(level_weights)

        # Dataset
        self.dataset = ImageFolderDataset(data_dir, resolution=resolution)
        self.sampler = (
            DistributedSampler(self.dataset, shuffle=True)
            if self.distributed
            else None
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=(self.sampler is None),
            sampler=self.sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

        # LR scheduler
        total_steps = len(self.dataloader) * num_epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps, eta_min=lr * 0.01
        )

        # Mixed precision
        self.scaler = GradScaler()

        eff_batch = batch_size * self.world_size
        dist_print(f"  Dataset: {len(self.dataset)} images from {data_dir}")
        dist_print(
            f"  Batch: {batch_size}/GPU x {self.world_size} GPUs = {eff_batch} eff"
        )
        dist_print(f"  Epochs: {num_epochs}, Steps/epoch: {len(self.dataloader)}")

    @torch.no_grad()
    def _get_teacher_features(self, images: torch.Tensor) -> list:
        images_fp16 = images.half()
        sam3_out, _, _, _ = self.teacher(images_fp16)
        if self.scalp > 0:
            sam3_out = sam3_out[: -self.scalp]
        return [f.float() for f in sam3_out]

    def _get_student_features(self, images: torch.Tensor) -> list:
        student = (
            self.student.module if self.distributed else self.student
        )
        sam3_out, _, _, _ = student(images)
        if self.scalp > 0:
            sam3_out = sam3_out[: -self.scalp]
        return sam3_out

    def train_epoch(self, epoch: int) -> float:
        self.student.train()

        if self.sampler is not None:
            self.sampler.set_epoch(epoch)

        total_loss = 0.0
        num_steps = 0
        t_epoch = time.perf_counter()

        for step, images in enumerate(self.dataloader):
            images = images.to(self.device, non_blocking=True)

            teacher_feats = self._get_teacher_features(images)

            self.optimizer.zero_grad()
            with autocast():
                student_feats = self._get_student_features(images)
                loss_dict = self.criterion(student_feats, teacher_feats)
                loss = loss_dict["loss"]

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()
            num_steps += 1

            if step == 0 and epoch == 0:
                _log_vram("After first step")

            if (step + 1) % self.log_every == 0 and is_main_process():
                avg = total_loss / num_steps
                lr_val = self.scheduler.get_last_lr()[0]
                per_level = loss_dict["per_level_losses"]
                elapsed = time.perf_counter() - t_epoch
                eff_imgs = (step + 1) * self.batch_size * self.world_size
                imgs_per_sec = eff_imgs / elapsed
                print(
                    f"  [Epoch {epoch+1}][{step+1}/{len(self.dataloader)}] "
                    f"loss={avg:.6f} (L0={per_level[0]:.4f} L1={per_level[1]:.4f} "
                    f"L2={per_level[2]:.4f}) lr={lr_val:.2e} {imgs_per_sec:.1f} img/s"
                )

        avg_loss = total_loss / max(num_steps, 1)
        return reduce_scalar(avg_loss, self.device)

    def save_checkpoint(self, epoch: int, loss: float, final: bool = False):
        if not is_main_process():
            return

        state = self._student_unwrapped.state_dict()

        # Remove masked/skipped block weights from checkpoint
        removed = _remove_masked_weights(state, self.mask_blocks, self.skip_blocks)

        suffix = "final" if final else f"epoch{epoch+1}"
        path = os.path.join(self.output_dir, f"pruned_{suffix}.pt")
        ckpt = {
            "epoch": epoch + 1,
            "loss": loss,
            "mask_blocks": self.mask_blocks_str,
            "skip_blocks": sorted(self.skip_blocks) if self.skip_blocks else [],
            "pruned_state_dict": state,
            "removed_keys": removed,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
        }
        torch.save(ckpt, path)
        print(f"  Saved pruned checkpoint: {path} ({len(removed)} keys removed)")

    def resume_from_checkpoint(self, checkpoint_path: str):
        """Load training state from a checkpoint to resume training.

        Args:
            checkpoint_path: path to a pruned_epoch*.pt checkpoint
        """
        dist_print(f"Resuming from checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Restore model weights (pruned_state_dict has masked keys removed,
        # so use strict=False to skip them)
        self._student_unwrapped.load_state_dict(
            ckpt["pruned_state_dict"], strict=False
        )
        self.start_epoch = ckpt["epoch"]
        dist_print(f"  Restored model weights from epoch {self.start_epoch}")

        # Restore optimizer/scheduler/scaler if available (new-style checkpoints)
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
            dist_print(f"  Restored optimizer, scheduler, and scaler state")
        else:
            # Old-style checkpoint without optimizer state — fast-forward scheduler
            steps_per_epoch = len(self.dataloader)
            steps_done = self.start_epoch * steps_per_epoch
            dist_print(
                f"  No optimizer state in checkpoint (old format). "
                f"Fast-forwarding scheduler by {steps_done} steps."
            )
            for _ in range(steps_done):
                self.scheduler.step()

    def train(self):
        start_epoch = getattr(self, "start_epoch", 0)
        if start_epoch >= self.num_epochs:
            dist_print(
                f"Already completed {start_epoch}/{self.num_epochs} epochs. Nothing to do."
            )
            return

        dist_print(f"\nStarting pruning self-distillation...")
        if start_epoch > 0:
            dist_print(f"Resuming from epoch {start_epoch + 1}")
        dist_print(f"{'='*60}")

        for epoch in range(start_epoch, self.num_epochs):
            t0 = time.perf_counter()
            avg_loss = self.train_epoch(epoch)
            elapsed = time.perf_counter() - t0

            dist_print(
                f"Epoch {epoch+1}/{self.num_epochs}: "
                f"avg_loss={avg_loss:.6f} time={elapsed:.1f}s"
            )

            if (epoch + 1) % self.save_every == 0 or (epoch + 1) == self.num_epochs:
                self.save_checkpoint(epoch, avg_loss)

            if self.distributed:
                dist.barrier()

        # Save final
        self.save_checkpoint(self.num_epochs - 1, avg_loss, final=True)

        dist_print(f"\n{'='*60}")
        dist_print("Pruning self-distillation complete!")
