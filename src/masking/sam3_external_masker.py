"""
External SAM3.cpp masking integration (batch-capable).

Shells out to segment_persons.exe once per batch, not once per image.
The model loads a single time, dramatically reducing overhead for large
image sets.

Stage 3 pipeline flow
─────────────────────
1. Collect all images to process.
2. Write a temp image-list file.
3. Call segment_persons.exe --image-list --prompts --output-dir
4. Parse PROCESSED lines from stdout for per-image progress.
5. For each image:
   a. Load the raw SAM3 mask ({stem}_mask.png).
   b. Resize to original image dimensions with OpenCV.
   c. Apply morph (erode >0 / dilate <0) if morph_radius != 0.
   d. Invert to RealityScan convention (0=remove, 255=keep).
   e. Save {stem}_mask.png.
   f. Optionally save {stem}_alpha.png (RGBA, alpha=0 where detected).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import threading as _threading

import cv2
import numpy as np

from ..config.defaults import MASK_VALUE_KEEP, MASK_VALUE_REMOVE
from .mask_refinement import MaskRefinementSettings, refine_detected_mask

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt catalogue
# ---------------------------------------------------------------------------
_PROMPT_MAP: Dict[str, str] = {
    'persons':  'person',
    'bags':     'bag',
    'phones':   'phone',
    'hats':     'hat',
    'helmets':  'helmet',
    'sky':      'sky',
}


class SAM3ExternalMasker:
    """Masker adapter for external SAM3.cpp executables (batch mode)."""

    _IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    def __init__(
        self,
        segment_persons_exe: str,
        model_path: str,
        sam3_image_exe: Optional[str] = None,
        use_gpu: bool = True,
        feather_radius: int = 0,
        morph_radius: int = 0,
        alpha_export: bool = False,
        alpha_only: bool = False,
        max_input_width: int = 0,
        score_threshold: float = 0.5,
        nms_threshold: float = 0.1,
        enable_refinement: bool = True,
        refine_sky_only: bool = True,
        seam_aware_refinement: bool = True,
        edge_sharpen_strength: float = 0.75,
    ):
        self.segment_persons_exe = Path(segment_persons_exe).expanduser()
        self.model_path = Path(model_path).expanduser()
        self.sam3_image_exe = Path(sam3_image_exe).expanduser() if sam3_image_exe else None
        self.use_gpu = use_gpu
        self.morph_radius = int(morph_radius)
        self.alpha_export = bool(alpha_export)
        self.alpha_only = bool(alpha_only)
        self.max_input_width = int(max_input_width)
        self.score_threshold = float(score_threshold)
        self.nms_threshold = float(nms_threshold)
        self.feather_radius = int(feather_radius)
        self.enable_refinement = bool(enable_refinement)
        self.refine_sky_only = bool(refine_sky_only)
        self.seam_aware_refinement = bool(seam_aware_refinement)
        self.edge_sharpen_strength = float(edge_sharpen_strength)
        self.cancelled = False

        self.enabled_categories: Dict[str, bool] = {k: True for k in _PROMPT_MAP}
        self.custom_prompts: List[str] = []

        self._validate_runtime()

    # ------------------------------------------------------------------
    # Validation / runtime helpers
    # ------------------------------------------------------------------

    def _validate_runtime(self) -> None:
        if not self.segment_persons_exe.exists():
            raise FileNotFoundError(
                f"SAM3.cpp segmenter not found: {self.segment_persons_exe}"
            )
        if not self.model_path.exists():
            raise FileNotFoundError(f"SAM3.cpp model not found: {self.model_path}")
        if self.sam3_image_exe and not self.sam3_image_exe.exists():
            logger.warning("SAM3.cpp interactive GUI not found: %s", self.sam3_image_exe)

    def _build_runtime_env(self, executable_path: Path) -> dict:
        env = os.environ.copy()
        search = []
        exe_dir = executable_path.parent
        if exe_dir.exists():
            search.append(str(exe_dir))
        if len(executable_path.parents) >= 3:
            dll_dir = exe_dir.parent.parent / 'bin' / exe_dir.name
            if dll_dir.exists():
                search.append(str(dll_dir))
        if search:
            existing = env.get('PATH', '')
            env['PATH'] = os.pathsep.join(search + ([existing] if existing else []))
        return env

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def set_enabled_categories(self, categories: Dict[str, bool]) -> None:
        if categories:
            self.enabled_categories.update(categories)

    def set_custom_prompts(self, text: str) -> None:
        self.custom_prompts = [t.strip() for t in text.split(',') if t.strip()]

    def _build_prompts(self) -> List[str]:
        prompts = []
        for key, prompt in _PROMPT_MAP.items():
            if self.enabled_categories.get(key, False):
                prompts.append(prompt)
        prompts.extend(self.custom_prompts)
        seen: set = set()
        unique = []
        for p in prompts:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        return unique or ['person']

    def request_cancellation(self) -> None:
        self.cancelled = True

    def cancel(self) -> None:
        self.cancelled = True

    def reset_cancel(self) -> None:
        self.cancelled = False

    # ------------------------------------------------------------------
    # Optional downscale for faster SAM3 encoding
    # ------------------------------------------------------------------

    def _maybe_downscale(
        self, image_path: Path, dest_dir: Path
    ) -> Tuple[Path, int, int]:
        img = cv2.imread(str(image_path))
        if img is None:
            raise RuntimeError(f"Cannot load image: {image_path}")
        h, w = img.shape[:2]
        if self.max_input_width > 0 and w > self.max_input_width:
            scale = self.max_input_width / w
            new_w = self.max_input_width
            new_h = max(1, int(h * scale))
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            dest = dest_dir / image_path.name
            cv2.imwrite(str(dest), resized)
            return dest, w, h
        return image_path, w, h

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _should_refine_mask(self) -> bool:
        if not self.enable_refinement or self.feather_radius <= 0:
            return False
        if self.refine_sky_only and not self.enabled_categories.get('sky', False):
            return False
        return True

    def _detected_to_rs_mask(self, detected_mask: np.ndarray) -> np.ndarray:
        rs_mask = np.full_like(detected_mask, MASK_VALUE_KEEP)
        rs_mask[detected_mask > 0] = MASK_VALUE_REMOVE
        return rs_mask

    def _save_overlay(self, orig_img: np.ndarray, rs_mask: np.ndarray, dest_path: Path) -> None:
        display = orig_img
        if display.ndim == 2:
            display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
        elif display.ndim == 3 and display.shape[2] == 4:
            display = cv2.cvtColor(display, cv2.COLOR_BGRA2BGR)
        overlay = display.copy()
        overlay[rs_mask == MASK_VALUE_REMOVE] = [0, 0, 255]
        blended = cv2.addWeighted(display, 0.72, overlay, 0.28, 0)
        cv2.imwrite(str(dest_path), blended)

    def _postprocess_mask(
        self,
        raw_mask_path: Path,
        orig_w: int,
        orig_h: int,
        orig_image_path: Path,
        output_dir: Path,
    ) -> Path:
        raw = cv2.imread(str(raw_mask_path), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            raw = np.zeros((orig_h, orig_w), dtype=np.uint8)

        _, bin_mask = cv2.threshold(raw, 127, 255, cv2.THRESH_BINARY)

        # Apply morph at the RAW mask resolution (SAM3 processing scale).
        # Applying morph BEFORE upscaling ensures radius-3 means the same
        # 3 pixels relative to the SAM3 encode size, not the 8K original.
        # bin_mask convention: white (255) = detected/remove, black (0) = keep.
        # Dilate (positive): grow detected area → expand removed region → fills gaps
        # Erode  (negative): shrink detected area → tighter boundary around objects
        if self.morph_radius != 0:
            r = abs(self.morph_radius)
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)
            )
            bin_mask = cv2.dilate(bin_mask, kernel) if self.morph_radius > 0 \
                else cv2.erode(bin_mask, kernel)

        # NOW upscale to original dimensions
        if bin_mask.shape[1] != orig_w or bin_mask.shape[0] != orig_h:
            bin_mask = cv2.resize(bin_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        orig_img = cv2.imread(str(orig_image_path), cv2.IMREAD_UNCHANGED)
        if orig_img is not None and self._should_refine_mask():
            try:
                bin_mask = refine_detected_mask(
                    orig_img,
                    bin_mask,
                    MaskRefinementSettings(
                        enabled=True,
                        edge_band_radius=self.feather_radius,
                        sharpen_strength=self.edge_sharpen_strength,
                        seam_aware=self.seam_aware_refinement,
                    ),
                )
            except Exception as exc:
                logger.warning('[SAM3] Refinement failed for %s: %s', orig_image_path.name, exc)

        # RealityScan: 0=remove (detected), 255=keep (background)
        rs_mask = self._detected_to_rs_mask(bin_mask)

        stem = raw_mask_path.stem.replace('_mask', '')
        final_mask_path = output_dir / f'{stem}_mask.png'

        # skip mask file when user chose alpha-only mode
        if not self.alpha_only:
            cv2.imwrite(str(final_mask_path), rs_mask)

        if self.alpha_export:
            if orig_img is not None:
                if orig_img.ndim == 2:
                    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)
                elif orig_img.ndim == 3 and orig_img.shape[2] == 4:
                    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGRA2BGR)
                rgba = cv2.cvtColor(orig_img, cv2.COLOR_BGR2BGRA)
                rgba[:, :, 3] = rs_mask   # 0=transparent where detected
                # Save alpha PNG in the output dir so user can find it easily
                alpha_dest = output_dir / f'{stem}_alpha.png'
                cv2.imwrite(str(alpha_dest), rgba)

        return final_mask_path

    # ------------------------------------------------------------------
    # Core: run segment_persons.exe (batch)
    # ------------------------------------------------------------------

    def _run_batch_exe(
        self,
        image_paths: List[Path],
        raw_mask_dir: Path,
        progress_callback: Optional[Callable] = None,
        cancellation_check: Optional[Callable] = None,
    ) -> Dict[str, bool]:
        prompts = self._build_prompts()
        logger.info("[SAM3] batch: %d image(s), prompts: %s", len(image_paths), prompts)

        list_fd, list_path = tempfile.mkstemp(suffix='.txt', prefix='sam3_list_')
        try:
            with os.fdopen(list_fd, 'w', encoding='utf-8') as f:
                for p in image_paths:
                    f.write(str(p) + '\n')

            cmd = [
                str(self.segment_persons_exe),
                '--model',      str(self.model_path),
                '--image-list', list_path,
                '--output-dir', str(raw_mask_dir),
                '--prompts',    ','.join(prompts),
                '--score',      str(self.score_threshold),
                '--nms',        str(self.nms_threshold),
            ]
            if self.use_gpu:
                cmd.append('--gpu')
            else:
                cmd.append('--no-gpu')

            logger.info("[SAM3] cmd: %s", ' '.join(cmd))
            env = self._build_runtime_env(self.segment_persons_exe)

            proc = subprocess.Popen(
                cmd,
                cwd=str(raw_mask_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )

            processed: Dict[str, bool] = {}
            total = len(image_paths)
            stderr_lines: List[str] = []

            # Drain stderr in a background thread to prevent pipe-buffer deadlock.
            def _drain_stderr():
                for ln in proc.stderr:
                    stderr_lines.append(ln.rstrip())

            stderr_thread = _threading.Thread(target=_drain_stderr, daemon=True)
            stderr_thread.start()

            for line in proc.stdout:
                line = line.rstrip()
                if line.startswith('PROCESSED '):
                    parts = line.split()
                    stem = parts[1] if len(parts) > 1 else '?'
                    dets = int(parts[6]) if len(parts) > 6 else -1
                    processed[stem] = True
                    logger.debug("[SAM3] %s  detections=%d", stem, dets)
                    if progress_callback:
                        progress_callback(len(processed), total, f"SAM3: {stem}")
                elif line:
                    logger.debug("[SAM3 stderr] %s", line)

                if cancellation_check and cancellation_check():
                    proc.terminate()
                    self.cancelled = True
                    break

            proc.wait()
            stderr_thread.join(timeout=5)
            stderr_tail = '\n'.join(stderr_lines[-20:]) if stderr_lines else ''
            if proc.returncode not in (0, 5):
                raise RuntimeError(
                    f"segment_persons.exe exited {proc.returncode}.\n{stderr_tail}"
                )
        finally:
            try:
                os.unlink(list_path)
            except OSError:
                pass

        return processed

    # ------------------------------------------------------------------
    # Public: generate_preview_assets (single image)
    # ------------------------------------------------------------------

    def generate_preview_assets(
        self,
        image_path: Path,
        work_dir: Optional[Path] = None,
    ) -> Dict[str, str]:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Preview image not found: {image_path}")

        resolved = work_dir or (
            Path(tempfile.gettempdir()) / '360toolkit_sam3_preview' / image_path.stem
        )
        resolved.mkdir(parents=True, exist_ok=True)
        raw_dir    = resolved / '_raw';     raw_dir.mkdir(exist_ok=True)
        scaled_dir = resolved / '_scaled';  scaled_dir.mkdir(exist_ok=True)

        proc_path, orig_w, orig_h = self._maybe_downscale(image_path, scaled_dir)
        self._run_batch_exe([proc_path], raw_dir)

        raw_mask = raw_dir / f'{proc_path.stem}_mask.png'
        if not raw_mask.exists():
            raise RuntimeError("SAM3.cpp finished without producing a mask file.")

        final_mask_path = self._postprocess_mask(
            raw_mask, orig_w, orig_h, image_path, resolved
        )

        # Build a colour overlay for the preview widget.
        # Use the FINAL post-processed mask (after morph) so dilate/erode is visible.
        overlay_path = resolved / f'{image_path.stem}_persons_overlay.png'
        final_data = cv2.imread(str(final_mask_path), cv2.IMREAD_GRAYSCALE)
        if final_data is None:
            # fallback: read raw mask if final wasn't written (alpha_only mode)
            final_data = cv2.imread(str(raw_mask), cv2.IMREAD_GRAYSCALE)
            if final_data is not None:
                _, final_data = cv2.threshold(final_data, 127, 255, cv2.THRESH_BINARY)
                # raw is 255=person; invert to rs_mask convention (0=person)
                final_data = cv2.bitwise_not(final_data)
        if final_data is not None:
            orig_img = cv2.imread(str(image_path))
            if orig_img is not None:
                if final_data.shape[1] != orig_w or final_data.shape[0] != orig_h:
                    final_data = cv2.resize(
                        final_data, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
                    )
                self._save_overlay(orig_img, final_data, overlay_path)

        return {
            'image_path':   str(image_path),
            'overlay_path': str(overlay_path) if overlay_path.exists() else '',
            'mask_path':    str(final_mask_path),
            'work_dir':     str(resolved),
        }

    # ------------------------------------------------------------------
    # Public: generate_mask (single image, compat)
    # ------------------------------------------------------------------

    def generate_mask(
        self,
        image_path: Path,
        output_path: Optional[Path] = None,
    ) -> Optional[np.ndarray]:
        preview = self.generate_preview_assets(image_path)
        mask = cv2.imread(preview['mask_path'], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(preview['mask_path'], output_path)
        return mask

    # ------------------------------------------------------------------
    # Public: process_batch (Stage 3 entry point)
    # ------------------------------------------------------------------

    def process_batch(
        self,
        input_dir: str,
        output_dir: str,
        save_visualization: bool = False,
        progress_callback: Optional[Callable] = None,
        cancellation_check: Optional[Callable] = None,
    ) -> Dict:
        input_path  = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        image_files: List[Path] = sorted(
            [
                p for p in input_path.rglob('*')
                if p.is_file()
                and p.suffix.lower() in self._IMAGE_EXTENSIONS
                and '_mask' not in p.stem
                and '_alpha' not in p.stem
            ],
            key=lambda p: p.name.lower(),
        )

        total = len(image_files)
        if total == 0:
            return {
                'successful': 0, 'failed': 0, 'skipped': 0,
                'total': 0, 'masks_created': 0,
            }

        todo: List[Path] = []
        skipped = 0
        for img in image_files:
            # In alpha_only mode no _mask.png is written, so check for _alpha.png
            done_file = f'{img.stem}_alpha.png' if self.alpha_only else f'{img.stem}_mask.png'
            if (output_path / done_file).exists():
                skipped += 1
            else:
                todo.append(img)

        if not todo:
            return {
                'successful': 0, 'failed': 0, 'skipped': skipped,
                'total': total, 'masks_created': 0,
            }

        raw_dir    = output_path / '_sam3_raw';    raw_dir.mkdir(exist_ok=True)
        scaled_dir = output_path / '_sam3_scaled'; scaled_dir.mkdir(exist_ok=True)

        orig_dims: Dict[str, Tuple[int, int]] = {}
        proc_paths: List[Path] = []
        proc_to_orig: Dict[str, Path] = {}

        for img in todo:
            if self.cancelled or (cancellation_check and cancellation_check()):
                break
            proc_p, ow, oh = self._maybe_downscale(img, scaled_dir)
            orig_dims[proc_p.stem]  = (ow, oh)
            proc_to_orig[proc_p.stem] = img
            proc_paths.append(proc_p)

        successful = 0
        failed = 0

        try:
            self._run_batch_exe(
                proc_paths, raw_dir,
                progress_callback=progress_callback,
                cancellation_check=cancellation_check,
            )
        except Exception as exc:
            logger.error("[SAM3] Batch exe failed: %s", exc)
            shutil.rmtree(raw_dir, ignore_errors=True)
            shutil.rmtree(scaled_dir, ignore_errors=True)
            return {
                'successful': 0, 'failed': len(todo), 'skipped': skipped,
                'total': total, 'masks_created': 0,
            }

        for proc_stem, orig_img in proc_to_orig.items():
            raw_mask = raw_dir / f'{proc_stem}_mask.png'
            ow, oh   = orig_dims.get(proc_stem, (0, 0))
            if not raw_mask.exists():
                logger.warning("[SAM3] No mask for %s", orig_img.name)
                failed += 1
                continue
            if ow == 0:
                tmp = cv2.imread(str(orig_img))
                if tmp is not None:
                    oh, ow = tmp.shape[:2]
            try:
                self._postprocess_mask(raw_mask, ow, oh, orig_img, output_path)
                if save_visualization:
                    vis = raw_dir / f'{proc_stem}_overlay.png'
                    if vis.exists():
                        shutil.copy2(vis, output_path / f'{orig_img.stem}_vis.png')
                successful += 1
            except Exception as exc:
                logger.error("[SAM3] Post-process failed for %s: %s", orig_img.name, exc)
                failed += 1

        shutil.rmtree(raw_dir, ignore_errors=True)
        shutil.rmtree(scaled_dir, ignore_errors=True)

        return {
            'successful':    successful,
            'failed':        failed,
            'skipped':       skipped,
            'total':         total,
            'masks_created': successful,
        }

    # ------------------------------------------------------------------
    # Public: launch interactive GUI
    # ------------------------------------------------------------------

    def launch_interactive_gui(self, image_path: Path) -> subprocess.Popen:
        if not self.sam3_image_exe or not self.sam3_image_exe.exists():
            raise FileNotFoundError(
                "SAM3.cpp interactive GUI executable is not configured."
            )
        cmd = [
            str(self.sam3_image_exe),
            '--model', str(self.model_path),
            '--image', str(image_path),
        ]
        logger.info("[SAM3] launching GUI: %s", ' '.join(cmd))
        return subprocess.Popen(
            cmd,
            cwd=str(self.sam3_image_exe.parent),
            env=self._build_runtime_env(self.sam3_image_exe),
        )
