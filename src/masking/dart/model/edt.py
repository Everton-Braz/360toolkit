# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""Triton kernel for euclidean distance transform (EDT)"""

import torch
import cv2
import numpy as np

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = None
    tl = None

def edt_cv2(data: torch.Tensor) -> torch.Tensor:
    """Computes EDT using OpenCV fallback."""
    device = data.device
    np_data = data.detach().cpu().numpy().astype(np.uint8)
    out = np.zeros_like(np_data, dtype=np.float32)
    for b in range(np_data.shape[0]):
        src = (1 - np_data[b]).astype(np.uint8)
        out[b] = cv2.distanceTransform(src, cv2.DIST_L2, 0)
    return torch.from_numpy(out).to(device)

if HAS_TRITON:
    @triton.jit
    def edt_kernel(inputs_ptr, outputs_ptr, v, z, height, width, horizontal: tl.constexpr):
        batch_id = tl.program_id(axis=0)
        if horizontal:
            row_id = tl.program_id(axis=1)
            block_start = (batch_id * height * width) + row_id * width
            length = width
            stride = 1
        else:
            col_id = tl.program_id(axis=1)
            block_start = (batch_id * height * width) + col_id
            length = height
            stride = width

        k = 0
        tl.store(v + block_start, 0)
        tl.store(z + block_start + 0, -1e18)
        tl.store(z + block_start + 1, 1e18)

        for q in range(1, length):
            f_q = tl.load(inputs_ptr + block_start + (q * stride))
            v_k = tl.load(v + block_start + (k * stride))
            f_v_k = tl.load(inputs_ptr + block_start + (v_k * stride))

            s = (f_q - f_v_k + q * q - v_k * v_k) / (2 * q - 2 * v_k)
            z_k = tl.load(z + block_start + (k * stride))

            while s <= z_k:
                k -= 1
                v_k = tl.load(v + block_start + (k * stride))
                f_v_k = tl.load(inputs_ptr + block_start + (v_k * stride))
                s = (f_q - f_v_k + q * q - v_k * v_k) / (2 * q - 2 * v_k)
                z_k = tl.load(z + block_start + (k * stride))

            k += 1
            tl.store(v + block_start + (k * stride), q)
            tl.store(z + block_start + (k * stride), s)
            tl.store(z + block_start + ((k + 1) * stride), 1e18)

        k = 0
        for q in range(length):
            z_k_plus_1 = tl.load(z + block_start + ((k + 1) * stride))
            while z_k_plus_1 < q:
                k += 1
                z_k_plus_1 = tl.load(z + block_start + ((k + 1) * stride))

            r = tl.load(v + block_start + (k * stride))
            d = q - r
            old_value = tl.load(inputs_ptr + block_start + (r * stride))
            tl.store(outputs_ptr + block_start + (q * stride), old_value + d * d)

    def edt_triton(data: torch.Tensor):
        assert data.dim() == 3
        if not data.is_cuda:
            return edt_cv2(data)
        B, H, W = data.shape
        data = data.contiguous()
        output = torch.where(data, 1e18, 0.0)
        assert output.is_contiguous()
        parabola_loc = torch.zeros(B, H, W, dtype=torch.uint32, device=data.device)
        parabola_inter = torch.empty(B, H, W, dtype=torch.float, device=data.device)
        parabola_inter[:, :, 0] = -1e18
        parabola_inter[:, :, 1] = 1e18
        grid = (B, H)
        edt_kernel[grid](
            output.clone(),
            output,
            parabola_loc,
            parabola_inter,
            H,
            W,
            horizontal=True,
        )
        parabola_loc.zero_()
        parabola_inter[:, :, 0] = -1e18
        parabola_inter[:, :, 1] = 1e18
        grid = (B, W)
        edt_kernel[grid](
            output.clone(),
            output,
            parabola_loc,
            parabola_inter,
            H,
            W,
            horizontal=False,
        )
        return output.sqrt()
else:
    edt_triton = edt_cv2
