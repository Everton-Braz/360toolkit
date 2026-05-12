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
import sys
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import threading as _threading

import cv2
import numpy as np

from ..config.defaults import MASK_VALUE_KEEP, MASK_VALUE_REMOVE
from ..utils.fisheye_mask import build_fisheye_keep_mask, clamp_radius_percent, combine_with_keep_mask
from ..utils.mask_output import build_related_output_path, build_unique_staged_filename
from ..utils.sam3_backend import (
    SAM3_BACKEND_AUTO,
    SAM3_BACKEND_CPU,
    SAM3_BACKEND_CUDA,
    SAM3_BACKEND_VULKAN,
    inspect_sam3_executable_backend,
    normalize_sam3_backend_mode,
    sam3_backend_matches_mode,
    sam3_backend_supports_gpu,
)
from .mask_refinement import MaskRefinementSettings, refine_detected_mask

logger = logging.getLogger(__name__)


def _sam3_build_output_relative_candidates(filename: str) -> List[Path]:
    return [
        Path('bvv') / 'examples' / filename,
        Path('bvv') / 'examples' / 'Release' / filename,
        Path('build-cuda') / 'examples' / filename,
        Path('build-cuda') / 'examples' / 'Release' / filename,
        Path('build-vulkan-verified') / 'examples' / filename,
        Path('build-vulkan-verified') / 'examples' / 'Release' / filename,
        Path('build-vulkan') / 'examples' / filename,
        Path('build-vulkan') / 'examples' / 'Release' / filename,
        Path('build-vulkan-vs') / 'examples' / filename,
        Path('build-vulkan-vs') / 'examples' / 'Release' / filename,
        Path('build') / 'examples' / filename,
        Path('build') / 'examples' / 'Release' / filename,
    ]


def _read_image_unicode(path: Path, flags: int = cv2.IMREAD_UNCHANGED) -> Optional[np.ndarray]:
    """Read images robustly on Windows paths with non-ASCII characters."""
    # On Windows, prefer fromfile+imdecode first to avoid noisy warnings and
    # intermittent failures with non-ASCII path segments in cv2.imread.
    if os.name == 'nt':
        try:
            data = np.fromfile(str(path), dtype=np.uint8)
            if data.size > 0:
                decoded = cv2.imdecode(data, flags)
                if decoded is not None:
                    return decoded
        except Exception:
            pass
    try:
        img = cv2.imread(str(path), flags)
    except Exception:
        img = None
    if img is not None:
        return img
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def _write_image_unicode(path: Path, image: np.ndarray, params: Optional[List[int]] = None) -> bool:
    """Write images robustly on Windows paths with non-ASCII characters."""
    suffix = path.suffix.lower() or '.png'
    encode_ext = suffix if suffix in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'} else '.png'
    try:
        if params:
            ok, encoded = cv2.imencode(encode_ext, image, params)
        else:
            ok, encoded = cv2.imencode(encode_ext, image)
        if ok:
            encoded.tofile(str(path))
            return True
    except Exception:
        pass
    try:
        if params:
            return bool(cv2.imwrite(str(path), image, params))
        return bool(cv2.imwrite(str(path), image))
    except Exception:
        return False


def _subprocess_no_window_kwargs() -> dict:
    kwargs = {}
    if os.name == 'nt':
        creationflags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= getattr(subprocess, 'STARTF_USESHOWWINDOW', 0)
        kwargs['creationflags'] = creationflags
        kwargs['startupinfo'] = startupinfo
    return kwargs

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
        backend_mode: str = SAM3_BACKEND_AUTO,
        feather_radius: int = 0,
        morph_radius: int = 0,
        alpha_export: bool = False,
        alpha_only: bool = False,
        max_input_width: int = 0,
        score_threshold: float = 0.04,
        nms_threshold: float = 0.1,
        mask_logit_threshold: float = 0.75,
        enable_refinement: bool = True,
        refine_sky_only: bool = True,
        seam_aware_refinement: bool = True,
        edge_sharpen_strength: float = 0.75,
    ):
        self.segment_persons_exe = self._resolve_segmenter_path(Path(segment_persons_exe).expanduser())
        self.model_path = self._resolve_model_path(Path(model_path).expanduser())
        self.sam3_image_exe = Path(sam3_image_exe).expanduser() if sam3_image_exe else None
        self.use_gpu = use_gpu
        self.backend_mode = normalize_sam3_backend_mode(backend_mode, use_gpu=use_gpu)
        self.morph_radius = int(morph_radius)
        self.alpha_export = bool(alpha_export)
        self.alpha_only = bool(alpha_only)
        self.max_input_width = int(max_input_width)
        self.score_threshold = float(score_threshold)
        self.nms_threshold = float(nms_threshold)
        self.mask_logit_threshold = float(mask_logit_threshold)
        self.feather_radius = int(feather_radius)
        self.enable_refinement = bool(enable_refinement)
        self.refine_sky_only = bool(refine_sky_only)
        self.seam_aware_refinement = bool(seam_aware_refinement)
        self.edge_sharpen_strength = float(edge_sharpen_strength)
        self.cancelled = False
        self.fisheye_circle_mask_enabled = False
        self.fisheye_circle_mask_radius_percent = 94
        self._onnx_fallback_masker = None
        self._segmenter_backend_info = inspect_sam3_executable_backend(self.segment_persons_exe)
        self.effective_use_gpu = self.use_gpu and sam3_backend_supports_gpu(self._segmenter_backend_info.backend)

        self.enabled_categories: Dict[str, bool] = {
            'persons': True,
            'bags': False,
            'phones': False,
            'hats': False,
            'helmets': False,
            'sky': False,
        }
        self.custom_prompts: List[str] = []

        self._validate_runtime()

    def _sam3_search_roots(self) -> List[Path]:
        roots: List[Path] = []
        if getattr(sys, 'frozen', False):
            meipass = Path(getattr(sys, '_MEIPASS', Path('.')))
            exe_root = Path(sys.executable).resolve().parent
            roots.extend([
                meipass / 'sam3cpp',
                exe_root / '_internal' / 'sam3cpp',
            ])
        roots.append(Path(__file__).resolve().parents[2] / 'downloads' / 'sam3cpp')

        unique: List[Path] = []
        seen: set[str] = set()
        for root in roots:
            key = str(root)
            if key in seen:
                continue
            seen.add(key)
            unique.append(root)
        return unique

    def _sam3_executable_candidates(self, filename: str) -> List[Path]:
        candidates: List[Path] = []
        seen: set[str] = set()
        for root in self._sam3_search_roots():
            for relative in _sam3_build_output_relative_candidates(filename):
                candidate = root / relative
                key = str(candidate)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(candidate)
        return candidates

    def _set_active_segmenter(self, executable_path: Path) -> None:
        self.segment_persons_exe = executable_path.resolve()
        self._segmenter_backend_info = inspect_sam3_executable_backend(self.segment_persons_exe)
        self.effective_use_gpu = self.use_gpu and sam3_backend_supports_gpu(self._segmenter_backend_info.backend)
        if self.backend_mode == SAM3_BACKEND_CPU:
            self.effective_use_gpu = False

    @staticmethod
    def _is_backend_related_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return any(token in text for token in (
            'backend',
            'cuda',
            'vulkan',
            'no kernel image',
            'cublas',
            'cudart',
            'gpu mode requested',
            'segment_persons_pvs.exe not found',
        ))

    @staticmethod
    def _auto_fallback_rank(candidate_backend: str, failed_backend: str) -> int:
        if failed_backend == SAM3_BACKEND_CUDA:
            ranking = {SAM3_BACKEND_VULKAN: 0, 'unverified-vulkan': 1, SAM3_BACKEND_CPU: 2, SAM3_BACKEND_CUDA: 3}
        elif failed_backend in {SAM3_BACKEND_VULKAN, 'unverified-vulkan'}:
            ranking = {SAM3_BACKEND_CPU: 0, SAM3_BACKEND_CUDA: 1, SAM3_BACKEND_VULKAN: 2, 'unverified-vulkan': 3}
        else:
            ranking = {SAM3_BACKEND_CPU: 0, SAM3_BACKEND_VULKAN: 1, 'unverified-vulkan': 2, SAM3_BACKEND_CUDA: 3}
        return ranking.get(candidate_backend, 99)

    def _next_auto_fallback_segmenter(self, attempted: set[str]) -> Optional[Path]:
        failed_backend = self._segmenter_backend_info.backend
        ranked: List[Tuple[int, Path]] = []
        for candidate in self._sam3_executable_candidates('segment_persons.exe'):
            if not candidate.is_file():
                continue
            resolved = candidate.resolve()
            if str(resolved) in attempted:
                continue
            info = inspect_sam3_executable_backend(resolved)
            rank = self._auto_fallback_rank(info.backend, failed_backend)
            if rank >= 99:
                continue
            ranked.append((rank, resolved))
        if not ranked:
            return None
        ranked.sort(key=lambda item: (item[0], str(item[1]).lower()))
        return ranked[0][1]

    def _maybe_switch_auto_fallback(self, exc: Exception, attempted: set[str], label: str) -> bool:
        if self.backend_mode != SAM3_BACKEND_AUTO or not self._is_backend_related_error(exc):
            return False

        next_segmenter = self._next_auto_fallback_segmenter(attempted)
        if next_segmenter is None:
            return False

        next_info = inspect_sam3_executable_backend(next_segmenter)
        logger.warning(
            '[SAM3] Auto backend fallback after %s failed on %s (%s): %s. Retrying with %s (%s).',
            label,
            self.segment_persons_exe,
            self._segmenter_backend_info.backend,
            exc,
            next_segmenter,
            next_info.backend,
        )
        attempted.add(str(next_segmenter))
        self._set_active_segmenter(next_segmenter)
        return True

    def _resolve_segmenter_path(self, candidate: Path) -> Path:
        if candidate.is_file() and candidate.suffix.lower() == '.exe':
            return candidate.resolve()

        for path in self._sam3_executable_candidates('segment_persons.exe'):
            if path.is_file():
                return path
        return candidate

    def _resolve_model_path(self, candidate: Path) -> Path:
        if candidate.is_file() and candidate.suffix.lower() in {'.ggml', '.bin'}:
            return candidate.resolve()

        search: List[Path] = []
        if getattr(sys, 'frozen', False):
            meipass = Path(getattr(sys, '_MEIPASS', Path('.')))
            exe_root = Path(sys.executable).resolve().parent
            search.extend([
                meipass / 'sam3cpp' / 'models' / 'sam3-f16.ggml',
                meipass / 'sam3cpp' / 'models' / 'sam3-q8_0.ggml',
                meipass / 'sam3cpp' / 'models' / 'sam3-q4_1.ggml',
                meipass / 'sam3cpp' / 'models' / 'sam3-q4_0.ggml',
                exe_root / '_internal' / 'sam3cpp' / 'models' / 'sam3-f16.ggml',
                exe_root / '_internal' / 'sam3cpp' / 'models' / 'sam3-q8_0.ggml',
                exe_root / '_internal' / 'sam3cpp' / 'models' / 'sam3-q4_1.ggml',
                exe_root / '_internal' / 'sam3cpp' / 'models' / 'sam3-q4_0.ggml',
            ])

        project_root = Path(__file__).resolve().parents[2]
        search.extend([
            project_root / 'downloads' / 'sam3cpp' / 'models' / 'sam3-f16.ggml',
            project_root / 'downloads' / 'sam3cpp' / 'models' / 'sam3-q8_0.ggml',
            project_root / 'downloads' / 'sam3cpp' / 'models' / 'sam3-q4_1.ggml',
            project_root / 'downloads' / 'sam3cpp' / 'models' / 'sam3-q4_0.ggml',
        ])

        for path in search:
            if path.is_file():
                return path
        return candidate

    # ------------------------------------------------------------------
    # Validation / runtime helpers
    # ------------------------------------------------------------------

    def _validate_runtime(self) -> None:
        if not self.segment_persons_exe.is_file():
            raise FileNotFoundError(
                f"SAM3.cpp segmenter not found: {self.segment_persons_exe}"
            )
        if not self.model_path.is_file():
            raise FileNotFoundError(f"SAM3.cpp model not found: {self.model_path}")
        if self.sam3_image_exe and not self.sam3_image_exe.exists():
            logger.warning("SAM3.cpp interactive GUI not found: %s", self.sam3_image_exe)
        if not sam3_backend_matches_mode(
            self._segmenter_backend_info.backend,
            self.backend_mode,
            allow_unverified_vulkan=self.backend_mode == SAM3_BACKEND_VULKAN,
        ):
            raise RuntimeError(
                'SAM3 backend mode does not match the selected executable: '
                f'mode={self.backend_mode}, executable={self.segment_persons_exe}, '
                f'detected={self._segmenter_backend_info.backend} ({self._segmenter_backend_info.detail})'
            )
        if self.backend_mode == SAM3_BACKEND_CPU:
            self.effective_use_gpu = False
        elif self.use_gpu and self.backend_mode in {SAM3_BACKEND_CUDA, SAM3_BACKEND_VULKAN}:
            self.effective_use_gpu = True

    def _build_runtime_env(self, executable_path: Path) -> dict:
        env = os.environ.copy()
        search = []
        exe_dir = executable_path.parent
        if exe_dir.exists():
            search.append(str(exe_dir))
        build_root = exe_dir.parent.parent if exe_dir.name.lower() == 'release' else exe_dir.parent
        dll_candidates = [
            build_root / 'bin',
            build_root / 'bin' / exe_dir.name,
            build_root / 'bin' / 'Release',
        ]
        for dll_dir in dll_candidates:
            if dll_dir.exists():
                search.append(str(dll_dir))
        if search:
            existing = env.get('PATH', '')
            env['PATH'] = os.pathsep.join(search + ([existing] if existing else []))
        effective_logit_threshold = self._effective_mask_logit_threshold()
        env['SAM3_PCS_MASK_LOGIT_THRESHOLD'] = str(effective_logit_threshold)
        if self._segmenter_backend_info.backend in {SAM3_BACKEND_VULKAN, 'unverified-vulkan'}:
            env['SAM3_STRICT_COMPONENT_CLEANUP'] = '1'
        else:
            env['SAM3_STRICT_COMPONENT_CLEANUP'] = '0'
        logger.info(
            '[SAM3] Effective logit threshold %.3f (base=%.3f, backend=%s, sky=%s)',
            effective_logit_threshold,
            self.mask_logit_threshold,
            self._segmenter_backend_info.backend,
            bool(self.enabled_categories.get('sky', False)),
        )
        return env

    def _effective_mask_logit_threshold(self) -> float:
        """Apply the same threshold policy for CUDA and Vulkan backends."""
        threshold = float(self.mask_logit_threshold)
        backend = self._segmenter_backend_info.backend
        detail_prompts_enabled = any(
            bool(self.enabled_categories.get(key, False))
            for key in ('bags', 'phones', 'hats', 'helmets')
        )

        # Keep CUDA and Vulkan on identical effective values when detail prompts are enabled.
        if backend in {SAM3_BACKEND_CUDA, SAM3_BACKEND_VULKAN, 'unverified-vulkan'} and detail_prompts_enabled:
            threshold = max(0.55, threshold - 0.08)

        return threshold

    def _ensure_gpu_backend(self, executable_path: Path, label: str) -> None:
        if not self.effective_use_gpu:
            return

        info = inspect_sam3_executable_backend(executable_path)
        expected_mode = self.backend_mode
        if expected_mode == SAM3_BACKEND_AUTO and info.backend in {SAM3_BACKEND_VULKAN, SAM3_BACKEND_CUDA}:
            logger.info('[SAM3] GPU backend for %s: %s (%s)', label, executable_path, info.backend)
            return
        if sam3_backend_matches_mode(info.backend, expected_mode, allow_unverified_vulkan=expected_mode == SAM3_BACKEND_VULKAN):
            logger.info('[SAM3] Backend for %s matches requested mode %s: %s', label, expected_mode, executable_path)
            return

        raise RuntimeError(
            f'GPU mode requested, but {label} does not match requested backend mode {expected_mode}: '
            f'{executable_path} ({info.backend}: {info.detail})'
        )

    @staticmethod
    def _detect_runtime_backend(stderr_lines: List[str]) -> str:
        joined = '\n'.join(stderr_lines).lower()
        if 'using vulkan backend' in joined:
            return 'vulkan'
        if 'using cuda backend' in joined:
            return 'cuda'
        if 'using metal backend' in joined:
            return 'metal'
        if 'using cpu backend' in joined:
            return 'cpu'
        return 'unknown'

    def _ensure_runtime_backend(self, stderr_lines: List[str], executable_path: Path, label: str) -> None:
        if not self.effective_use_gpu:
            return

        runtime_backend = self._detect_runtime_backend(stderr_lines)
        if self.backend_mode == SAM3_BACKEND_AUTO and runtime_backend in {SAM3_BACKEND_VULKAN, SAM3_BACKEND_CUDA}:
            logger.info('[SAM3] Runtime backend confirmed as %s for %s: %s', runtime_backend, label, executable_path)
            return
        if runtime_backend == self.backend_mode:
            logger.info('[SAM3] Runtime backend confirmed as %s for %s: %s', runtime_backend, label, executable_path)
            return

        stderr_tail = '\n'.join(stderr_lines[-20:]) if stderr_lines else 'No stderr output captured'
        raise RuntimeError(
            f'GPU mode requested, but {label} did not activate the expected backend at runtime: '
            f'{executable_path} (runtime_backend={runtime_backend}).\n{stderr_tail}'
        )

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def set_enabled_categories(self, categories: Dict[str, bool]) -> None:
        if categories:
            self.enabled_categories.update(categories)

    def set_custom_prompts(self, text: str) -> None:
        self.custom_prompts = [t.strip() for t in text.split(',') if t.strip()]

    def set_fisheye_circle_mask(self, enabled: bool, radius_percent: int | float = 94) -> None:
        self.fisheye_circle_mask_enabled = bool(enabled)
        self.fisheye_circle_mask_radius_percent = clamp_radius_percent(radius_percent)

    def _apply_fisheye_circle_mask(self, mask: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
        if not self.fisheye_circle_mask_enabled:
            return mask
        keep_mask = build_fisheye_keep_mask(image_shape, self.fisheye_circle_mask_radius_percent)
        return combine_with_keep_mask(mask, keep_mask)

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
        self, image_path: Path, dest_dir: Path, output_name: Optional[str] = None
    ) -> Tuple[Path, int, int]:
        """Get image dimensions efficiently without loading full image.
        
        Only loads the image if we need to downscale it. Otherwise,
        just reads dimensions from headers (PIL) or using ffprobe.
        """
        # Try to get dimensions without loading full image
        w, h = 0, 0
        
        # Try PIL first (fast, reads header only)
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                w, h = img.size  # (width, height)
        except Exception:
            pass
        
        # Fallback: load with OpenCV to get dimensions
        if w == 0 or h == 0:
            img = _read_image_unicode(image_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise RuntimeError(f"Cannot load image: {image_path}")
            h_cv, w_cv = img.shape[:2]
            if w == 0: w = w_cv
            if h == 0: h = h_cv
            del img  # Free memory immediately
        
        # Check if downscaling needed
        if self.max_input_width > 0 and w > self.max_input_width:
            # Now load and downscale
            img = _read_image_unicode(image_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise RuntimeError(f"Cannot load image: {image_path}")
            scale = self.max_input_width / w
            new_w = self.max_input_width
            new_h = max(1, int(h * scale))
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            dest = dest_dir / (output_name or image_path.name)
            if not _write_image_unicode(dest, resized):
                raise RuntimeError(f"Cannot save resized image: {dest}")
            return dest, w, h

        if output_name and output_name != image_path.name:
            dest = dest_dir / output_name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, dest)
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
        _write_image_unicode(dest_path, blended)

    def _rs_mask_has_detections(self, rs_mask: Optional[np.ndarray]) -> bool:
        return bool(rs_mask is not None and np.any(rs_mask == MASK_VALUE_REMOVE))

    def _write_rs_mask_outputs(
        self,
        rs_mask: np.ndarray,
        orig_img: Optional[np.ndarray],
        orig_image_path: Path,
        output_dir: Path,
        input_root: Optional[Path] = None,
    ) -> Path:
        final_mask_path = build_related_output_path(
            orig_image_path,
            input_root or orig_image_path.parent,
            output_dir,
            '_mask.png',
        )
        final_mask_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.alpha_only:
            _write_image_unicode(final_mask_path, rs_mask)

        if self.alpha_export and orig_img is not None:
            export_img = orig_img
            if export_img.ndim == 2:
                export_img = cv2.cvtColor(export_img, cv2.COLOR_GRAY2BGR)
            elif export_img.ndim == 3 and export_img.shape[2] == 4:
                export_img = cv2.cvtColor(export_img, cv2.COLOR_BGRA2BGR)
            rgba = cv2.cvtColor(export_img, cv2.COLOR_BGR2BGRA)
            rgba[:, :, 3] = rs_mask
            alpha_dest = build_related_output_path(
                orig_image_path,
                input_root or orig_image_path.parent,
                output_dir,
                '_alpha.png',
            )
            alpha_dest.parent.mkdir(parents=True, exist_ok=True)
            _write_image_unicode(alpha_dest, rgba)

        return final_mask_path

    def _generate_onnx_fallback_mask(self, orig_image_path: Path) -> Optional[np.ndarray]:
        persons_enabled = bool(self.enabled_categories.get('persons', False))
        objects_enabled = any(self.enabled_categories.get(key, False) for key in ('bags', 'phones', 'hats', 'helmets'))
        if not persons_enabled and not objects_enabled:
            return None

        try:
            if self._onnx_fallback_masker is None:
                from ..utils.dependency_provisioning import resolve_masking_model_path
                from .onnx_masker import ONNXMasker

                model_path = None
                for model_name in ('yolo26s-seg.onnx', 'yolov8s-seg.onnx', 'yolov8n-seg.onnx', 'yolov8m-seg.onnx'):
                    candidate = resolve_masking_model_path(model_name)
                    if candidate.exists():
                        model_path = candidate
                        break

                if model_path is None:
                    logger.warning('[SAM3] ONNX fallback requested but no ONNX model was found')
                    return None

                self._onnx_fallback_masker = ONNXMasker(
                    model_path=str(model_path),
                    confidence_threshold=min(self.score_threshold, 0.25),
                    use_gpu=self.use_gpu,
                )

            self._onnx_fallback_masker.set_confidence_threshold(min(self.score_threshold, 0.25))
            self._onnx_fallback_masker.set_enabled_categories(
                persons=persons_enabled,
                personal_objects=objects_enabled,
                animals=False,
            )
            if hasattr(self._onnx_fallback_masker, 'set_fisheye_circle_mask'):
                self._onnx_fallback_masker.set_fisheye_circle_mask(
                    self.fisheye_circle_mask_enabled,
                    self.fisheye_circle_mask_radius_percent,
                )

            image = _read_image_unicode(orig_image_path, cv2.IMREAD_COLOR)
            if image is None:
                logger.warning('[SAM3] ONNX fallback could not read image: %s', orig_image_path)
                return None

            fallback_mask = self._onnx_fallback_masker.generate_mask_from_array(image)
            if not self._rs_mask_has_detections(fallback_mask):
                return None

            logger.info('[SAM3] ONNX fallback produced detections for %s', orig_image_path.name)
            return fallback_mask
        except Exception as exc:
            logger.warning('[SAM3] ONNX fallback failed for %s: %s', orig_image_path.name, exc)
            return None

    def _postprocess_mask(
        self,
        raw_mask_path: Path,
        orig_w: int,
        orig_h: int,
        orig_image_path: Path,
        output_dir: Path,
        input_root: Optional[Path] = None,
    ) -> Tuple[Path, np.ndarray]:
        raw = _read_image_unicode(raw_mask_path, cv2.IMREAD_GRAYSCALE)
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

        orig_img = _read_image_unicode(orig_image_path, cv2.IMREAD_UNCHANGED)
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
        rs_mask = self._apply_fisheye_circle_mask(rs_mask, (orig_h, orig_w))

        final_mask_path = self._write_rs_mask_outputs(
            rs_mask,
            orig_img,
            orig_image_path,
            output_dir,
            input_root,
        )

        return final_mask_path, rs_mask

    def _write_fisheye_circle_only_output(self, orig_image_path: Path, output_dir: Path, input_root: Optional[Path] = None) -> Optional[Path]:
        image = _read_image_unicode(orig_image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            return None

        keep_mask = build_fisheye_keep_mask(image.shape, self.fisheye_circle_mask_radius_percent)
        if keep_mask is None:
            return None

        final_mask_path = build_related_output_path(
            orig_image_path,
            input_root or orig_image_path.parent,
            output_dir,
            '_mask.png',
        )
        final_mask_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.alpha_only:
            _write_image_unicode(final_mask_path, keep_mask)

        if self.alpha_export:
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.ndim == 3 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
            rgba[:, :, 3] = keep_mask
            alpha_path = build_related_output_path(
                orig_image_path,
                input_root or orig_image_path.parent,
                output_dir,
                '_alpha.png',
            )
            alpha_path.parent.mkdir(parents=True, exist_ok=True)
            _write_image_unicode(alpha_path, rgba)

        return final_mask_path

    # ------------------------------------------------------------------
    # Core: run segment_persons.exe (batch)
    # ------------------------------------------------------------------

    def _run_batch_exe_once(
        self,
        executable_path: Path,
        image_paths: List[Path],
        raw_mask_dir: Path,
        progress_callback: Optional[Callable] = None,
        cancellation_check: Optional[Callable] = None,
    ) -> Dict[str, bool]:
        logger.info("[SAM3._run_batch_exe] [START] ENTRY: %d images, output=%s", len(image_paths), raw_mask_dir)
        prompts = self._build_prompts()
        logger.info("[SAM3._run_batch_exe] Prompts: %s", prompts)

        processed: Dict[str, bool] = {}
        list_fd, list_path = tempfile.mkstemp(suffix='.txt', prefix='sam3_list_')
        logger.info("[SAM3._run_batch_exe] Image list file: %s", list_path)
        try:
            safe_encoding = 'utf-8' if os.name != 'nt' else 'cp1252'

            with os.fdopen(list_fd, 'w', encoding=safe_encoding) as f:
                for p in image_paths:
                    path_str = str(p)
                    if os.name == 'nt' and any(ord(ch) > 127 for ch in path_str):
                        try:
                            import ctypes
                            GetShortPathName = ctypes.windll.kernel32.GetShortPathNameW
                            short_path = ctypes.create_unicode_buffer(260)
                            result = GetShortPathName(path_str, short_path, 260)
                            if result:
                                path_str = short_path.value
                                logger.debug("[SAM3._run_batch_exe] Using short path: %s -> %s", p.name, path_str)
                        except Exception as e:
                            logger.debug("[SAM3._run_batch_exe] Could not convert to short path: %s", e)
                    f.write(path_str + '\n')
            logger.info("[SAM3._run_batch_exe] Wrote %d image paths to list file", len(image_paths))

            cmd = [
                str(executable_path),
                '--model',      str(self.model_path),
                '--image-list', list_path,
                '--output-dir', str(raw_mask_dir),
                '--prompts',    ','.join(prompts),
                '--score',      str(self.score_threshold),
                '--nms',        str(self.nms_threshold),
            ]
            if self.effective_use_gpu:
                self._ensure_gpu_backend(executable_path, 'segment_persons.exe')
                cmd.append('--gpu')
            else:
                cmd.append('--no-gpu')

            logger.info("[SAM3._run_batch_exe] Command: %s", ' '.join(cmd))
            logger.info("[SAM3._run_batch_exe] Exe exists: %s", executable_path.exists())
            logger.info("[SAM3._run_batch_exe] Model exists: %s", self.model_path.exists())
            env = self._build_runtime_env(executable_path)

            logger.info("[SAM3._run_batch_exe] [START] Launching subprocess...")
            proc = subprocess.Popen(
                cmd,
                cwd=str(raw_mask_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                **_subprocess_no_window_kwargs(),
            )
            logger.info("[SAM3._run_batch_exe] Subprocess PID: %d", proc.pid)

            total = len(image_paths)
            stderr_lines: List[str] = []

            def _drain_stderr():
                for ln in proc.stderr:
                    stderr_lines.append(ln.rstrip())

            stderr_thread = _threading.Thread(target=_drain_stderr, daemon=True)
            stderr_thread.start()
            logger.info("[SAM3._run_batch_exe] Started stderr drain thread")

            logger.info("[SAM3._run_batch_exe] [START] Reading stdout from subprocess...")
            for line in proc.stdout:
                line = line.rstrip()
                if line.startswith('PROCESSED '):
                    parts = line.split()
                    stem = parts[1] if len(parts) > 1 else '?'
                    dets = int(parts[6]) if len(parts) > 6 else -1
                    processed[stem] = True
                    logger.debug("[SAM3._run_batch_exe] PROCESSED: %s (detections=%d)", stem, dets)
                    if progress_callback:
                        progress_callback(len(processed), total, f"SAM3: {stem}")
                elif line:
                    logger.debug("[SAM3._run_batch_exe] OUTPUT: %s", line)

                if cancellation_check and cancellation_check():
                    logger.warning("[SAM3._run_batch_exe] Cancellation requested, terminating subprocess")
                    proc.terminate()
                    self.cancelled = True
                    break

            logger.info("[SAM3._run_batch_exe] Waiting for subprocess to complete...")
            proc.wait()
            logger.info("[SAM3._run_batch_exe] Subprocess exited with code: %d", proc.returncode)
            stderr_thread.join(timeout=5)
            self._ensure_runtime_backend(stderr_lines, executable_path, 'segment_persons.exe')
            stderr_tail = '\n'.join(stderr_lines[-20:]) if stderr_lines else ''
            if proc.returncode not in (0, 5):
                logger.error("[SAM3._run_batch_exe] Subprocess stderr: %s", stderr_tail)
                raise RuntimeError(
                    f"segment_persons.exe exited {proc.returncode}.\n{stderr_tail}"
                )
        finally:
            try:
                os.unlink(list_path)
            except OSError:
                pass

            logger.info("[SAM3._run_batch_exe] [OK] Completed: %d images processed", len(processed))
        return processed

    def _run_batch_exe(
        self,
        image_paths: List[Path],
        raw_mask_dir: Path,
        progress_callback: Optional[Callable] = None,
        cancellation_check: Optional[Callable] = None,
    ) -> Dict[str, bool]:
        attempted = {str(self.segment_persons_exe.resolve())}
        while True:
            try:
                return self._run_batch_exe_once(
                    self.segment_persons_exe,
                    image_paths,
                    raw_mask_dir,
                    progress_callback=progress_callback,
                    cancellation_check=cancellation_check,
                )
            except Exception as exc:
                if not self._maybe_switch_auto_fallback(exc, attempted, 'segment_persons.exe'):
                    raise

    # ------------------------------------------------------------------
    # PVS helpers: equirectangular 360° images (YOLO boxes → SAM3 PVS)
    # ------------------------------------------------------------------

    @staticmethod
    def _is_equirectangular(width: int, height: int) -> bool:
        """Return True if the image looks like a 2:1 equirectangular panorama."""
        return height > 0 and (width / height) >= 1.8

    def _resolve_pvs_segmenter_path(self) -> Optional[Path]:
        """Find segment_persons_pvs.exe (sibling of segment_persons.exe)."""
        pcs_dir = self.segment_persons_exe.parent
        pvs_candidate = pcs_dir / 'segment_persons_pvs.exe'
        if pvs_candidate.is_file():
            return pvs_candidate

        for path in self._sam3_executable_candidates('segment_persons_pvs.exe'):
            if path.is_file():
                return path
        return None

    def _get_pvs_onnx_detector(self):
        """Lazily create the ONNX box detector used for PVS prompt generation."""
        if self._onnx_fallback_masker is None:
            try:
                from ..utils.dependency_provisioning import resolve_masking_model_path
                from .onnx_masker import ONNXMasker

                model_path = None
                for model_name in ('yolo26s-seg.onnx', 'yolov8s-seg.onnx', 'yolov8n-seg.onnx', 'yolov8m-seg.onnx'):
                    candidate = resolve_masking_model_path(model_name)
                    if candidate.exists():
                        model_path = candidate
                        break

                if model_path is None:
                    logger.warning('[SAM3-PVS] No ONNX model found for box detection')
                    return None

                self._onnx_fallback_masker = ONNXMasker(
                    model_path=str(model_path),
                    confidence_threshold=0.25,
                    use_gpu=self.use_gpu,
                )
                logger.info('[SAM3-PVS] ONNX box detector initialized: %s', model_path.name)
            except Exception as exc:
                logger.warning('[SAM3-PVS] Failed to init ONNX box detector: %s', exc)
                return None

        persons_enabled = bool(self.enabled_categories.get('persons', False))
        objects_enabled = any(
            self.enabled_categories.get(key, False)
            for key in ('bags', 'phones', 'hats', 'helmets')
        )
        self._onnx_fallback_masker.set_confidence_threshold(0.25)
        self._onnx_fallback_masker.set_enabled_categories(
            persons=persons_enabled,
            personal_objects=objects_enabled,
            animals=False,
        )
        return self._onnx_fallback_masker

    def _run_pvs_batch_exe_once(
        self,
        pvs_exe: Path,
        image_paths: List[Path],
        raw_mask_dir: Path,
        progress_callback: Optional[Callable] = None,
        cancellation_check: Optional[Callable] = None,
    ) -> Dict[str, bool]:
        """Run SAM3 PVS segmentation using YOLO bounding boxes as visual prompts.

        Correct path for equirectangular (360°) panoramic images where SAM3 PCS
        (text/CLIP) fails due to geometric distortion — presence_logit ≈ 12%.
        PVS bypasses CLIP and uses the bounding box as a spatial anchor.
        """
        logger.info("[SAM3-PVS] Running for %d images", len(image_paths))
        onnx_detector = self._get_pvs_onnx_detector()
        boxes_by_stem: Dict[str, List[Tuple[int, int, int, int]]] = {}

        for img_path in image_paths:
            img = _read_image_unicode(img_path, cv2.IMREAD_COLOR)
            if img is None:
                logger.warning('[SAM3-PVS] Cannot load %s, skipping box detection', img_path.name)
                boxes_by_stem[img_path.stem] = []
                continue

            boxes: List[Tuple[int, int, int, int]] = []
            if onnx_detector is not None:
                try:
                    boxes = onnx_detector.detect_boxes(img)
                    logger.debug('[SAM3-PVS] YOLO detected %d box(es) in %s', len(boxes), img_path.name)
                except Exception as exc:
                    logger.warning('[SAM3-PVS] YOLO detection failed for %s: %s', img_path.name, exc)

            if not boxes:
                h, w = img.shape[:2]
                x0 = int(w * 0.2); y0 = int(h * 0.3)
                x1 = int(w * 0.8); y1 = int(h * 0.9)
                boxes = [(x0, y0, x1, y1)]
                logger.debug('[SAM3-PVS] No YOLO boxes for %s, fallback centre box used', img_path.name)

            boxes_by_stem[img_path.stem] = boxes

        boxes_fd, boxes_tsv = tempfile.mkstemp(suffix='.tsv', prefix='sam3_pvs_boxes_')
        list_fd,  list_path = tempfile.mkstemp(suffix='.txt', prefix='sam3_list_')
        try:
            with os.fdopen(boxes_fd, 'w', encoding='utf-8') as f:
                for stem, boxes in boxes_by_stem.items():
                    for (x0, y0, x1, y1) in boxes:
                        f.write(f"{stem}\t{x0}\t{y0}\t{x1}\t{y1}\n")

            safe_encoding = 'utf-8' if os.name != 'nt' else 'cp1252'
            with os.fdopen(list_fd, 'w', encoding=safe_encoding) as f:
                for p in image_paths:
                    path_str = str(p)
                    if os.name == 'nt' and any(ord(ch) > 127 for ch in path_str):
                        try:
                            import ctypes
                            GetShortPathName = ctypes.windll.kernel32.GetShortPathNameW
                            short_path = ctypes.create_unicode_buffer(260)
                            result = GetShortPathName(path_str, short_path, 260)
                            if result:
                                path_str = short_path.value
                        except Exception:
                            pass
                    f.write(path_str + '\n')

            cmd = [
                str(pvs_exe),
                '--model',      str(self.model_path),
                '--image-list', list_path,
                '--boxes-file', boxes_tsv,
                '--output-dir', str(raw_mask_dir),
            ]
            if self.effective_use_gpu:
                self._ensure_gpu_backend(pvs_exe, 'segment_persons_pvs.exe')
                cmd.append('--gpu')
            else:
                cmd.append('--no-gpu')

            logger.info("[SAM3-PVS] Command: %s", ' '.join(cmd))
            env = self._build_runtime_env(pvs_exe)

            proc = subprocess.Popen(
                cmd,
                cwd=str(raw_mask_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                **_subprocess_no_window_kwargs(),
            )

            total = len(image_paths)
            processed: Dict[str, bool] = {}
            stderr_lines: List[str] = []

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
                    logger.debug("[SAM3-PVS] PROCESSED: %s (detections=%d)", stem, dets)
                    if progress_callback:
                        progress_callback(len(processed), total, f"SAM3-PVS: {stem}")
                elif line:
                    logger.debug("[SAM3-PVS] OUTPUT: %s", line)

                if cancellation_check and cancellation_check():
                    proc.terminate()
                    self.cancelled = True
                    break

            proc.wait()
            stderr_thread.join(timeout=5)
            self._ensure_runtime_backend(stderr_lines, pvs_exe, 'segment_persons_pvs.exe')
            stderr_tail = '\n'.join(stderr_lines[-20:]) if stderr_lines else ''
            if proc.returncode not in (0, 5):
                logger.error("[SAM3-PVS] Subprocess stderr: %s", stderr_tail)
                raise RuntimeError(
                    f"segment_persons_pvs.exe exited {proc.returncode}.\n{stderr_tail}"
                )
        finally:
            try:
                os.unlink(boxes_tsv)
            except OSError:
                pass
            try:
                os.unlink(list_path)
            except OSError:
                pass

        return processed

    def _run_pvs_batch_exe(
        self,
        image_paths: List[Path],
        raw_mask_dir: Path,
        progress_callback: Optional[Callable] = None,
        cancellation_check: Optional[Callable] = None,
    ) -> Dict[str, bool]:
        attempted = {str(self.segment_persons_exe.resolve())}
        while True:
            pvs_exe = self._resolve_pvs_segmenter_path()
            if pvs_exe is None or not pvs_exe.is_file():
                exc = FileNotFoundError(
                    'segment_persons_pvs.exe not found — rebuild sam3cpp with the PVS target'
                )
                if not self._maybe_switch_auto_fallback(exc, attempted, 'segment_persons_pvs.exe'):
                    raise exc
                continue
            try:
                return self._run_pvs_batch_exe_once(
                    pvs_exe,
                    image_paths,
                    raw_mask_dir,
                    progress_callback=progress_callback,
                    cancellation_check=cancellation_check,
                )
            except Exception as exc:
                if not self._maybe_switch_auto_fallback(exc, attempted, 'segment_persons_pvs.exe'):
                    raise

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

        # When the user selects SAM3, keep the masking path SAM3-only.
        # Do not silently switch 2:1 panoramas to the YOLO-backed PVS flow.
        self._run_batch_exe([proc_path], raw_dir)
        mask_source = 'sam3'

        raw_mask = raw_dir / f'{proc_path.stem}_mask.png'
        if not raw_mask.exists():
            if self.fisheye_circle_mask_enabled:
                final_mask_path = self._write_fisheye_circle_only_output(
                    image_path, resolved, image_path.parent
                )
                if final_mask_path is None:
                    raise RuntimeError("SAM3.cpp finished without producing a mask file.")
                final_data = _read_image_unicode(final_mask_path, cv2.IMREAD_GRAYSCALE)
                mask_source = 'fisheye-circle'
            else:
                raise RuntimeError("SAM3.cpp finished without producing a mask file.")
        else:
            final_mask_path, final_data = self._postprocess_mask(
                raw_mask, orig_w, orig_h, image_path, resolved, image_path.parent
            )

        has_detections = self._rs_mask_has_detections(final_data)

        overlay_path = resolved / f'{image_path.stem}_persons_overlay.png'
        if final_data is None:
            final_data = _read_image_unicode(raw_mask, cv2.IMREAD_GRAYSCALE)
            if final_data is not None:
                _, final_data = cv2.threshold(final_data, 127, 255, cv2.THRESH_BINARY)
                final_data = cv2.bitwise_not(final_data)
        if final_data is not None:
            orig_img = _read_image_unicode(image_path, cv2.IMREAD_UNCHANGED)
            if orig_img is not None:
                if final_data.shape[1] != orig_w or final_data.shape[0] != orig_h:
                    final_data = cv2.resize(
                        final_data, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
                    )
                self._save_overlay(orig_img, final_data, overlay_path)

        if has_detections:
            logger.info('[SAM3] Preview mask contains detections for %s (source=%s)',
                        image_path.name, mask_source)
        else:
            logger.info('[SAM3] Preview mask is empty for %s (all keep / no detections)', image_path.name)

        return {
            'image_path':   str(image_path),
            'overlay_path': str(overlay_path) if overlay_path.exists() else '',
            'mask_path':    str(final_mask_path),
            'work_dir':     str(resolved),
            'has_detections': str(has_detections),
            'mask_source':  mask_source,
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
        mask = _read_image_unicode(Path(preview['mask_path']), cv2.IMREAD_GRAYSCALE)
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
        logger.info("[SAM3] [START] process_batch() ENTRY: input=%s", input_dir)
        self.reset_cancel()

        input_path  = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info("[SAM3] Created output directory: %s", output_path)

        logger.info("[SAM3] Collecting images from: %s", input_path)
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
        logger.info("[SAM3] Found %d image file(s) to process", total)
        if total == 0:
            logger.warning("[SAM3] No images found in input directory")
            return {
                'successful': 0, 'failed': 0, 'skipped': 0,
                'total': 0, 'masks_created': 0,
            }

        todo: List[Path] = []
        skipped = 0
        logger.info("[SAM3] Filtering already-processed images...")
        stem_counts: Dict[str, int] = {}
        for img in image_files:
            stem_counts[img.stem] = stem_counts.get(img.stem, 0) + 1
        for img in image_files:
            # In alpha_only mode no _mask.png is written, so check for _alpha.png
            done_path = build_related_output_path(
                img,
                input_path,
                output_path,
                '_alpha.png' if self.alpha_only else '_mask.png',
            )
            if done_path.exists():
                skipped += 1
            else:
                todo.append(img)

        logger.info("[SAM3] Filtered: %d todo, %d already-done/skipped", len(todo), skipped)
        if not todo:
            logger.info("[SAM3] All images already processed, returning early")
            return {
                'successful': 0, 'failed': 0, 'skipped': skipped,
                'total': total, 'masks_created': 0,
            }

        logger.info("[SAM3] Creating temp directories...")
        raw_dir    = output_path / '_sam3_raw';    raw_dir.mkdir(exist_ok=True)
        scaled_dir = output_path / '_sam3_scaled'; scaled_dir.mkdir(exist_ok=True)
        logger.info("[SAM3] Raw dir: %s", raw_dir)
        logger.info("[SAM3] Scaled dir: %s", scaled_dir)

        orig_dims: Dict[str, Tuple[int, int]] = {}
        proc_paths: List[Path] = []
        proc_to_orig: Dict[str, Path] = {}

        logger.info("[SAM3] Downscaling %d images...", len(todo))
        for idx, img in enumerate(todo):
            if self.cancelled or (cancellation_check and cancellation_check()):
                logger.warning("[SAM3] Cancelled during downscaling at image %d/%d", idx, len(todo))
                break
            if idx % 100 == 0:
                logger.info("[SAM3] Downscaling progress: %d/%d (%.1f%%)", idx, len(todo), 100.0*idx/len(todo) if len(todo) > 0 else 0)
            staged_name = build_unique_staged_filename(img, input_path) if stem_counts.get(img.stem, 0) > 1 else None
            proc_p, ow, oh = self._maybe_downscale(img, scaled_dir, output_name=staged_name)
            orig_dims[proc_p.stem]  = (ow, oh)
            proc_to_orig[proc_p.stem] = img
            proc_paths.append(proc_p)

        logger.info("[SAM3] Downscaled %d images ready for batch", len(proc_paths))
        if self.cancelled or (cancellation_check and cancellation_check()):
            logger.warning("[SAM3] Cancelled after downscaling, cleaning up...")
            shutil.rmtree(raw_dir, ignore_errors=True)
            shutil.rmtree(scaled_dir, ignore_errors=True)
            return {
                'successful': 0,
                'failed': 0,
                'skipped': skipped,
                'total': total,
                'masks_created': 0,
                'cancelled': True,
                'error': 'Cancelled by user',
            }

        if not proc_paths:
            logger.warning("[SAM3] No images to process after downscaling")
            shutil.rmtree(raw_dir, ignore_errors=True)
            shutil.rmtree(scaled_dir, ignore_errors=True)
            return {
                'successful': 0,
                'failed': 0,
                'skipped': skipped,
                'total': total,
                'masks_created': 0,
            }

        successful = 0
        failed = 0

        runner_name = '_run_batch_exe'
        logger.info("[SAM3] [START] Calling %s() with %d images...", runner_name, len(proc_paths))
        try:
            result = self._run_batch_exe(
                proc_paths, raw_dir,
                progress_callback=progress_callback,
                cancellation_check=cancellation_check,
            )
            logger.info("[SAM3] [OK] %s() completed: %d images processed", runner_name, len(result))
        except Exception as exc:
            logger.error("[SAM3] ❌ Batch exe FAILED with exception: %s", exc, exc_info=True)
            shutil.rmtree(raw_dir, ignore_errors=True)
            shutil.rmtree(scaled_dir, ignore_errors=True)
            return {
                'successful': 0, 'failed': len(todo), 'skipped': skipped,
                'total': total, 'masks_created': 0,
            }

        logger.info("[SAM3] Post-processing %d masks...", len(proc_to_orig))
        if self.cancelled or (cancellation_check and cancellation_check()):
            logger.warning("[SAM3] Cancelled after batch exe, returning early")
            shutil.rmtree(raw_dir, ignore_errors=True)
            shutil.rmtree(scaled_dir, ignore_errors=True)
            return {
                'successful': 0,
                'failed': 0,
                'skipped': skipped,
                'total': total,
                'masks_created': 0,
                'cancelled': True,
                'error': 'Cancelled by user',
            }

        for proc_stem, orig_img in proc_to_orig.items():
            if self.cancelled or (cancellation_check and cancellation_check()):
                logger.warning("[SAM3] Cancelled during post-processing")
                break
            raw_mask = raw_dir / f'{proc_stem}_mask.png'
            ow, oh   = orig_dims.get(proc_stem, (0, 0))
            if not raw_mask.exists():
                if self.fisheye_circle_mask_enabled:
                    fallback = self._write_fisheye_circle_only_output(orig_img, output_path, input_path)
                    if fallback is not None:
                        logger.info("[SAM3] No prompt detections for %s; wrote fisheye circle-only mask", orig_img.name)
                        successful += 1
                        continue
                logger.warning("[SAM3] No mask for %s", orig_img.name)
                failed += 1
                continue
            if ow == 0:
                tmp = _read_image_unicode(orig_img, cv2.IMREAD_UNCHANGED)
                if tmp is not None:
                    oh, ow = tmp.shape[:2]
            try:
                final_mask_path, rs_mask = self._postprocess_mask(raw_mask, ow, oh, orig_img, output_path, input_path)
                if save_visualization:
                    vis_dest = build_related_output_path(orig_img, input_path, output_path, '_vis.png')
                    vis_dest.parent.mkdir(parents=True, exist_ok=True)
                    base_img = _read_image_unicode(orig_img, cv2.IMREAD_UNCHANGED)
                    if base_img is not None:
                        self._save_overlay(base_img, rs_mask, vis_dest)
                successful += 1
            except Exception as exc:
                logger.error("[SAM3] Post-process failed for %s: %s", orig_img.name, exc)
                failed += 1

        shutil.rmtree(raw_dir, ignore_errors=True)
        shutil.rmtree(scaled_dir, ignore_errors=True)

        if self.cancelled or (cancellation_check and cancellation_check()):
            logger.info("[SAM3] ✋ CANCELLED during post-processing: %d successful, %d failed", successful, failed)
            return {
                'successful': successful,
                'failed': failed,
                'skipped': skipped,
                'total': total,
                'masks_created': successful,
                'cancelled': True,
                'error': 'Cancelled by user',
            }

        logger.info("[SAM3] [OK] COMPLETE: successful=%d, failed=%d, skipped=%d, total=%d",
                   successful, failed, skipped, total)
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
            **_subprocess_no_window_kwargs(),
        )
