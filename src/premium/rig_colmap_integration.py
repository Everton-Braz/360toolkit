
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Callable, List, Tuple, Any
import numpy as np
import cv2
from PIL import Image
# pycolmap is imported lazily inside methods to avoid DLL crash at module load
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import json
import shutil
import subprocess
import os
import sys
import tempfile
import re
import urllib.request
import sqlite3
import shlex

from src.utils.dependency_provisioning import ensure_colmap_downloaded
from src.utils.colmap_paths import build_colmap_cli_context, get_colmap_runtime_dirs, normalize_colmap_executable, resolve_default_colmap_path

logger = logging.getLogger(__name__)

_COLMAP_LEARNED_MODEL_URLS = {
    "aliked-n16rot.onnx": "https://github.com/colmap/colmap/releases/download/3.13.0/aliked-n16rot.onnx",
    "aliked-n32.onnx": "https://github.com/colmap/colmap/releases/download/3.13.0/aliked-n32.onnx",
    "aliked-lightglue.onnx": "https://github.com/colmap/colmap/releases/download/3.13.0/aliked-lightglue.onnx",
    "sift-lightglue.onnx": "https://github.com/colmap/colmap/releases/download/3.13.0/sift-lightglue.onnx",
}

# GPU probe cache: None = not tested, True = works, False = doesn't work
_colmap_gpu_probe_result: Optional[bool] = None
_colmap_learned_crash_cache: Optional[Dict[str, Dict[str, Any]]] = None
_windows_dll_dir_handles: List[Any] = []


def _get_colmap_learned_crash_cache_path() -> Path:
    return Path.home() / ".360toolkit" / "colmap_learned_crash_cache.json"


def _load_colmap_learned_crash_cache() -> Dict[str, Dict[str, Any]]:
    global _colmap_learned_crash_cache
    if _colmap_learned_crash_cache is not None:
        return _colmap_learned_crash_cache

    cache_path = _get_colmap_learned_crash_cache_path()
    try:
        if cache_path.exists():
            with cache_path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
            if isinstance(raw, dict):
                _colmap_learned_crash_cache = {
                    str(key): value for key, value in raw.items() if isinstance(value, dict)
                }
            else:
                _colmap_learned_crash_cache = {}
        else:
            _colmap_learned_crash_cache = {}
    except Exception:
        _colmap_learned_crash_cache = {}
    return _colmap_learned_crash_cache


def _save_colmap_learned_crash_cache(cache: Dict[str, Dict[str, Any]]) -> None:
    cache_path = _get_colmap_learned_crash_cache_path()
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as handle:
            json.dump(cache, handle, indent=2)
    except Exception as exc:
        logger.warning("[RigSfM] Failed to persist COLMAP learned crash cache: %s", exc)

def _import_pycolmap():
    """Lazy import of pycolmap to avoid DLL crash at module-level."""
    import pycolmap
    return pycolmap


def _split_cli_flags(flags: str) -> List[str]:
    raw = str(flags or '').strip()
    if not raw:
        return []
    try:
        return shlex.split(raw, posix=False)
    except ValueError:
        return raw.split()


def _register_windows_dll_dirs(dll_dirs: List[Path]) -> None:
    """Keep add_dll_directory handles alive for the process lifetime on Windows."""
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return

    current_path = os.environ.get("PATH", "")
    prepend_path: List[str] = []
    existing_registered = {str(getattr(handle, "path", "")) for handle in _windows_dll_dir_handles}

    for dll_dir in dll_dirs:
        if not dll_dir.exists():
            continue
        dll_dir_str = str(dll_dir)
        if dll_dir_str not in existing_registered:
            try:
                handle = os.add_dll_directory(dll_dir_str)
            except Exception:
                handle = None
            if handle is not None:
                _windows_dll_dir_handles.append(handle)
                existing_registered.add(dll_dir_str)
        if dll_dir_str not in current_path:
            prepend_path.append(dll_dir_str)

    if prepend_path:
        os.environ["PATH"] = os.pathsep.join(prepend_path + [current_path])


def _resolve_runtime_path(path_value: Path | str) -> Path:
    return Path(path_value).expanduser().resolve()

class RigColmapIntegrator:
    """
    Implements the Rig-based SfM approach for 360 videos.
    Projects 360 images into virtual perspective cameras configured in a rig,
    then runs COLMAP on these perspective images.
    """

    def __init__(self, settings):
        self.settings = settings
        self._cli_help_cache: Dict[Tuple[str, str], str] = {}

    def _is_global_backend(self, mapping_backend: str) -> bool:
        return str(mapping_backend).strip().lower() in {"glomap", "global", "global_mapper", "colmap_global"}

    def _colmap_binary_cache_key(self, colmap_bin: str) -> str:
        try:
            return str(Path(colmap_bin).resolve()).lower()
        except Exception:
            return str(colmap_bin).strip().lower()

    def _is_colmap_learned_marked_broken(self, colmap_bin: str) -> bool:
        cache = _load_colmap_learned_crash_cache()
        return self._colmap_binary_cache_key(colmap_bin) in cache

    def _mark_colmap_learned_broken(self, colmap_bin: str, reason: str) -> None:
        cache = _load_colmap_learned_crash_cache()
        cache[self._colmap_binary_cache_key(colmap_bin)] = {
            "reason": str(reason),
        }
        _save_colmap_learned_crash_cache(cache)

    def _read_colmap_version(self, colmap_bin: str) -> str:
        try:
            run_cwd, run_env = self._get_cli_env(colmap_bin)
            for cmd in ([colmap_bin, "help"], [colmap_bin, "--version"], [colmap_bin, "version"]):
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=8,
                    cwd=run_cwd,
                    env=run_env,
                )
                output = (result.stdout or result.stderr or "").strip()
                if result.returncode == 0 and output:
                    return output.splitlines()[0]
            return "unknown"
        except Exception as exc:
            return f"error({exc})"

    def _get_cli_command_help(self, executable: str, command_name: str) -> str:
        cache_key = (self._colmap_binary_cache_key(executable), str(command_name).strip().lower())
        if cache_key in self._cli_help_cache:
            return self._cli_help_cache[cache_key]

        try:
            run_cwd, run_env = self._get_cli_env(executable)
            result = subprocess.run(
                [executable, command_name, "-h"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=run_cwd,
                env=run_env,
            )
            help_text = f"{result.stdout or ''}\n{result.stderr or ''}".lower()
        except Exception:
            help_text = ""

        self._cli_help_cache[cache_key] = help_text
        return help_text

    def _filter_supported_cli_args(self, executable: str, command_name: str, args: List[str]) -> List[str]:
        help_text = self._get_cli_command_help(executable, command_name)
        if not help_text:
            return list(args)

        filtered: List[str] = []
        index = 0
        while index < len(args):
            token = args[index]
            if not token.startswith("--"):
                filtered.append(token)
                index += 1
                continue

            option_name = token.split("=", 1)[0].lstrip("-").lower()
            has_inline_value = "=" in token
            value_token = None
            has_separate_value = False
            if not has_inline_value and index + 1 < len(args) and not args[index + 1].startswith("--"):
                value_token = args[index + 1]
                has_separate_value = True

            if option_name in help_text:
                filtered.append(token)
                if has_separate_value:
                    filtered.append(value_token)
            else:
                logger.warning("[RigSfM] Dropping unsupported %s option for %s: %s", command_name, Path(executable).name, token)

            index += 2 if has_separate_value else 1

        return filtered

    def _database_has_features_and_matches(self, database_path: Path) -> bool:
        if not database_path.exists():
            return False
        try:
            with sqlite3.connect(str(database_path)) as con:
                cur = con.cursor()
                image_count = int(cur.execute("SELECT COUNT(*) FROM images").fetchone()[0])
                keypoint_count = int(cur.execute("SELECT COUNT(*) FROM keypoints").fetchone()[0])
                match_count = int(cur.execute("SELECT COUNT(*) FROM matches").fetchone()[0])
                twoview_count = int(cur.execute("SELECT COUNT(*) FROM two_view_geometries").fetchone()[0])
            return image_count > 0 and keypoint_count > 0 and (match_count > 0 or twoview_count > 0)
        except Exception:
            return False

    def _database_match_stats(self, database_path: Path) -> Tuple[int, int]:
        if not database_path.exists():
            return (0, 0)
        try:
            with sqlite3.connect(str(database_path)) as con:
                cur = con.cursor()
                match_count = int(cur.execute("SELECT COUNT(*) FROM matches").fetchone()[0])
                twoview_count = int(cur.execute("SELECT COUNT(*) FROM two_view_geometries").fetchone()[0])
            return (match_count, twoview_count)
        except Exception:
            return (0, 0)

    def create_virtual_camera(self, pano_height: int, fov_deg: float = 90) -> pycolmap.Camera:
        """Create a virtual perspective camera."""
        pycolmap = _import_pycolmap()
        image_size = int(pano_height * fov_deg / 180)
        focal = image_size / (2 * np.tan(np.deg2rad(fov_deg) / 2))
        return pycolmap.Camera.create(0, "PINHOLE", focal, image_size, image_size)

    def get_virtual_camera_rays(self, camera: pycolmap.Camera) -> np.ndarray:
        size = (camera.width, camera.height)
        y, x = np.indices(size).astype(np.float32)
        xy = np.column_stack([x.ravel(), y.ravel()])
        # The center of the upper left most pixel has coordinate (0.5, 0.5)
        xy += 0.5
        xy_norm = camera.cam_from_img(xy)
        rays = np.concatenate([xy_norm, np.ones_like(xy_norm[:, :1])], -1)
        rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
        return rays

    def spherical_img_from_cam(self, image_size, rays_in_cam: np.ndarray) -> np.ndarray:
        """Project rays into a 360 panorama (spherical) image.
        Supports both 2:1 equirectangular and 16:9 SDK-extracted formats."""
        if rays_in_cam.ndim != 2 or rays_in_cam.shape[1] != 3:
            raise ValueError(f"{rays_in_cam.shape=} but expected (N,3).")
        
        width, height = image_size
        aspect_ratio = width / height
        is_equirect = abs(aspect_ratio - 2.0) < 0.01
        is_sdk_16_9 = abs(aspect_ratio - (16/9)) < 0.01
        
        if not (is_equirect or is_sdk_16_9):
            raise ValueError(f"Unsupported aspect ratio {aspect_ratio:.2f}:1. Need 2:1 or 16:9.")
        
        r = rays_in_cam.T
        yaw = np.arctan2(r[0], r[2])
        pitch = -np.arctan2(r[1], np.linalg.norm(r[[0, 2]], axis=0))
        
        # For 16:9 SDK format, adjust the vertical FOV mapping
        if is_sdk_16_9:
            # 16:9 captures less vertical FOV than full 360°
            # Map pitch range to fit the 16:9 frame
            u = (1 + yaw / np.pi) / 2
            v = (1 - pitch * 1.5 / np.pi) / 2  # Adjusted vertical mapping
        else:
            # Standard 2:1 equirectangular
            u = (1 + yaw / np.pi) / 2
            v = (1 - pitch * 2 / np.pi) / 2
        
        return np.stack([u, v], -1) * image_size

    def get_virtual_rotations(self) -> List[np.ndarray]:
        """Custom virtual camera rotations defined by exact pitch/yaw angles."""
        pitch_yaw_pairs = [
            (0, 90), # Reference Pose
            (33, 0),
            (-42, 0),
            (0, 42),
            (0, -27),
            (42, 180),
            (-33, 180),
            (0, 207),
            (0, 138),
        ]
        cams_from_pano_r = []
        for pitch_deg, yaw_deg in pitch_yaw_pairs:
            cam_from_pano_r = Rotation.from_euler(
                "YX", [yaw_deg, pitch_deg], degrees=True
            ).as_matrix()
            cams_from_pano_r.append(cam_from_pano_r)
        return cams_from_pano_r

    def create_pano_rig_config(
        self, cams_from_pano_rotation: List[np.ndarray], ref_idx: int = 0
    ) -> pycolmap.RigConfig:
        """Create a RigConfig with proper stereo-style outward Z-offsets."""
        pycolmap = _import_pycolmap()
        rig_cameras = []
        baseline = 0.065  # 6.5cm stereo separation

        for idx, cam_from_pano_rotation in enumerate(cams_from_pano_rotation):
            if idx == ref_idx:
                cam_from_rig = None
            else:
                cam_from_ref_rotation = (
                    cam_from_pano_rotation @ cams_from_pano_rotation[ref_idx].T
                )

                # Views 1–5 = right lens, 6–10 = left lens
                # Note: Assuming 0-4 are right, 5+ are left based on order?
                # The original code logic was: side = 1 if idx <= 4 else -1
                side = 1 if idx <= 4 else -1
                local_offset = np.array([-baseline * side, 0, 0])
                translation = cam_from_ref_rotation @ local_offset

                cam_from_rig = pycolmap.Rigid3d(
                    pycolmap.Rotation3d(cam_from_ref_rotation),
                    translation
                )

            rig_cameras.append(
                pycolmap.RigConfigCamera(
                    ref_sensor=(idx == ref_idx),
                    image_prefix=f"pano_camera{idx}/",  # Matches subfolder structure
                    cam_from_rig=cam_from_rig,
                )
            )
        return pycolmap.RigConfig(cameras=rig_cameras)

    def render_perspective_images(
        self,
        pano_image_paths: List[Path],
        output_image_dir: Path,
        mask_dir: Path,
        progress_callback: Optional[Callable] = None
    ) -> pycolmap.RigConfig:
        pycolmap = _import_pycolmap()
        
        cams_from_pano_rotation = self.get_virtual_rotations()
        ref_idx = 0
        rig_config = self.create_pano_rig_config(cams_from_pano_rotation, ref_idx=ref_idx)

        # We assign each pano pixel to the virtual camera with the closest center.
        cam_centers_in_pano = np.einsum(
            "nij,i->nj", cams_from_pano_rotation, [0, 0, 1]
        )

        camera = None
        pano_size = None
        rays_in_cam = None
        
        total_steps = len(pano_image_paths) * len(cams_from_pano_rotation)
        current_step = 0

        for pano_path in tqdm(pano_image_paths, desc="Processing Panos"):
            try:
                pano_image = Image.open(pano_path)
            except Exception as e:
                logger.warning(f"Skipping file {pano_path}: {e}")
                continue
            
            # Preserve EXIF / GPS if useful
            pano_exif = pano_image.getexif()
            pano_array = np.asarray(pano_image)
            
            # Simple GPS transfer logic if needed
            gpsonly_exif = Image.Exif()
            # This part depends on if PIL.ExifTags is imported and available
            # We can skip complex EXIF logic for now or implement as needed
            
            pano_height, pano_width, *_ = pano_array.shape
            
            # Check if it's a valid format (2:1 equirectangular or 16:9 from SDK)
            aspect_ratio = pano_width / pano_height
            is_equirectangular = abs(aspect_ratio - 2.0) < 0.01  # 2:1 aspect
            is_sdk_format = abs(aspect_ratio - (16/9)) < 0.01    # 16:9 aspect (3840x2160)
            
            if not (is_equirectangular or is_sdk_format):
                logger.warning(f"Image {pano_path.name} has unsupported aspect ratio {aspect_ratio:.2f}:1. Skipping.")
                continue

            if camera is None:
                camera = self.create_virtual_camera(pano_height)
                # Assign camera to rig
                for rig_camera in rig_config.cameras:
                    rig_camera.camera = camera
                
                pano_size = (pano_width, pano_height)
                rays_in_cam = self.get_virtual_camera_rays(camera)
            else:
                 if (pano_width, pano_height) != pano_size:
                     logger.warning(f"Variable pano sizes not supported. Skipping {pano_path.name}")
                     continue

            for cam_idx, cam_from_pano_r in enumerate(cams_from_pano_rotation):
                if progress_callback:
                    progress_callback(f"Rendering camera {cam_idx+1}/{len(cams_from_pano_rotation)} for {pano_path.name}")

                rays_in_pano = rays_in_cam @ cam_from_pano_r
                xy_in_pano = self.spherical_img_from_cam(pano_size, rays_in_pano)
                
                xy_in_pano = xy_in_pano.reshape(
                    camera.width, camera.height, 2
                ).astype(np.float32)
                xy_in_pano -= 0.5
                
                image_remapped = cv2.remap(
                    pano_array,
                    *np.moveaxis(xy_in_pano, -1, 0),
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_WRAP,
                )
                
                # Mask generation
                closest_camera = np.argmax(rays_in_pano @ cam_centers_in_pano.T, -1)
                mask = (
                    ((closest_camera == cam_idx) * 255)
                    .astype(np.uint8)
                    .reshape(camera.width, camera.height)
                )
                
                # SUBFOLDER structure for COLMAP PER_FOLDER camera mode
                # Naming: pano_camera{idx}/{pano}.png (matches Kevin's approach)
                # Each subfolder gets unique camera ID matching rig camera definition
                pano_stem = pano_path.stem  # e.g., "0" or "frame_001"
                cam_folder = output_image_dir / f"pano_camera{cam_idx}"
                cam_folder.mkdir(parents=True, exist_ok=True)
                image_name = f"{pano_stem}.png"
                
                # Save to subfolder
                target_path = cam_folder / image_name
                Image.fromarray(image_remapped).save(target_path)
                
                # Masks - matching subfolder structure
                mask_cam_folder = mask_dir / f"pano_camera{cam_idx}"
                mask_cam_folder.mkdir(parents=True, exist_ok=True)
                mask_path = mask_cam_folder / image_name
                Image.fromarray(mask).save(mask_path)
                
                current_step += 1

        return rig_config

    def _resolve_colmap_binary(self) -> str:
        explicit_colmap = getattr(self.settings, 'colmap_path', None)
        if explicit_colmap:
            normalized = normalize_colmap_executable(explicit_colmap)
            if normalized and normalized.exists():
                return str(normalized)
            logger.warning("[RigSfM] Ignoring invalid configured COLMAP path: %s", explicit_colmap)

        custom = getattr(self.settings, 'sphere_alignment_path', None)
        if custom:
            normalized = normalize_colmap_executable(custom)
            if normalized and normalized.exists():
                return str(normalized)
            logger.warning("[RigSfM] Ignoring invalid custom alignment path: %s", custom)

        # Prefer bundled COLMAP binary in project (Windows build used by this app)
        try:
            project_root = Path(__file__).resolve().parents[2]
            bundled_colmap = resolve_default_colmap_path(project_root)
            if bundled_colmap.exists():
                return str(bundled_colmap)
        except Exception:
            pass

        try:
            downloaded_colmap = ensure_colmap_downloaded()
            return str(downloaded_colmap)
        except Exception as exc:
            logger.warning("[RigSfM] Lazy COLMAP download failed: %s", exc)

        return "colmap"

    def _resolve_global_mapper_binary(self) -> str:
        """
        Resolve binary used for global mapping.

        GLOMAP is integrated in modern COLMAP builds (global_mapper command),
        so we intentionally reuse the COLMAP executable here.
        """
        return self._resolve_colmap_binary()

    def _get_cli_env(self, executable: str):
        """Build environment and cwd for running COLMAP CLI."""
        extra_dirs: List[Path | str] = []

        colmap_path = getattr(self.settings, 'colmap_path', None) or getattr(self.settings, 'sphere_alignment_path', None)
        if colmap_path:
            extra_dirs.append(Path(colmap_path).parent)

        glomap_path = getattr(self.settings, 'glomap_path', None)
        if glomap_path:
            extra_dirs.append(Path(glomap_path).parent)

        return build_colmap_cli_context(executable, extra_dirs=extra_dirs)

    def _probe_colmap_gpu(self, colmap_bin: str) -> bool:
        """Test if COLMAP GPU feature extraction actually works on this hardware.
        
        Some GPUs (e.g. GTX 1650, older cards) may not be supported by the
        COLMAP CUDA kernels. RTX 30/40/50 series should work.
        Result is cached so the probe only runs once per session.
        """
        global _colmap_gpu_probe_result
        if _colmap_gpu_probe_result is not None:
            return _colmap_gpu_probe_result

        logger.info("[RigSfM] Probing COLMAP GPU support...")
        try:
            tmpdir = tempfile.mkdtemp(prefix="colmap_gpu_probe_")
            db_path = os.path.join(tmpdir, "probe.db")
            img_dir = os.path.join(tmpdir, "images")
            os.makedirs(img_dir)

            # Create a tiny test image
            test_img = np.random.randint(50, 200, (200, 200, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(img_dir, "probe.jpg"), test_img)

            run_cwd, run_env = self._get_cli_env(colmap_bin)
            result = subprocess.run(
                [colmap_bin, "feature_extractor",
                 "--database_path", db_path,
                 "--image_path", img_dir,
                 "--FeatureExtraction.use_gpu", "1"],
                capture_output=True, text=True, timeout=30,
                cwd=run_cwd, env=run_env
            )

            stderr = result.stderr or ""
            # Check for CUDA kernel incompatibility
            if "no kernel image is available" in stderr:
                logger.warning(
                    "[RigSfM] COLMAP GPU probe FAILED: CUDA kernels not compatible "
                    "with this GPU. Falling back to CPU mode. "
                    "(GPU feature extraction requires RTX 30/40/50 series or compatible)"
                )
                _colmap_gpu_probe_result = False
            elif "FAILURE" in stderr or result.returncode != 0:
                logger.warning("[RigSfM] COLMAP GPU probe FAILED (error). Using CPU mode.")
                _colmap_gpu_probe_result = False
            else:
                logger.info("[RigSfM] COLMAP GPU probe SUCCESS - GPU acceleration available")
                _colmap_gpu_probe_result = True

            # Cleanup
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception as e:
            logger.warning(f"[RigSfM] COLMAP GPU probe failed with exception: {e}. Using CPU mode.")
            _colmap_gpu_probe_result = False

        return _colmap_gpu_probe_result

    def _get_effective_gpu_setting(self, colmap_bin: str) -> Tuple[str, str]:
        """Return the effective GPU flag value and gpu_index for COLMAP CLI.
        
        If user requested GPU and it's available, returns ("1", gpu_index).
        If GPU was requested but probe fails, raises by default (GPU-only mode).
        
        Returns:
            Tuple of (use_gpu_str, gpu_index_str)
        """
        if not self.settings.use_gpu:
            return ("0", "-1")

        if self._probe_colmap_gpu(colmap_bin):
            gpu_index = str(getattr(self.settings, 'gpu_index', -1))
            return ("1", gpu_index)

        strict_gpu_only = bool(getattr(self.settings, 'strict_gpu_only', False))
        if strict_gpu_only:
            raise RuntimeError(
                "COLMAP GPU requested but probe failed. CPU fallback is disabled (strict_gpu_only=True). "
                "Check CUDA/driver/COLMAP compatibility or disable strict GPU-only mode."
            )

        logger.warning("[RigSfM] GPU probe failed; falling back to CPU mode because strict_gpu_only=False")
        return ("0", "-1")

    def _run_cli(self, executable: str, args: List[str], progress_callback: Optional[Callable], step_name: str):
        cmd = [executable] + args
        logger.info("[RigSfM] Running CLI: %s", " ".join(cmd))
        if progress_callback:
            progress_callback(step_name)
        run_cwd, run_env = self._get_cli_env(executable)
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=run_cwd, env=run_env)
        if result.returncode != 0:
            # Keep error payload readable in logs/UI for long COLMAP runs.
            stdout_tail = (result.stdout or "")[-4000:]
            stderr_tail = (result.stderr or "")[-8000:]
            if self._is_windows_interrupted_code(result.returncode):
                raise RuntimeError(
                    f"{step_name} interrupted (code {result.returncode}, hex {result.returncode & 0xFFFFFFFF:#010x})."
                )
            raise RuntimeError(
                f"{step_name} failed (code {result.returncode}, hex {result.returncode & 0xFFFFFFFF:#010x})"
                f"\nSTDOUT (tail):\n{stdout_tail}\nSTDERR (tail):\n{stderr_tail}"
            )
        return result

    def _learned_models_dir(self, colmap_bin: str) -> Path:
        exe_path = Path(colmap_bin)
        if exe_path.exists():
            runtime_dirs = get_colmap_runtime_dirs(exe_path)
            ordered_runtime_dirs = sorted(
                runtime_dirs,
                key=lambda path: (0 if path.name.lower() == "bin" else 1, len(path.parts)),
            )
            candidate_dirs = [runtime_dir / "models" for runtime_dir in ordered_runtime_dirs]
            for candidate_dir in candidate_dirs:
                if candidate_dir.exists():
                    return candidate_dir
            if candidate_dirs:
                return candidate_dirs[0]

        fallback_root = Path(__file__).resolve().parents[2] / "bin" / "COLMAP-windows-latest-CUDA-cuDSS-GUI"
        fallback_candidates = [fallback_root / "bin" / "models", fallback_root / "models"]
        for candidate_dir in fallback_candidates:
            if candidate_dir.exists():
                return candidate_dir
        return fallback_candidates[0]

    def _ensure_learned_model(self, colmap_bin: str, model_filename: str) -> str:
        url = _COLMAP_LEARNED_MODEL_URLS.get(model_filename)
        if not url:
            raise RuntimeError(f"No download URL configured for model: {model_filename}")

        models_dir = self._learned_models_dir(colmap_bin)
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / model_filename
        if model_path.exists() and model_path.stat().st_size > 0:
            return str(model_path)

        logger.info("[RigSfM] Downloading learned model: %s", model_filename)
        urllib.request.urlretrieve(url, str(model_path))
        if not model_path.exists() or model_path.stat().st_size <= 0:
            raise RuntimeError(f"Downloaded model is missing/empty: {model_path}")
        return str(model_path)

    def _prepare_learned_model_args(
        self,
        extraction_type: Optional[str],
        matcher_type: Optional[str],
        colmap_bin: str,
    ) -> Tuple[List[str], List[str]]:
        feature_model_args: List[str] = []
        matcher_model_args: List[str] = []

        if extraction_type == "ALIKED_N32":
            n32_path = self._ensure_learned_model(colmap_bin, "aliked-n32.onnx")
            feature_model_args += ["--AlikedExtraction.n32_model_path", n32_path]
        elif extraction_type == "ALIKED_N16ROT":
            n16_path = self._ensure_learned_model(colmap_bin, "aliked-n16rot.onnx")
            feature_model_args += ["--AlikedExtraction.n16rot_model_path", n16_path]

        if matcher_type == "ALIKED_LIGHTGLUE":
            aliked_lg_path = self._ensure_learned_model(colmap_bin, "aliked-lightglue.onnx")
            matcher_model_args += ["--AlikedMatching.lightglue_model_path", aliked_lg_path]
        elif matcher_type == "SIFT_LIGHTGLUE":
            sift_lg_path = self._ensure_learned_model(colmap_bin, "sift-lightglue.onnx")
            matcher_model_args += ["--SiftMatching.lightglue_model_path", sift_lg_path]

        return feature_model_args, matcher_model_args

    def _camera_grouping_args(self) -> List[str]:
        grouping = str(getattr(self.settings, 'camera_grouping', 'per_folder')).strip().lower()
        if grouping == 'single':
            return ["--ImageReader.single_camera", "1"]
        if grouping == 'per_image':
            return ["--ImageReader.single_camera_per_image", "1"]
        return ["--ImageReader.single_camera_per_folder", "1"]

    def _colmap_feature_flag_args(self) -> List[str]:
        return _split_cli_flags(getattr(self.settings, 'colmap_feature_flags', '')) + _split_cli_flags(getattr(self.settings, 'colmap_extra_args', ''))

    def _colmap_matcher_flag_args(self) -> List[str]:
        return _split_cli_flags(getattr(self.settings, 'colmap_matcher_flags', '')) + _split_cli_flags(getattr(self.settings, 'colmap_extra_args', ''))

    def _colmap_mapper_flag_args(self) -> List[str]:
        return _split_cli_flags(getattr(self.settings, 'colmap_mapper_flags', '')) + _split_cli_flags(getattr(self.settings, 'colmap_extra_args', ''))

    def _build_cli_matcher_args(
        self,
        database_path: Path,
        use_gpu_str: str,
        gpu_idx_str: str,
        selected_matcher_type: Optional[str],
        selected_matcher_model_args: List[str],
        stable: bool = False,
    ) -> List[str]:
        method = str(getattr(self.settings, 'matching_method', 'sequential')).strip().lower()
        if method == 'exhaustive':
            matcher_args = [
                "exhaustive_matcher",
                "--database_path", str(database_path),
                "--FeatureMatching.use_gpu", use_gpu_str,
                "--FeatureMatching.gpu_index", gpu_idx_str,
            ]
        elif method == 'vocab_tree':
            matcher_args = [
                "vocab_tree_matcher",
                "--database_path", str(database_path),
                "--FeatureMatching.use_gpu", use_gpu_str,
                "--FeatureMatching.gpu_index", gpu_idx_str,
            ]
        else:
            learned_matcher = selected_matcher_type in {"ALIKED_LIGHTGLUE", "SIFT_LIGHTGLUE"}
            vocab_tree_path = str(getattr(self.settings, 'vocab_tree_path', '') or '').strip()
            allow_loop_detection = (not learned_matcher) or bool(vocab_tree_path)
            configured_overlap = int(getattr(self.settings, 'colmap_sequential_overlap', 10) or 10)
            overlap = str(min(configured_overlap, 8) if stable else configured_overlap)
            loop_detection = "0" if (stable or not allow_loop_detection) else "1"
            matcher_args = [
                "sequential_matcher",
                "--database_path", str(database_path),
                "--SequentialMatching.overlap", overlap,
                "--SequentialMatching.loop_detection", loop_detection,
                "--FeatureMatching.use_gpu", use_gpu_str,
                "--FeatureMatching.gpu_index", gpu_idx_str,
            ]
            if allow_loop_detection and vocab_tree_path:
                matcher_args += ["--SequentialMatching.vocab_tree_path", vocab_tree_path]

        if selected_matcher_type:
            matcher_args += ["--FeatureMatching.type", selected_matcher_type]
            matcher_args += selected_matcher_model_args
        matcher_args += self._colmap_matcher_flag_args()
        return matcher_args

    def _run_hloc_aliked_lightglue_to_db(
        self,
        image_dir: Path,
        database_path: Path,
        progress_callback: Optional[Callable] = None,
    ) -> Tuple[bool, str]:
        if os.name == "nt" and hasattr(os, "add_dll_directory"):
            # Collect torch DLL directories from both the package location AND sys.prefix
            try:
                import torch as _torch_probe
                torch_lib = Path(_torch_probe.__file__).parent / "lib"
            except Exception:
                torch_lib = None

            dll_candidates = [
                Path(sys.prefix) / "Library" / "bin",
                Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib",
                Path(sys.prefix) / "Lib" / "site-packages" / "onnxruntime" / "capi",
            ]
            if torch_lib and torch_lib.exists():
                dll_candidates.insert(0, torch_lib)
            _register_windows_dll_dirs(dll_candidates)

        try:
            from hloc import extract_features as hloc_extract_features
            from hloc import match_features as hloc_match_features
            from hloc import pairs_from_retrieval as hloc_pairs_from_retrieval
            from hloc import reconstruction as hloc_reconstruction
            pycolmap = _import_pycolmap()
        except Exception as exc:
            return False, f"HLOC import failed: {exc}"

        try:
            if progress_callback:
                progress_callback("HLOC fallback: extracting ALIKED + NetVLAD features...")

            hloc_dir = database_path.parent / "hloc"
            hloc_dir.mkdir(parents=True, exist_ok=True)

            if database_path.exists():
                database_path.unlink()

            feat_conf = hloc_extract_features.confs["aliked-n16"]
            matcher_conf = hloc_match_features.confs["aliked+lightglue"]
            retrieval_conf = hloc_extract_features.confs["netvlad"]

            features_path = hloc_extract_features.main(feat_conf, image_dir, hloc_dir)
            retrieval_path = hloc_extract_features.main(
                retrieval_conf,
                image_dir,
                hloc_dir,
                feature_path=hloc_dir / "retrieval-netvlad.h5",
            )

            netvlad_pairs_path = hloc_dir / "pairs-netvlad-raw.txt"
            combined_pairs_path = hloc_dir / "pairs-netvlad.txt"
            # Scale retrieval pairs: fewer per image for large datasets to keep
            # matching time and SfM complexity manageable.
            # ≤200 imgs → 20 pairs, ≤500 → 15, ≤1000 → 10, >1000 → 7
            _n_imgs = len([
                p for p in image_dir.iterdir()
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
            ])
            if _n_imgs <= 200:
                _num_matched = 20
            elif _n_imgs <= 500:
                _num_matched = 15
            elif _n_imgs <= 1000:
                _num_matched = 10
            else:
                _num_matched = 7
            logger.info("[RigSfM] HLOC retrieval: %d images → %d pairs/image (%d total pairs)",
                        _n_imgs, _num_matched, _n_imgs * _num_matched)
            hloc_pairs_from_retrieval.main(retrieval_path, netvlad_pairs_path, num_matched=_num_matched)

            # Generate cross-frame-only pairs (Soft C) and merge with NetVLAD pairs.
            # Intra-frame pairs are EXCLUDED — all cameras at the same frame share the
            # same physical position (zero baseline) → degenerate epipolar geometry.
            # Only temporal (same-cam across frames) + cross-cam-temporal + NetVLAD.
            n_pairs = self._generate_rig_aware_pairs(
                image_dir,
                combined_pairs_path,
                netvlad_pairs_path=netvlad_pairs_path,
                temporal_window=5,
            )
            pairs_path = combined_pairs_path
            logger.info("[RigSfM] Cross-frame+NetVLAD pairs (Soft C, no intra-frame): %d", n_pairs)

            matches_path = hloc_dir / "matches-aliked-lightglue.h5"
            matches_path = hloc_match_features.main(
                matcher_conf,
                pairs_path,
                features_path,
                export_dir=hloc_dir,
                matches=matches_path,
            )

            hloc_reconstruction.create_empty_db(database_path)

            valid_image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
            image_list = sorted([
                p.name for p in image_dir.iterdir()
                if p.is_file() and p.suffix.lower() in valid_image_exts
            ])
            if not image_list:
                return False, "HLOC processing failed: no valid image files found for database import"

            grouping = str(getattr(self.settings, 'camera_grouping', 'per_folder')).strip().lower()
            if grouping == 'single':
                camera_mode = pycolmap.CameraMode.SINGLE
            elif grouping == 'per_image':
                camera_mode = pycolmap.CameraMode.PER_IMAGE
            else:
                # For rig datasets (frame_NNNNN_cam_CC naming), all cameras share the
                # same physical intrinsics (same FOV, same lens, just different yaw).
                # Detect rig pattern and force SINGLE camera mode for better convergence.
                import re as _re_grp
                _rig_pat = _re_grp.compile(r'frame_\d+_cam_\d+', _re_grp.IGNORECASE)
                _is_rig = any(_rig_pat.search(n) for n in image_list[:10])
                if _is_rig:
                    camera_mode = pycolmap.CameraMode.SINGLE
                    logger.info("[RigSfM] Rig dataset detected -- forcing SINGLE camera model (shared intrinsics)")
                else:
                    camera_mode = pycolmap.CameraMode.PER_FOLDER

            hloc_reconstruction.import_images(
                image_dir,
                database_path,
                camera_mode=camera_mode,
                image_list=image_list,
            )

            image_ids: Dict[str, int] = {}
            with sqlite3.connect(str(database_path)) as con:
                cur = con.cursor()
                for image_id, name in cur.execute("SELECT image_id, name FROM images"):
                    image_ids[str(name)] = int(image_id)
            if not image_ids:
                return False, "HLOC processing failed: no images imported into COLMAP database"

            db = pycolmap.Database.open(str(database_path))
            hloc_reconstruction.import_features(image_ids, db, features_path)
            hloc_reconstruction.import_matches(
                image_ids,
                db,
                pairs_path,
                matches_path,
                skip_geometric_verification=True,  # Let COLMAP geometric_verifier handle this with proper RANSAC
            )
            db.close()

            # Store h5 paths so the mapping stage can run pycolmap seed reconstruction
            self._hloc_features_path = features_path
            self._hloc_pairs_path = pairs_path
            self._hloc_matches_path = matches_path
            self._hloc_image_dir = image_dir

            return True, "HLOC ALIKED+LightGlue database prepared"
        except Exception as exc:
            return False, f"HLOC processing failed: {exc}"

    def _run_hloc_pycolmap_seed(
        self,
        seed_dir: Path,
        progress_callback: Optional[Callable] = None,
    ) -> Tuple[bool, str, int]:
        """Run HLOC pycolmap incremental SfM as a seed model for GLOMAP.

        Uses the ALIKED features + LightGlue matches already computed by
        _run_hloc_aliked_lightglue_to_db (stored as self._hloc_* paths).
        Returns (success, message, num_aligned).
        """
        features_path = getattr(self, "_hloc_features_path", None)
        pairs_path = getattr(self, "_hloc_pairs_path", None)
        matches_path = getattr(self, "_hloc_matches_path", None)
        image_dir = getattr(self, "_hloc_image_dir", None)
        if not all([features_path, pairs_path, matches_path, image_dir]):
            return False, "HLOC h5 paths not available — run _run_hloc_aliked_lightglue_to_db first", 0

        try:
            from hloc import reconstruction as hloc_reconstruction
        except Exception as exc:
            return False, f"HLOC import failed for pycolmap seed: {exc}", 0

        try:
            if progress_callback:
                progress_callback("Building HLOC pycolmap seed model (incremental SfM)...")

            seed_dir.mkdir(parents=True, exist_ok=True)

            logger.info("[RigSfM] Running HLOC pycolmap incremental SfM seed → %s", seed_dir)
            model = hloc_reconstruction.main(
                sfm_dir=seed_dir,
                image_dir=image_dir,
                pairs=pairs_path,
                features=features_path,
                matches=matches_path,
                mapper_options={
                    "ba_global_max_num_iterations": 25,
                    "min_num_matches": 10,
                },
                verbose=False,
            )

            if model is None:
                return False, "HLOC pycolmap seed returned None", 0

            num_registered = int(model.num_reg_images()) if hasattr(model, "num_reg_images") else 0
            logger.info("[RigSfM] HLOC pycolmap seed: %d images registered", num_registered)
            if num_registered < 3:
                return False, f"HLOC pycolmap seed registered only {num_registered} images — too few", num_registered

            # Write seed model in COLMAP txt format so GLOMAP can read it
            model.write_text(str(seed_dir))
            return True, f"HLOC pycolmap seed: {num_registered} images", num_registered

        except Exception as exc:
            return False, f"HLOC pycolmap seed failed: {exc}", 0

    def _extract_return_code(self, text: str) -> Optional[int]:
        match = re.search(r"code\s+(-?\d+)", text)
        if not match:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None

    @staticmethod
    def _generate_rig_aware_pairs(
        image_dir: Path,
        pairs_path: Path,
        netvlad_pairs_path: Optional[Path] = None,
        temporal_window: int = 5,
    ) -> int:
        """Generate rig-aware matching pairs for multi-camera rig datasets.

        For 360-rig datasets (frame_NNNNN_cam_CC naming) and cubemap exports
        (frame_NNNNN_front/back/left/right/top/bottom naming), this generates
        CROSS-FRAME pairs ONLY (Soft C — zero-baseline fix):

        1. Temporal pairs: same camera across ±temporal_window frames (real
           baseline — camera moved between frames, standard stereo geometry).
        2. Cross-camera temporal pairs: adjacent cameras (±1 in yaw ring)
           across ±2 frames (overlap between neighboring views over time).
        3. NetVLAD retrieval pairs (merged in, if provided) for long-range.

        ** Intra-frame pairs (cams within the same frame) are EXCLUDED **
        All 24 cameras at a given frame share the SAME physical position →
        zero baseline → degenerate essential matrix → broken relative poses.
        Matching them against each other floods the view graph with invalid
        constraints and causes scattered reconstructions.

        Falls back to pure NetVLAD pairs when the rig pattern is not detected.
        Returns total number of unique pairs written.
        """
        import re as _re

        valid_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        all_images = sorted([
            p.name for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in valid_exts
        ])

        # Parse rig structure from either numbered camera ids or cubemap face names.
        rig_pattern = _re.compile(r'frame_(\d+)_cam_(\d+)', _re.IGNORECASE)
        cubemap_pattern = _re.compile(
            r'frame_(\d+)_(front|back|left|right|top|bottom)(?=\.[^.]+$)',
            _re.IGNORECASE,
        )
        cubemap_face_ids = {
            'front': 0,
            'right': 1,
            'back': 2,
            'left': 3,
            'top': 4,
            'bottom': 5,
        }
        cubemap_neighbors = {
            cubemap_face_ids['front']: {
                cubemap_face_ids['left'],
                cubemap_face_ids['right'],
                cubemap_face_ids['top'],
                cubemap_face_ids['bottom'],
            },
            cubemap_face_ids['back']: {
                cubemap_face_ids['left'],
                cubemap_face_ids['right'],
                cubemap_face_ids['top'],
                cubemap_face_ids['bottom'],
            },
            cubemap_face_ids['left']: {
                cubemap_face_ids['front'],
                cubemap_face_ids['back'],
                cubemap_face_ids['top'],
                cubemap_face_ids['bottom'],
            },
            cubemap_face_ids['right']: {
                cubemap_face_ids['front'],
                cubemap_face_ids['back'],
                cubemap_face_ids['top'],
                cubemap_face_ids['bottom'],
            },
            cubemap_face_ids['top']: {
                cubemap_face_ids['front'],
                cubemap_face_ids['back'],
                cubemap_face_ids['left'],
                cubemap_face_ids['right'],
            },
            cubemap_face_ids['bottom']: {
                cubemap_face_ids['front'],
                cubemap_face_ids['back'],
                cubemap_face_ids['left'],
                cubemap_face_ids['right'],
            },
        }

        def _parse_rig_image(name: str) -> Optional[Tuple[int, int, str]]:
            rig_match = rig_pattern.search(name)
            if rig_match:
                return int(rig_match.group(1)), int(rig_match.group(2)), 'cam'

            cubemap_match = cubemap_pattern.search(name)
            if cubemap_match:
                frame_id = int(cubemap_match.group(1))
                face_name = cubemap_match.group(2).lower()
                return frame_id, cubemap_face_ids[face_name], 'cubemap'

            return None

        frames: Dict[int, List[Tuple[int, str]]] = {}
        non_rig: List[str] = []
        detected_layouts: set[str] = set()
        for name in all_images:
            parsed = _parse_rig_image(name)
            if parsed:
                fid, cid, layout = parsed
                detected_layouts.add(layout)
                frames.setdefault(fid, []).append((cid, name))
            else:
                non_rig.append(name)

        if not frames:
            # No rig pattern detected — just use NetVLAD pairs unchanged
            logger.info("[RigSfM] No rig naming pattern found; using NetVLAD pairs only.")
            if netvlad_pairs_path and netvlad_pairs_path.exists():
                import shutil as _shutil
                _shutil.copy2(str(netvlad_pairs_path), str(pairs_path))
                lines = pairs_path.read_text().splitlines()
                return len([l for l in lines if l.strip()])
            return 0

        sorted_frame_ids = sorted(frames.keys())
        n_frames = len(sorted_frame_ids)
        frame_idx = {fid: i for i, fid in enumerate(sorted_frame_ids)}
        logger.info(
            "[RigSfM] Rig structure: %d frames × up to %d cameras",
            n_frames,
            max(len(v) for v in frames.values()),
        )

        pairs: set = set()

        # 1. Temporal pairs: same camera across ±temporal_window frames
        # Group images by camera id
        cam_images: Dict[int, List[Tuple[int, str]]] = {}  # cam_id → [(frame_seq_idx, name)]
        for fid, cams in frames.items():
            fi = frame_idx[fid]
            for cid, name in sorted(cams, key=lambda item: item[0]):
                cam_images.setdefault(cid, []).append((fi, name))

        for cid, entries in cam_images.items():
            entries.sort()  # sort by frame_seq_idx
            for i, (fi, name_a) in enumerate(entries):
                for j in range(i + 1, min(i + 1 + temporal_window, len(entries))):
                    fj, name_b = entries[j]
                    if fj - fi <= temporal_window:
                        a, b = (name_a, name_b) if name_a < name_b else (name_b, name_a)
                        pairs.add((a, b))

        # 2. Cross-camera temporal pairs (neighboring cam IDs across ±2 frames)
        all_cam_ids = sorted(cam_images.keys())
        n_cams = len(all_cam_ids)
        use_cubemap_adjacency = detected_layouts == {'cubemap'}
        for ci_idx, cid in enumerate(all_cam_ids):
            if use_cubemap_adjacency:
                neighbor_cids = [ncid for ncid in sorted(cubemap_neighbors.get(cid, set())) if ncid in cam_images]
            else:
                # adjacent cameras in rig (wrap-around)
                neighbor_cids = [
                    all_cam_ids[(ci_idx - 1) % n_cams],
                    all_cam_ids[(ci_idx + 1) % n_cams],
                ]
            entries_a = cam_images[cid]
            for ncid in neighbor_cids:
                entries_b = cam_images[ncid]
                # Pair each frame ±2
                for i, (fi, name_a) in enumerate(entries_a):
                    for j, (fj, name_b) in enumerate(entries_b):
                        if 0 < abs(fi - fj) <= 2:
                            a, b = (name_a, name_b) if name_a < name_b else (name_b, name_a)
                            pairs.add((a, b))

        # 3. Merge NetVLAD pairs — skip any intra-frame pairs that retrieval may produce
        # (NetVLAD may retrieve same-frame images since they look similar → degenerate)
        if netvlad_pairs_path and netvlad_pairs_path.exists():
            for line in netvlad_pairs_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    a, b = parts[0], parts[1]
                    # Skip if both images are from the same frame (zero baseline)
                    parsed_a = _parse_rig_image(a)
                    parsed_b = _parse_rig_image(b)
                    if parsed_a and parsed_b and parsed_a[0] == parsed_b[0]:
                        continue  # same frame → skip
                    pairs.add((a, b) if a < b else (b, a))

        # Write combined pairs
        pairs_lines = [f"{a} {b}" for a, b in sorted(pairs)]
        pairs_path.write_text("\n".join(pairs_lines) + "\n")
        logger.info(
            "[RigSfM] Cross-frame pairs (Soft C): %d total (temporal + cross-cam + NetVLAD, NO intra-frame)",
            len(pairs_lines),
        )
        return len(pairs_lines)

    def _is_windows_interrupted_code(self, code: int) -> bool:
        # 0xC000013A (3221225786 / -1073741510): terminated by Ctrl+C/Break or external interruption.
        return code in (3221225786, -1073741510)

    def _is_windows_stack_buffer_overrun_code(self, code: int) -> bool:
        # 0xC0000409 (3221226505 / -1073740791): process terminated by stack buffer overrun / fast fail.
        return code in (3221226505, -1073740791)

    def _normalize_sparse_output(self, sparse_path: Path) -> Path:
        direct_model = sparse_path / "cameras.bin"
        if direct_model.exists():
            target = sparse_path / "0"
            target.mkdir(parents=True, exist_ok=True)
            for name in ["cameras.bin", "images.bin", "points3D.bin", "cameras.txt", "images.txt", "points3D.txt"]:
                src = sparse_path / name
                if src.exists():
                    shutil.copy2(src, target / name)
            return target

        nested = sparse_path / "0"
        if nested.exists():
            return nested

        candidates = [path for path in sparse_path.iterdir() if path.is_dir() and (path / "cameras.bin").exists()]
        if candidates:
            return candidates[0]

        return sparse_path / "0"

    def _count_registered_images_in_model(self, model_dir: Path) -> int:
        images_txt = model_dir / "images.txt"
        if images_txt.exists():
            try:
                with images_txt.open("r", encoding="utf-8", errors="ignore") as handle:
                    for line in handle:
                        line = line.strip()
                        if line.startswith("# Number of images:"):
                            match = re.search(r"Number of images:\s*(\d+)", line)
                            if match:
                                return int(match.group(1))
            except Exception:
                pass

        images_bin = model_dir / "images.bin"
        if images_bin.exists():
            try:
                pycolmap = _import_pycolmap()
                rec = pycolmap.Reconstruction(str(model_dir))
                return int(rec.num_images())
            except Exception:
                pass

        return 0

    def _looks_like_perspective_input(self, image_paths: List[Path], input_dir: Path) -> bool:
        """Heuristic: Stage 2 outputs are already perspective views and should not be re-rendered."""
        if input_dir.name.lower() in {"perspective_views", "stage2_perspectives"}:
            return True
        sample = image_paths[: min(10, len(image_paths))]
        for image_path in sample:
            stem = image_path.stem.lower()
            if "_cam_" in stem or stem.startswith("cam"):
                return True
        return False

    def _count_images_recursive(self, root: Path) -> int:
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        return sum(1 for path in root.rglob("*") if path.is_file() and path.suffix.lower() in exts)

    def _prepare_colmap_masks(
        self,
        images_dir: Path,
        masks_dir: Path,
        colmap_masks_dir: Path,
    ) -> Path:
        """
        COLMAP mask convention: for image 'foo.jpeg', its mask must be at
        '<mask_path>/foo.jpeg.png'.

        Our pipeline stores masks as '<stem>_mask.png' (e.g. frame_00000_cam_00_mask.png).
        This method creates a temporary folder with correctly-named symlinks/copies.

        Returns the prepared mask directory (colmap_masks_dir) if any masks were
        converted, otherwise returns the original masks_dir unchanged.
        """
        colmap_masks_dir.mkdir(parents=True, exist_ok=True)

        # Build a lookup: image_stem -> image_filename (with extension)
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        stem_to_imgname: Dict[str, str] = {}
        for img in images_dir.iterdir():
            if img.is_file() and img.suffix.lower() in valid_exts:
                stem_to_imgname[img.stem] = img.name

        converted = 0
        for mask_file in masks_dir.iterdir():
            if not mask_file.is_file():
                continue
            mask_stem = mask_file.stem  # e.g. frame_00000_cam_00_mask

            # Try stripping common suffixes to find the matching image stem
            candidate_stems = [
                mask_stem,
                mask_stem.replace("_mask", ""),
                mask_stem.removesuffix("_mask"),
            ]
            img_name = None
            for candidate in candidate_stems:
                if candidate in stem_to_imgname:
                    img_name = stem_to_imgname[candidate]
                    break

            if img_name is None:
                continue  # No matching image found — skip

            # COLMAP expects: <image_filename>.png
            colmap_mask_name = img_name + ".png"
            dest = colmap_masks_dir / colmap_mask_name
            if not dest.exists():
                shutil.copy2(mask_file, dest)
            converted += 1

        logger.info("[RigSfM] Prepared %d COLMAP-compatible masks in %s", converted, colmap_masks_dir)
        if converted == 0:
            logger.warning(
                "[RigSfM] No masks could be matched to images — skipping mask_path for feature extractor."
            )
            return masks_dir  # Return original (will still be checked by count)
        return colmap_masks_dir

    def run_alignment(
        self,
        frames_dir: Path,
        masks_dir: Optional[Path],
        output_dir: Path,
        progress_callback: Optional[Callable] = None
    ) -> Dict:
        frames_dir = _resolve_runtime_path(frames_dir)
        masks_dir = _resolve_runtime_path(masks_dir) if masks_dir is not None else None
        output_dir = _resolve_runtime_path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Temp/working dirs for perspectives
        perspectives_dir = output_dir / "images"
        perspectives_mask_dir = output_dir / "masks"
        database_path = output_dir / "database.db"
        sparse_path = output_dir / "sparse"
        
        perspectives_dir.mkdir(exist_ok=True)
        perspectives_mask_dir.mkdir(exist_ok=True)
        sparse_path.mkdir(exist_ok=True)
        
        reuse_colmap_database = bool(getattr(self.settings, 'reuse_colmap_database', True))
        reusing_existing_database = False
        if database_path.exists():
            if reuse_colmap_database and self._database_has_features_and_matches(database_path):
                reusing_existing_database = True
                logger.info("[RigSfM] Reusing existing COLMAP database (features/matches already present): %s", database_path)
            else:
                database_path.unlink()

        # Gather inputs
        input_images = sorted(
            list(frames_dir.glob("*.jpg")) +
            list(frames_dir.glob("*.jpeg")) +
            list(frames_dir.glob("*.png"))
        )
        if not input_images:
            return {'success': False, 'error': 'No input images found'}

        using_existing_perspectives = True

        try:
            pycolmap = _import_pycolmap()
            mapping_backend = getattr(self.settings, 'mapping_backend', 'glomap')
            colmap_bin = self._resolve_colmap_binary()
            colmap_path_obj = Path(colmap_bin)
            colmap_cli_available = colmap_path_obj.exists() or (shutil.which(colmap_bin) is not None)
            prefer_colmap_cli = bool(getattr(self.settings, 'prefer_colmap_cli', True))
            use_external_colmap = colmap_cli_available and prefer_colmap_cli
            global_mapper_bin = self._resolve_global_mapper_binary()

            if use_external_colmap and (not reusing_existing_database):
                logger.info(f"[RigSfM] Using COLMAP CLI backend: {colmap_bin}")
                logger.info("[RigSfM] COLMAP executable path: %s", colmap_bin)
                logger.info("[RigSfM] COLMAP version: %s", self._read_colmap_version(colmap_bin))
            else:
                logger.info("[RigSfM] Using pycolmap backend")

            # 1. Use existing images directly (no reconstruction-time splitting)
            rig_config = None
            perspectives_dir = frames_dir
            if masks_dir and masks_dir.exists() and self._count_images_recursive(masks_dir) > 0:
                # COLMAP requires masks named exactly as: <image_filename>.png
                # e.g. frame_00000_cam_00.jpeg  ->  masks/frame_00000_cam_00.jpeg.png
                # Our pipeline produces _mask.png suffix — auto-convert into a colmap-compatible folder.
                perspectives_mask_dir = self._prepare_colmap_masks(
                    perspectives_dir, masks_dir, output_dir / "masks_colmap"
                )
                logger.info("[RigSfM] COLMAP-compatible masks prepared in: %s", perspectives_mask_dir)
            logger.info(f"Using existing reconstruction images ({self._count_images_recursive(perspectives_dir)} images): {perspectives_dir}")
            
            # 2. Feature Extraction
            pycolmap.set_random_seed(0)

            if use_external_colmap:
                use_gpu_str, gpu_idx_str = self._get_effective_gpu_setting(colmap_bin)
                logger.info(f"[RigSfM] Feature extraction GPU flag: {use_gpu_str} (gpu_index={gpu_idx_str})")
                use_lightglue_aliked = bool(getattr(self.settings, 'use_lightglue_aliked', True))
                prefer_colmap_learned = bool(getattr(self.settings, 'prefer_colmap_learned', False))
                require_learned_pipeline = bool(getattr(self.settings, 'require_learned_pipeline', False))
                enable_hloc_fallback = bool(getattr(self.settings, 'enable_hloc_fallback', True))
                force_hloc_learned = bool(getattr(self.settings, 'force_hloc_learned', False))
                prefer_hloc_for_learned = bool(getattr(self.settings, 'prefer_hloc_for_learned', True))
                # Default threshold = 50: always use HLOC for any meaningful dataset
                hloc_prefer_min_images = int(getattr(self.settings, 'hloc_prefer_min_images', 50) or 50)
                camera_grouping_args = self._camera_grouping_args()
                selected_matcher_type = None
                selected_matcher_model_args: List[str] = []
                used_hloc_pipeline = False
                extraction_attempts = [(None, None, "SIFT")]
                colmap_learned_marked_broken = False
                if use_lightglue_aliked:
                    total_input_images = self._count_images_recursive(perspectives_dir)
                    prefer_hloc_now = (not prefer_colmap_learned) and (force_hloc_learned or (
                        prefer_hloc_for_learned and total_input_images >= hloc_prefer_min_images
                    ))

                    colmap_learned_marked_broken = self._is_colmap_learned_marked_broken(colmap_bin)
                    if colmap_learned_marked_broken:
                        logger.warning(
                            "[RigSfM] COLMAP learned extractor previously marked unstable for this binary; skipping COLMAP learned attempts."
                        )
                        prefer_hloc_now = True

                    if prefer_hloc_now and enable_hloc_fallback:
                        logger.info(
                            "[RigSfM] Preferring HLOC learned pipeline for %d images (threshold=%d).",
                            total_input_images,
                            hloc_prefer_min_images,
                        )
                        hloc_ok, hloc_msg = self._run_hloc_aliked_lightglue_to_db(
                            image_dir=perspectives_dir,
                            database_path=database_path,
                            progress_callback=progress_callback,
                        )
                        if hloc_ok:
                            logger.info("[RigSfM] %s", hloc_msg)
                            selected_matcher_type = None
                            selected_matcher_model_args = []
                            extraction_succeeded = True
                            used_hloc_pipeline = True
                        else:
                            logger.warning("[RigSfM] Preferred HLOC learned pipeline failed: %s", hloc_msg)

                    if colmap_learned_marked_broken:
                        extraction_attempts = [
                            (None, "SIFT_LIGHTGLUE", "SIFT"),
                        ]
                    else:
                        extraction_attempts = [
                            ("ALIKED_N16ROT", "ALIKED_LIGHTGLUE", "ALIKED_N16ROT"),
                            ("ALIKED_N32", "ALIKED_LIGHTGLUE", "ALIKED_N32"),
                            (None, "SIFT_LIGHTGLUE", "SIFT"),
                        ]

                    if colmap_learned_marked_broken and require_learned_pipeline and (not enable_hloc_fallback):
                        raise RuntimeError(
                            "Learned pipeline required, but this COLMAP binary is marked unstable for learned ALIKED extraction and HLOC fallback is disabled."
                        )

                extraction_succeeded = used_hloc_pipeline
                last_extraction_error = None
                skip_remaining_aliked_after_crash = False
                for extraction_type, matcher_type, extraction_label in extraction_attempts:
                    if extraction_succeeded:
                        break
                    if skip_remaining_aliked_after_crash and extraction_type in {"ALIKED_N16ROT", "ALIKED_N32"}:
                        logger.info("[RigSfM] Skipping %s after prior ALIKED crash.", extraction_label)
                        continue

                    feature_args = [
                        "feature_extractor",
                        "--database_path", str(database_path),
                        "--image_path", str(perspectives_dir),
                        "--FeatureExtraction.use_gpu", use_gpu_str,
                        "--FeatureExtraction.gpu_index", gpu_idx_str,
                    ]
                    feature_args += camera_grouping_args
                    feature_args += ["--ImageReader.camera_model", str(getattr(self.settings, 'colmap_camera_model', 'PINHOLE'))]
                    feature_args += ["--FeatureExtraction.max_image_size", str(int(getattr(self.settings, 'colmap_max_image_size', 3200) or 3200))]
                    feature_args += ["--SiftExtraction.max_num_features", str(int(getattr(self.settings, 'colmap_max_num_features', 8192) or 8192))]
                    if extraction_type is not None:
                        feature_args += ["--FeatureExtraction.type", extraction_type]
                    if perspectives_mask_dir.exists() and self._count_images_recursive(perspectives_mask_dir) > 0:
                        feature_args += ["--ImageReader.mask_path", str(perspectives_mask_dir)]
                    feature_args += self._colmap_feature_flag_args()

                    matcher_model_args: List[str] = []
                    if use_lightglue_aliked:
                        try:
                            feature_model_args, matcher_model_args = self._prepare_learned_model_args(
                                extraction_type,
                                matcher_type,
                                colmap_bin,
                            )
                            feature_args += feature_model_args
                        except Exception as model_error:
                            logger.warning(
                                "[RigSfM] Failed to prepare learned models for %s (%s). Trying next fallback.",
                                extraction_label,
                                model_error,
                            )
                            continue

                    try:
                        feature_args = [feature_args[0]] + self._filter_supported_cli_args(
                            colmap_bin,
                            "feature_extractor",
                            feature_args[1:],
                        )
                        self._run_cli(
                            colmap_bin,
                            feature_args,
                            progress_callback,
                            f"Extracting features (COLMAP CLI, {extraction_label})..."
                        )
                        selected_matcher_type = matcher_type
                        selected_matcher_model_args = matcher_model_args
                        extraction_succeeded = True
                        break
                    except RuntimeError as feat_error:
                        last_extraction_error = feat_error
                        if not use_lightglue_aliked:
                            raise
                        feat_error_text = str(feat_error)
                        feat_code = self._extract_return_code(feat_error_text)
                        if feat_code is not None and self._is_windows_stack_buffer_overrun_code(feat_code):
                            self._mark_colmap_learned_broken(colmap_bin, f"feature_extractor_{extraction_label}_0xC0000409")
                            logger.warning(
                                "[RigSfM] Learned extractor crashed (%s / 0xC0000409). Trying learned fallbacks before plain SIFT.",
                                extraction_label,
                            )
                            if extraction_type in {"ALIKED_N16ROT", "ALIKED_N32"}:
                                skip_remaining_aliked_after_crash = True
                            if enable_hloc_fallback:
                                logger.info("[RigSfM] Attempting HLOC fallback (ALIKED + LightGlue) for database preparation...")
                                hloc_ok, hloc_msg = self._run_hloc_aliked_lightglue_to_db(
                                    image_dir=perspectives_dir,
                                    database_path=database_path,
                                    progress_callback=progress_callback,
                                )
                                if hloc_ok:
                                    logger.info("[RigSfM] %s", hloc_msg)
                                    selected_matcher_type = None
                                    selected_matcher_model_args = []
                                    extraction_succeeded = True
                                    used_hloc_pipeline = True
                                    break
                                logger.warning("[RigSfM] HLOC fallback failed: %s", hloc_msg)
                            continue
                        logger.warning(
                            "[RigSfM] %s extraction failed, trying next fallback. Error: %s",
                            extraction_label,
                            feat_error,
                        )

                if not extraction_succeeded and last_extraction_error is not None:
                    raise last_extraction_error

                if require_learned_pipeline and (not used_hloc_pipeline) and (selected_matcher_type != "ALIKED_LIGHTGLUE"):
                    raise RuntimeError(
                        "Learned pipeline required, but ALIKED+LightGlue was not active. "
                        "Enable HLOC fallback or fix learned model runtime stability."
                    )
            elif not reusing_existing_database:
                if progress_callback:
                    progress_callback("Extracting features (pycolmap)...")
                selected_matcher_type = None
                selected_matcher_model_args = []
                used_hloc_pipeline = False
                reader_options = {}
                if perspectives_mask_dir.exists() and self._count_images_recursive(perspectives_mask_dir) > 0:
                    reader_options["mask_path"] = str(perspectives_mask_dir)
                extraction_options = pycolmap.FeatureExtractionOptions()
                extraction_options.max_image_size = int(getattr(self.settings, 'colmap_max_image_size', 3200) or 3200)
                extraction_options.use_gpu = bool(getattr(self.settings, 'use_gpu', True))
                extraction_options.gpu_index = str(getattr(self.settings, 'gpu_index', -1))
                extraction_options.sift.max_num_features = int(getattr(self.settings, 'colmap_max_num_features', 8192) or 8192)
                pycolmap_device = pycolmap.Device.cuda if extraction_options.use_gpu else pycolmap.Device.cpu
                logger.info(
                    "[RigSfM] pycolmap feature extraction device=%s gpu_index=%s max_image_size=%s max_features=%s",
                    'cuda' if extraction_options.use_gpu else 'cpu',
                    extraction_options.gpu_index,
                    extraction_options.max_image_size,
                    extraction_options.sift.max_num_features,
                )
                pycolmap.extract_features(
                    str(database_path),
                    str(perspectives_dir),
                    reader_options=reader_options,
                    camera_mode=pycolmap.CameraMode.SINGLE,
                    extraction_options=extraction_options,
                    device=pycolmap_device,
                )
            
            if reusing_existing_database:
                logger.info("[RigSfM] Skipping feature extraction and matching (database reuse mode)")

            # 3. Apply Rig Config
            if rig_config is not None:
                db = pycolmap.Database.open(str(database_path))
                pycolmap.apply_rig_config([rig_config], db)
                db.close()
                
            # 4. Matching
            if use_external_colmap and (not reusing_existing_database):
                if used_hloc_pipeline:
                    logger.info("[RigSfM] Skipping COLMAP matcher: matches already prepared by HLOC (ALIKED + LightGlue).")
                else:
                    logger.info(f"[RigSfM] Matching GPU flag: {use_gpu_str} (gpu_index={gpu_idx_str})")
                    matcher_args = self._build_cli_matcher_args(
                        database_path=database_path,
                        use_gpu_str=use_gpu_str,
                        gpu_idx_str=gpu_idx_str,
                        selected_matcher_type=selected_matcher_type,
                        selected_matcher_model_args=selected_matcher_model_args,
                        stable=False,
                    )
                    try:
                        self._run_cli(
                            colmap_bin,
                            matcher_args,
                            progress_callback,
                            "Matching features (COLMAP CLI)..."
                        )
                    except RuntimeError as match_error:
                        match_error_text = str(match_error)
                        match_code = self._extract_return_code(match_error_text)
                        if match_code is not None and self._is_windows_interrupted_code(match_code):
                            raise RuntimeError(
                                "Matching was interrupted by user/system signal (0xC000013A). "
                                "Re-run the stage without stopping the process."
                            )
                        if selected_matcher_type and match_code is not None and self._is_windows_stack_buffer_overrun_code(match_code):
                            self._mark_colmap_learned_broken(colmap_bin, f"matcher_{selected_matcher_type}_0xC0000409")
                            logger.warning(
                                "[RigSfM] Learned matcher crashed in first attempt (%s / 0xC0000409). Switching directly to default matcher.",
                                selected_matcher_type,
                            )
                            if require_learned_pipeline:
                                raise RuntimeError(
                                    "Learned pipeline required, but ALIKED/LightGlue matcher crashed in first attempt."
                                )
                            fallback_retry_args = [
                                "sequential_matcher",
                                "--database_path", str(database_path),
                                "--SequentialMatching.overlap", "8",
                                "--SequentialMatching.loop_detection", "0",
                                "--FeatureMatching.use_gpu", use_gpu_str,
                                "--FeatureMatching.gpu_index", gpu_idx_str,
                            ]
                            self._run_cli(
                                colmap_bin,
                                fallback_retry_args,
                                progress_callback,
                                "Matching features (COLMAP CLI, default matcher after learned crash)..."
                            )
                            selected_matcher_type = None
                        else:
                            logger.warning(
                                "[RigSfM] Sequential matcher failed in loop-detection mode; retrying in stable mode "
                                "(loop_detection=0, overlap=8)."
                            )
                            retry_args = self._build_cli_matcher_args(
                                database_path=database_path,
                                use_gpu_str=use_gpu_str,
                                gpu_idx_str=gpu_idx_str,
                                selected_matcher_type=selected_matcher_type,
                                selected_matcher_model_args=selected_matcher_model_args,
                                stable=True,
                            )
                            try:
                                self._run_cli(
                                    colmap_bin,
                                    retry_args,
                                    progress_callback,
                                    "Matching features (COLMAP CLI, retry stable mode)..."
                                )
                            except RuntimeError as retry_match_error:
                                if selected_matcher_type:
                                    logger.warning(
                                        "[RigSfM] Learned matcher failed (%s), falling back to default matcher. Error: %s",
                                        selected_matcher_type,
                                        retry_match_error,
                                    )
                                    if require_learned_pipeline:
                                        raise RuntimeError(
                                            f"Learned pipeline required, but learned matcher failed: {retry_match_error}"
                                        )
                                    fallback_retry_args = self._build_cli_matcher_args(
                                        database_path=database_path,
                                        use_gpu_str=use_gpu_str,
                                        gpu_idx_str=gpu_idx_str,
                                        selected_matcher_type=None,
                                        selected_matcher_model_args=[],
                                        stable=True,
                                    )
                                    self._run_cli(
                                        colmap_bin,
                                        fallback_retry_args,
                                        progress_callback,
                                        "Matching features (COLMAP CLI, fallback default matcher)..."
                                    )
                                else:
                                    raise
            elif not reusing_existing_database:
                if progress_callback:
                    progress_callback("Matching features...")

                matching_options = pycolmap.FeatureMatchingOptions()
                matching_options.use_gpu = bool(getattr(self.settings, 'use_gpu', True))
                matching_options.gpu_index = str(getattr(self.settings, 'gpu_index', -1))
                pycolmap_device = pycolmap.Device.cuda if matching_options.use_gpu else pycolmap.Device.cpu
                logger.info(
                    "[RigSfM] pycolmap feature matching device=%s gpu_index=%s",
                    'cuda' if matching_options.use_gpu else 'cpu',
                    matching_options.gpu_index,
                )
                pycolmap.match_sequential(
                    str(database_path),
                    matching_options=matching_options,
                    pairing_options=pycolmap.SequentialPairingOptions(loop_detection=True),
                    device=pycolmap_device,
                )
            
            # 5. Mapping
            out_model_dir = None

            if use_external_colmap:
                match_count, twoview_count = self._database_match_stats(database_path)
                # HLOC import_matches() always writes dummy two_view_geometries (even with
                # skip_geometric_verification=True), so force geometric_verifier whenever
                # HLOC was used OR when two_view_geometries is genuinely empty.
                needs_gv = (match_count > 0) and (twoview_count == 0 or used_hloc_pipeline)
                if needs_gv:
                    if used_hloc_pipeline and twoview_count > 0:
                        # Clear HLOC's dummy geometry so COLMAP can recompute via RANSAC.
                        import sqlite3 as _sqlite3
                        with _sqlite3.connect(str(database_path)) as _con:
                            _con.execute("DELETE FROM two_view_geometries")
                            _con.commit()
                        logger.info(
                            "[RigSfM] Cleared %d HLOC dummy two_view_geometries -- COLMAP will recompute via RANSAC",
                            twoview_count,
                        )
                    else:
                        logger.info(
                            "[RigSfM] Database has %d matches but 0 two_view_geometries. Running geometric_verifier before mapping...",
                            match_count,
                        )
                    # NOTE: COLMAP 3.14 geometric_verifier does NOT accept --image_path
                    # or --FeatureMatching.* flags — only --database_path is supported.
                    # Lower min_num_inliers so ALIKED/LightGlue pairs (few keypoints) still pass.
                    self._run_cli(
                        colmap_bin,
                        [
                            "geometric_verifier",
                            "--database_path", str(database_path),
                            "--TwoViewGeometry.min_num_inliers", "4",
                        ],
                        progress_callback,
                        "Running geometric verification (COLMAP CLI)..."
                    )

            if self._is_global_backend(mapping_backend):
                # Always use GLOMAP global_mapper — fast, parallel, scales well at
                # any image count. pycolmap incremental SfM is intentionally skipped.
                logger.info("[RigSfM] Running GLOMAP global_mapper (always-GLOMAP mode)...")
                global_mapper_optional_args = self._filter_supported_cli_args(
                    global_mapper_bin,
                    "global_mapper",
                    [
                        # Increase relpose tolerance for rig cameras — cameras at the
                        # same frame share a position, creating near-degenerate geometry.
                        "--GlobalMapper.vgc_relpose_max_error", "4",
                        # Lower inlier threshold so same-frame camera pairs are used.
                        "--GlobalMapper.vgc_relpose_min_num_inliers", "15",
                        # Use pairs with fewer matches (rig pairs have fewer overlapping features).
                        "--GlobalMapper.min_num_matches", "8",
                    ],
                )
                self._run_cli(
                    global_mapper_bin,
                    [
                        "global_mapper",
                        "--database_path", str(database_path),
                        "--image_path", str(perspectives_dir),
                        "--output_path", str(sparse_path),
                    ] + global_mapper_optional_args + self._colmap_mapper_flag_args(),
                    progress_callback,
                    "Running GLOMAP global mapper..."
                )
                out_model_dir = self._normalize_sparse_output(sparse_path)
                if not out_model_dir.exists() or not ((out_model_dir / "images.bin").exists() or (out_model_dir / "images.txt").exists()):
                    return {'success': False, 'error': 'GLOMAP global mapper completed but no valid sparse model was created'}
                num_aligned = self._count_registered_images_in_model(out_model_dir)
                # Triangulate 3D points (GLOMAP only computes camera poses).
                # NOTE: COLMAP 3.14 point_triangulator crashes (0xC0000409) when the
                # model contains COLMAP 3.14 rig format files (rigs.bin / frames.bin).
                # Skip triangulation if those files are present.
                _has_rig_format = (out_model_dir / "rigs.bin").exists() or (out_model_dir / "rigs.txt").exists()
                if num_aligned > 0 and use_external_colmap and not _has_rig_format:
                    self._run_cli(
                        colmap_bin,
                        [
                            "point_triangulator",
                            "--database_path", str(database_path),
                            "--image_path", str(perspectives_dir),
                            "--input_path", str(out_model_dir),
                            "--output_path", str(out_model_dir),
                            "--Mapper.ba_refine_focal_length", "0",
                            "--Mapper.ba_refine_principal_point", "0",
                            "--Mapper.ba_refine_extra_params", "0",
                        ] + self._colmap_mapper_flag_args(),
                        progress_callback,
                        "Triangulating 3D points (COLMAP point_triangulator after GLOMAP)..."
                    )
                if num_aligned == 0:
                    logger.warning(
                        "[RigSfM] GLOMAP returned 0 registered images. Retrying with COLMAP incremental mapper (reuse database)."
                    )
                    self._run_cli(
                        colmap_bin,
                        [
                            "mapper",
                            "--database_path", str(database_path),
                            "--image_path", str(perspectives_dir),
                            "--output_path", str(sparse_path),
                            "--Mapper.ba_refine_sensor_from_rig", "0",
                            "--Mapper.ba_refine_focal_length", "0",
                            "--Mapper.ba_refine_principal_point", "0",
                            "--Mapper.ba_refine_extra_params", "0",
                        ] + self._colmap_mapper_flag_args(),
                        progress_callback,
                        "Running incremental mapping fallback (reuse database)..."
                    )
                    out_model_dir = self._normalize_sparse_output(sparse_path)
                    num_aligned = self._count_registered_images_in_model(out_model_dir)
                    if num_aligned == 0:
                        return {
                            'success': False,
                            'error': 'Mapping completed but registered 0 images (GLOMAP + incremental fallback). No poses created.'
                        }
            elif use_external_colmap:
                self._run_cli(
                    colmap_bin,
                    [
                        "mapper",
                        "--database_path", str(database_path),
                        "--image_path", str(perspectives_dir),
                        "--output_path", str(sparse_path),
                        "--Mapper.ba_refine_sensor_from_rig", "0",
                        "--Mapper.ba_refine_focal_length", "0",
                        "--Mapper.ba_refine_principal_point", "0",
                        "--Mapper.ba_refine_extra_params", "0",
                    ] + self._colmap_mapper_flag_args(),
                    progress_callback,
                    "Running incremental mapping (COLMAP CLI)..."
                )
                out_model_dir = self._normalize_sparse_output(sparse_path)
                if not out_model_dir.exists() or not ((out_model_dir / "images.bin").exists() or (out_model_dir / "images.txt").exists()):
                    return {'success': False, 'error': 'COLMAP mapper completed but no sparse model was created'}
                num_aligned = self._count_registered_images_in_model(out_model_dir)
            else:
                if progress_callback:
                    progress_callback("Running incremental mapping...")

                opts = pycolmap.IncrementalPipelineOptions(
                    ba_refine_sensor_from_rig=False,
                    ba_refine_focal_length=False,
                    ba_refine_principal_point=False,
                    ba_refine_extra_params=False,
                )

                recs = pycolmap.incremental_mapping(database_path, perspectives_dir, sparse_path, opts)

                if not recs:
                    return {'success': False, 'error': 'Reconstruction failed (no models created)'}

                best_idx = max(recs.keys(), key=lambda i: recs[i].num_points3D())
                rec = recs[best_idx]

                out_model_dir = sparse_path / "0"
                out_model_dir.mkdir(parents=True, exist_ok=True)
                rec.write_text(str(out_model_dir))
                num_aligned = rec.num_images()
            
            return {
                'success': True,
                'colmap_output': out_model_dir,
                'frames_dir': frames_dir, # Original inputs
                'perspectives_dir': perspectives_dir, # Generated inputs
                'num_aligned': num_aligned,
            }

        except Exception as e:
            logger.error(f"Rig SFM Error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'success': False, 'error': str(e)}

