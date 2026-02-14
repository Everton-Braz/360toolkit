
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Callable, List, Tuple
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
import tempfile
import re

logger = logging.getLogger(__name__)

# GPU probe cache: None = not tested, True = works, False = doesn't work
_colmap_gpu_probe_result: Optional[bool] = None

def _import_pycolmap():
    """Lazy import of pycolmap to avoid DLL crash at module-level."""
    import pycolmap
    return pycolmap

class RigColmapIntegrator:
    """
    Implements the Rig-based SfM approach for 360 videos.
    Projects 360 images into virtual perspective cameras configured in a rig,
    then runs COLMAP on these perspective images.
    """

    def __init__(self, settings):
        self.settings = settings

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
        custom = getattr(self.settings, 'sphere_alignment_path', None)
        if custom:
            return str(custom)

        # Prefer bundled COLMAP binary in project (Windows build used by this app)
        try:
            project_root = Path(__file__).resolve().parents[2]
            bundled_name = "colmap.exe" if os.name == "nt" else "colmap"
            bundled_colmap = project_root / "bin" / "colmap" / bundled_name
            if bundled_colmap.exists():
                return str(bundled_colmap)
        except Exception:
            pass

        return "colmap"

    def _resolve_glomap_binary(self) -> Optional[str]:
        custom = getattr(self.settings, 'glomap_path', None)
        if custom:
            return str(custom)
        return None

    def _get_cli_env(self, executable: str):
        """Build environment and cwd for running COLMAP/GLOMAP CLI."""
        exe_path = Path(executable)
        run_cwd = str(exe_path.parent) if exe_path.exists() else None
        run_env = None
        if exe_path.exists():
            run_env = os.environ.copy()
            extra_dirs = [str(exe_path.parent)]

            colmap_path = getattr(self.settings, 'sphere_alignment_path', None)
            if colmap_path:
                extra_dirs.append(str(Path(colmap_path).parent))

            glomap_path = getattr(self.settings, 'glomap_path', None)
            if glomap_path:
                extra_dirs.append(str(Path(glomap_path).parent))

            unique_dirs = []
            for d in extra_dirs:
                if d and d not in unique_dirs:
                    unique_dirs.append(d)

            run_env["PATH"] = ";".join(unique_dirs + [run_env.get('PATH', '')])
        return run_cwd, run_env

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
        If GPU was requested but probe fails, falls back to CPU with a warning.
        
        Returns:
            Tuple of (use_gpu_str, gpu_index_str)
        """
        if not self.settings.use_gpu:
            return ("0", "-1")

        if self._probe_colmap_gpu(colmap_bin):
            gpu_index = str(getattr(self.settings, 'gpu_index', -1))
            return ("1", gpu_index)
        else:
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

    def _extract_return_code(self, text: str) -> Optional[int]:
        match = re.search(r"code\s+(-?\d+)", text)
        if not match:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None

    def _is_windows_interrupted_code(self, code: int) -> bool:
        # 0xC000013A (3221225786 / -1073741510): terminated by Ctrl+C/Break or external interruption.
        return code in (3221225786, -1073741510)

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

    def run_alignment(
        self,
        frames_dir: Path,
        masks_dir: Optional[Path],
        output_dir: Path,
        progress_callback: Optional[Callable] = None
    ) -> Dict:
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Temp/working dirs for perspectives
        perspectives_dir = output_dir / "images"
        perspectives_mask_dir = output_dir / "masks"
        database_path = output_dir / "database.db"
        sparse_path = output_dir / "sparse"
        
        perspectives_dir.mkdir(exist_ok=True)
        perspectives_mask_dir.mkdir(exist_ok=True)
        sparse_path.mkdir(exist_ok=True)
        
        if database_path.exists():
            database_path.unlink()

        # Gather inputs
        input_images = sorted(
            list(frames_dir.glob("*.jpg")) +
            list(frames_dir.glob("*.jpeg")) +
            list(frames_dir.glob("*.png"))
        )
        if not input_images:
            return {'success': False, 'error': 'No input images found'}

        using_existing_perspectives = self._looks_like_perspective_input(input_images, frames_dir)
        
        # Validate only when we expect pano/equirectangular input
        if not using_existing_perspectives:
            try:
                test_img = Image.open(input_images[0])
                test_w, test_h = test_img.size
                test_aspect = test_w / test_h
                if abs(test_aspect - 2.0) > 0.1 and abs(test_aspect - (16/9)) > 0.1:
                    return {
                        'success': False,
                        'error': f'Invalid frame format: {test_w}x{test_h} (aspect {test_aspect:.2f}:1). '
                                 f'Expected equirectangular input (2:1 or 16:9 pano) for virtual rendering.'
                    }
            except Exception as e:
                return {'success': False, 'error': f'Could not read test frame: {e}'}

        try:
            pycolmap = _import_pycolmap()
            mapping_backend = getattr(self.settings, 'mapping_backend', 'colmap')
            colmap_bin = self._resolve_colmap_binary()
            colmap_path_obj = Path(colmap_bin)
            colmap_cli_available = colmap_path_obj.exists() or (shutil.which(colmap_bin) is not None)
            use_external_colmap = mapping_backend == 'colmap' and colmap_cli_available
            glomap_bin = self._resolve_glomap_binary()

            if use_external_colmap:
                logger.info(f"[RigSfM] Using COLMAP CLI backend: {colmap_bin}")
            else:
                logger.info("[RigSfM] Using pycolmap backend")

            # 1. Prepare perspectives input
            rig_config = None
            if using_existing_perspectives:
                perspectives_dir = frames_dir
                if masks_dir and masks_dir.exists() and self._count_images_recursive(masks_dir) > 0:
                    perspectives_mask_dir = masks_dir
                logger.info(f"Using existing perspective views ({self._count_images_recursive(perspectives_dir)} images): {perspectives_dir}")
            else:
                logger.info(f"Generating perspective views for {len(input_images)} frames...")
                if progress_callback:
                    progress_callback("Generating virtual perspective views...")
                rig_config = self.render_perspective_images(
                    input_images,
                    perspectives_dir,
                    perspectives_mask_dir,
                    progress_callback
                )
            
            # 2. Feature Extraction
            pycolmap.set_random_seed(0)

            if use_external_colmap:
                use_gpu_str, gpu_idx_str = self._get_effective_gpu_setting(colmap_bin)
                logger.info(f"[RigSfM] Feature extraction GPU flag: {use_gpu_str} (gpu_index={gpu_idx_str})")
                feature_args = [
                    "feature_extractor",
                    "--database_path", str(database_path),
                    "--image_path", str(perspectives_dir),
                    "--FeatureExtraction.use_gpu", use_gpu_str,
                    "--FeatureExtraction.gpu_index", gpu_idx_str,
                ]
                if using_existing_perspectives:
                    feature_args += ["--ImageReader.single_camera", "1"]
                else:
                    feature_args += ["--ImageReader.single_camera_per_folder", "1"]
                if perspectives_mask_dir.exists() and self._count_images_recursive(perspectives_mask_dir) > 0:
                    feature_args += ["--ImageReader.mask_path", str(perspectives_mask_dir)]

                self._run_cli(
                    colmap_bin,
                    feature_args,
                    progress_callback,
                    "Extracting features (COLMAP CLI)..."
                )
            else:
                if progress_callback:
                    progress_callback("Extracting features (pycolmap)...")
                reader_options = {}
                if perspectives_mask_dir.exists() and self._count_images_recursive(perspectives_mask_dir) > 0:
                    reader_options["mask_path"] = str(perspectives_mask_dir)
                pycolmap.extract_features(
                    str(database_path),
                    str(perspectives_dir),
                    reader_options=reader_options,
                    camera_mode=pycolmap.CameraMode.SINGLE if using_existing_perspectives else pycolmap.CameraMode.PER_FOLDER
                )
            
            # 3. Apply Rig Config
            if rig_config is not None:
                db = pycolmap.Database.open(str(database_path))
                pycolmap.apply_rig_config([rig_config], db)
                db.close()
                
            # 4. Matching
            if use_external_colmap:
                logger.info(f"[RigSfM] Matching GPU flag: {use_gpu_str} (gpu_index={gpu_idx_str})")
                matcher_args = [
                    "sequential_matcher",
                    "--database_path", str(database_path),
                    "--SequentialMatching.overlap", "12",
                    "--SequentialMatching.loop_detection", "1",
                    "--FeatureMatching.use_gpu", use_gpu_str,
                    "--FeatureMatching.gpu_index", gpu_idx_str,
                ]
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

                    logger.warning(
                        "[RigSfM] Sequential matcher failed in loop-detection mode; retrying in stable mode "
                        "(loop_detection=0, overlap=8)."
                    )
                    retry_args = [
                        "sequential_matcher",
                        "--database_path", str(database_path),
                        "--SequentialMatching.overlap", "8",
                        "--SequentialMatching.loop_detection", "0",
                        "--FeatureMatching.use_gpu", use_gpu_str,
                        "--FeatureMatching.gpu_index", gpu_idx_str,
                    ]
                    self._run_cli(
                        colmap_bin,
                        retry_args,
                        progress_callback,
                        "Matching features (COLMAP CLI, retry stable mode)..."
                    )
            else:
                if progress_callback:
                    progress_callback("Matching features...")

                pycolmap.match_sequential(
                    str(database_path),
                    pairing_options=pycolmap.SequentialPairingOptions(loop_detection=True)
                )
            
            # 5. Mapping
            out_model_dir = None

            if mapping_backend == 'glomap':
                if not glomap_bin:
                    return {'success': False, 'error': 'GloMAP backend selected but glomap_path is not configured'}
                self._run_cli(
                    glomap_bin,
                    [
                        "mapper",
                        "--database_path", str(database_path),
                        "--image_path", str(perspectives_dir),
                        "--output_path", str(sparse_path),
                    ],
                    progress_callback,
                    "Running GloMAP mapper..."
                )
                out_model_dir = self._normalize_sparse_output(sparse_path)
                if not out_model_dir.exists() or not ((out_model_dir / "images.bin").exists() or (out_model_dir / "images.txt").exists()):
                    return {'success': False, 'error': 'GloMAP completed but no valid sparse model was created'}
                num_aligned = self._count_images_recursive(perspectives_dir)
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
                    ],
                    progress_callback,
                    "Running incremental mapping (COLMAP CLI)..."
                )
                out_model_dir = self._normalize_sparse_output(sparse_path)
                if not out_model_dir.exists() or not ((out_model_dir / "images.bin").exists() or (out_model_dir / "images.txt").exists()):
                    return {'success': False, 'error': 'COLMAP mapper completed but no sparse model was created'}
                num_aligned = self._count_images_recursive(perspectives_dir)
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

