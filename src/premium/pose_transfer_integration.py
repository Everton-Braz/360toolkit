"""
Pose Transfer Integration - Mode C
SphereSfM alignment + 9-camera rig perspective extraction with pose transfer.

Based on Kevin's solution from panorama_sfm.py.
This implementation:
1. Runs SphereSfM on equirectangular images
2. Extracts 9 perspective crops per frame (90° FOV each)
3. Transfers poses from spherical to perspective images
4. Creates rig-based reconstruction compatible with 3DGS training
"""

import os
# Fix OpenMP conflict: PyTorch (libomp.dll) + NumPy/MKL (libiomp5md.dll) crash
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import sqlite3

import logging
import re
import shutil
import subprocess
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import cv2
import PIL.Image
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

# pycolmap is loaded lazily to avoid Windows fatal DLL crash (0xc0000138)
# when the native module has incompatible dependencies
HAS_PYCOLMAP = False
_pycolmap = None

def _get_pycolmap():
    """Lazy import of pycolmap. Returns module or None."""
    global HAS_PYCOLMAP, _pycolmap
    if _pycolmap is not None:
        return _pycolmap
    try:
        import pycolmap
        _pycolmap = pycolmap
        HAS_PYCOLMAP = True
        return pycolmap
    except (ImportError, OSError):
        HAS_PYCOLMAP = False
        logger.warning("pycolmap not available - Pose Transfer mode disabled")
        return None


def _open_database_compat(pycolmap_module, database_path: Path):
    """Open pycolmap database across API variants."""
    db_path = str(database_path)

    # Variant A: static/class open(path) returning Database
    try:
        db_obj = pycolmap_module.Database.open(db_path)
        if db_obj is not None:
            return db_obj
    except TypeError:
        pass
    except Exception:
        pass

    # Variant B: instance open(path)
    db_obj = pycolmap_module.Database()
    db_obj.open(db_path)
    return db_obj


def _enforce_rig_sensor_assignments(database_path: Path, rig_config: 'pycolmap.RigConfig') -> None:
    """Ensure frame_data sensor_id and images.camera_id follow rig image prefixes."""
    prefix_to_sensor = {}
    for idx, rig_camera in enumerate(rig_config.cameras):
        image_prefix = getattr(rig_camera, "image_prefix", None)
        if image_prefix:
            prefix_to_sensor[image_prefix] = idx + 1

    if not prefix_to_sensor:
        logger.warning("[Mode C] Rig sensor enforcement skipped: no image_prefix entries")
        return

    con = sqlite3.connect(str(database_path))
    cur = con.cursor()

    images = cur.execute("SELECT image_id, name FROM images").fetchall()
    image_sensor_pairs = []
    unresolved = 0

    for image_id, image_name in images:
        sensor_id = None
        for prefix, mapped_sensor_id in prefix_to_sensor.items():
            if image_name.startswith(prefix):
                sensor_id = mapped_sensor_id
                break

        if sensor_id is None:
            unresolved += 1
            continue

        image_sensor_pairs.append((sensor_id, image_id))

    if image_sensor_pairs:
        cur.executemany("UPDATE images SET camera_id=? WHERE image_id=?", image_sensor_pairs)
        cur.executemany("UPDATE frame_data SET sensor_id=? WHERE data_id=?", image_sensor_pairs)

    con.commit()

    camera_distribution = cur.execute(
        "SELECT camera_id, COUNT(*) FROM images GROUP BY camera_id ORDER BY camera_id"
    ).fetchall()
    sensor_distribution = cur.execute(
        "SELECT sensor_id, COUNT(*) FROM frame_data GROUP BY sensor_id ORDER BY sensor_id"
    ).fetchall()
    con.close()

    logger.info("[Mode C] Rig sensor assignment enforced for %d images", len(image_sensor_pairs))
    logger.info("[Mode C] Image camera_id distribution: %s", camera_distribution)
    logger.info("[Mode C] Frame sensor_id distribution: %s", sensor_distribution)
    if unresolved:
        logger.warning("[Mode C] %d images did not match any rig prefix", unresolved)


@dataclass
class VirtualCamera:
    """Virtual camera configuration for 9-camera rig."""
    index: int
    pitch: float  # degrees
    yaw: float    # degrees
    fov: float = 90.0  # degrees
    is_reference: bool = False


def get_virtual_camera_rotations() -> List[Tuple[float, float]]:
    """
    Get 9-camera rig configuration (Kevin's solution).
    
    Returns:
        List of (pitch, yaw) tuples in degrees
    """
    # Kevin's exact configuration from panorama_sfm.py
    pitch_yaw_pairs = [
        (0, 90),      # Camera 0: Reference (right side)
        (33, 0),      # Camera 1: Forward ring
        (-42, 0),     # Camera 2: Forward ring
        (0, 42),      # Camera 3: Forward ring
        (0, -27),     # Camera 4: Forward ring
        (42, 180),    # Camera 5: Backward ring
        (-33, 180),   # Camera 6: Backward ring
        (0, 207),     # Camera 7: Backward ring
        (0, 138),     # Camera 8: Backward ring
    ]
    return pitch_yaw_pairs


def create_virtual_camera(pano_height: int, fov_deg: float = 90) -> 'pycolmap.Camera':
    """Create a virtual perspective camera."""
    pycolmap = _get_pycolmap()
    if pycolmap is None:
        raise ImportError("pycolmap required for pose transfer")
    
    image_size = int(pano_height * fov_deg / 180)
    focal = image_size / (2 * np.tan(np.deg2rad(fov_deg) / 2))
    return pycolmap.Camera.create(0, "PINHOLE", focal, image_size, image_size)


def create_pano_rig_config(
    cams_from_pano_rotation: List[np.ndarray], 
    ref_idx: int = 0
) -> 'pycolmap.RigConfig':
    """
    Create a RigConfig with proper stereo-style outward Z-offsets.
    
    Args:
        cams_from_pano_rotation: List of rotation matrices
        ref_idx: Index of reference camera (default: 0)
    
    Returns:
        pycolmap.RigConfig
    """
    if not HAS_PYCOLMAP:
        raise ImportError("pycolmap required for rig config")
    
    pycolmap = _get_pycolmap()
    
    pitch_yaw_pairs = get_virtual_camera_rotations()
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
            side = 1 if idx <= 4 else -1
            local_offset = np.array([-baseline * side, 0, 0])
            translation = cam_from_ref_rotation @ local_offset

            cam_from_rig = pycolmap.Rigid3d(
                pycolmap.Rotation3d(cam_from_ref_rotation),
                translation
            )

        # Build descriptive filename prefix for flat folder structure
        # Format: cam{idx}_yaw{yaw}_pitch{pitch}_
        pitch_deg, yaw_deg = pitch_yaw_pairs[idx]
        prefix = f"cam{idx:02d}_yaw{int(yaw_deg):+04d}_pitch{int(pitch_deg):+04d}_"

        rig_cameras.append(
            pycolmap.RigConfigCamera(
                ref_sensor=(idx == ref_idx),
                image_prefix=prefix,
                cam_from_rig=cam_from_rig,
            )
        )
    return pycolmap.RigConfig(cameras=rig_cameras)


def get_virtual_camera_rays(camera: 'pycolmap.Camera') -> np.ndarray:
    """Get camera rays for remapping."""
    size = (camera.width, camera.height)
    y, x = np.indices(size).astype(np.float32)
    xy = np.column_stack([x.ravel(), y.ravel()])
    xy += 0.5
    xy_norm = camera.cam_from_img(xy)
    rays = np.concatenate([xy_norm, np.ones_like(xy_norm[:, :1])], -1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    return rays


def spherical_img_from_cam(image_size: Tuple[int, int], rays_in_cam: np.ndarray) -> np.ndarray:
    """Project rays into a 360 panorama (spherical) image."""
    if image_size[0] != image_size[1] * 2:
        raise ValueError("Only 360° panoramas are supported.")
    if rays_in_cam.ndim != 2 or rays_in_cam.shape[1] != 3:
        raise ValueError(f"{rays_in_cam.shape=} but expected (N,3).")
    r = rays_in_cam.T
    yaw = np.arctan2(r[0], r[2])
    pitch = -np.arctan2(r[1], np.linalg.norm(r[[0, 2]], axis=0))
    u = (1 + yaw / np.pi) / 2
    v = (1 - pitch * 2 / np.pi) / 2
    return np.stack([u, v], -1) * image_size


def _natural_frame_key(name: str):
    """Natural sort key for frame/image names ending with numeric index."""
    stem = Path(name).stem
    match = re.search(r"(-?\d+(?:\.\d+)?)$", stem)
    if match:
        try:
            return (0, float(match.group(1)), stem)
        except ValueError:
            pass
    return (1, stem.lower(), stem)


class PoseTransferIntegrator:
    """
    Integrator for Mode C: SphereSfM + Pose Transfer.
    
    Workflow:
    1. Run SphereSfM on equirectangular images
    2. Extract 9 perspective crops per frame
    3. Transfer poses from spherical to perspective
    4. Run COLMAP with rig constraints
    """
    
    def __init__(self, settings):
        self.settings = settings
        self._has_pycolmap = _get_pycolmap() is not None
    
    def run_alignment(
        self,
        frames_dir: Path,
        masks_dir: Optional[Path],
        output_dir: Path,
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        Run SphereSfM + Pose Transfer alignment.
        
        Two-pass masking:
        1. YOLO+SAM masks on equirectangular images (for SphereSfM)
        2. YOLO+SAM masks on perspective crops (for rig reconstruction)
        
        Args:
            frames_dir: Equirectangular images
            masks_dir: Optional pre-existing masks (overrides equirect masking)
            output_dir: Output directory
            progress_callback: Progress updates
        
        Returns:
            Result dictionary
        """
        try:
            logger.info("[Mode C] Starting SphereSfM + Pose Transfer")
            if not self._has_pycolmap:
                logger.warning("[Mode C] pycolmap not available - using fallback Mode C (SphereSfM + perspective export)")
            
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # ============================================================
            # PASS 1: Generate equirectangular masks (for SphereSfM)
            # ============================================================
            if masks_dir is None or not Path(masks_dir).exists():
                if progress_callback:
                    progress_callback("Pass 1: Generating equirect masks (YOLO+SAM)...")
                
                equirect_masks_dir = output_dir / "equirect_masks"
                eq_masks_count = self._generate_yolo_sam_masks(
                    frames_dir, equirect_masks_dir, progress_callback
                )
                if eq_masks_count > 0:
                    masks_dir = equirect_masks_dir
                    logger.info(f"[Mode C] Generated {eq_masks_count} equirectangular masks")
                else:
                    masks_dir = None
                    logger.info("[Mode C] No detections in equirect images, proceeding without masks")
            
            # ============================================================
            # Step 1: Run SphereSfM on equirectangular images
            # ============================================================
            if progress_callback:
                progress_callback("Running SphereSfM alignment...")
            
            sphere_output = output_dir / "spheresfm_reconstruction"
            sphere_result = self._run_spheresfm(frames_dir, masks_dir, sphere_output, progress_callback)
            
            if not sphere_result['success']:
                if self._has_pycolmap:
                    return sphere_result
                logger.warning("[Mode C] SphereSfM failed in fallback mode, building synthetic sparse model: %s", sphere_result.get('error'))
                if progress_callback:
                    progress_callback("SphereSfM failed, using synthetic fallback model...")

                perspective_dir = output_dir / "images"
                generated = self._extract_perspectives_fallback(
                    frames_dir=frames_dir,
                    output_image_dir=perspective_dir,
                    progress_callback=progress_callback,
                )
                if generated == 0:
                    return {
                        'success': False,
                        'error': f"SphereSfM failed and fallback perspective extraction generated no images: {sphere_result.get('error')}"
                    }

                perspective_masks_dir = output_dir / "masks"
                _ = self._generate_yolo_sam_masks(
                    perspective_dir, perspective_masks_dir, progress_callback
                )

                final_sparse = output_dir / "sparse" / "0"
                self._create_minimal_sparse_model(perspective_dir, final_sparse)

                return {
                    'success': True,
                    'colmap_output': str(final_sparse),
                    'mode': 'pose_transfer_synthetic_fallback',
                    'images_dir': str(perspective_dir),
                    'masks_dir': str(perspective_masks_dir) if perspective_masks_dir.exists() else None,
                    'equirect_masks_dir': str(output_dir / "equirect_masks") if (output_dir / "equirect_masks").exists() else None,
                    'registered_images': len(list(perspective_dir.glob("*.jpg")) + list(perspective_dir.glob("*.png"))),
                    'positions': {},
                }
            
            # ============================================================
            # Step 2: Extract 9 perspective crops per frame (FLAT FOLDER)
            # ============================================================
            if progress_callback:
                progress_callback("Extracting perspective crops...")
            
            perspective_dir = output_dir / "images"
            rig_config = None
            
            try:
                if self._has_pycolmap:
                    rig_config = self._extract_perspectives(
                        frames_dir=frames_dir,
                        output_image_dir=perspective_dir,
                        progress_callback=progress_callback
                    )
                    logger.info("[Mode C] Perspective extraction complete - rig_config created")
                else:
                    generated = self._extract_perspectives_fallback(
                        frames_dir=frames_dir,
                        output_image_dir=perspective_dir,
                        progress_callback=progress_callback,
                    )
                    logger.info("[Mode C] Perspective extraction complete (fallback): %d images", generated)
            except Exception as e:
                logger.error(f"[Mode C] Perspective extraction failed: {e}", exc_info=True)
                return {
                    'success': False,
                    'error': f'Perspective extraction failed: {e}'
                }
            
            # ============================================================
            # PASS 2: Generate perspective masks (YOLO+SAM)
            # ============================================================
            if progress_callback:
                progress_callback("Pass 2: Generating perspective masks (YOLO+SAM)...")
            
            perspective_masks_dir = output_dir / "masks"
            persp_masks_count = self._generate_yolo_sam_masks(
                perspective_dir, perspective_masks_dir, progress_callback
            )
            logger.info(f"[Mode C] Generated {persp_masks_count} perspective masks")
            
            # ============================================================
            # Step 3: Check SphereSfM reconstruction
            # ============================================================
            if progress_callback:
                progress_callback("Transferring poses...")
            
            sphere_sparse = sphere_output / "sparse" / "0"
            if not sphere_sparse.exists():
                return {
                    'success': False,
                    'error': 'SphereSfM reconstruction not found'
                }
            
            # ============================================================
            # Step 4: Transfer poses to perspective images
            # ============================================================
            final_sparse = output_dir / "sparse" / "0"
            
            try:
                if self._has_pycolmap:
                    logger.info(f"[Mode C] Starting pose transfer to {final_sparse}")
                    self._transfer_poses(
                        sphere_sparse, perspective_dir, rig_config, final_sparse,
                        masks_dir=perspective_masks_dir if persp_masks_count > 0 else None
                    )
                    logger.info("[Mode C] Pose transfer completed successfully")
                else:
                    logger.info("[Mode C] Fallback pose transfer: reusing SphereSfM sparse model")
                    self._copy_sparse_model(sphere_sparse, final_sparse)
            except Exception as e:
                logger.error(f"[Mode C] Pose transfer failed: {e}", exc_info=True)
                return {
                    'success': False,
                    'error': f'Pose transfer failed: {e}'
                }
            
            if progress_callback:
                progress_callback("Pose transfer complete!")
            
            return {
                'success': True,
                'colmap_output': str(final_sparse),
                'mode': 'pose_transfer' if self._has_pycolmap else 'pose_transfer_fallback',
                'images_dir': str(perspective_dir),
                'masks_dir': str(perspective_masks_dir),
                'equirect_masks_dir': str(output_dir / "equirect_masks") if (output_dir / "equirect_masks").exists() else None,
                'registered_images': len(list(perspective_dir.glob("*.jpg")) + list(perspective_dir.glob("*.png"))),
                'positions': {}
            }
            
        except Exception as e:
            logger.error(f"[Mode C] Error: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }

    def _extract_perspectives_fallback(
        self,
        frames_dir: Path,
        output_image_dir: Path,
        progress_callback=None,
    ) -> int:
        """Extract 9 perspective views per panorama without pycolmap."""
        from src.transforms import E2PTransform

        output_image_dir.mkdir(parents=True, exist_ok=True)

        pano_images = sorted(
            list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png")),
            key=lambda p: _natural_frame_key(p.name),
        )
        if not pano_images:
            raise ValueError(f"No images found in {frames_dir}")

        transformer = E2PTransform()
        pitch_yaw_pairs = get_virtual_camera_rotations()
        generated = 0

        for idx, pano_path in enumerate(pano_images):
            if progress_callback:
                progress_callback(f"Extracting perspectives (fallback): {idx + 1}/{len(pano_images)}")

            pano_bgr = cv2.imread(str(pano_path))
            if pano_bgr is None:
                logger.warning("[Mode C] Could not read panorama: %s", pano_path)
                continue

            pano_h, pano_w = pano_bgr.shape[:2]
            if pano_w != pano_h * 2:
                logger.warning("[Mode C] Skipping non-equirect image in fallback extraction: %s", pano_path)
                continue

            image_size = int(pano_h * 90 / 180)

            for cam_idx, (pitch_deg, yaw_deg) in enumerate(pitch_yaw_pairs):
                perspective = transformer.equirect_to_pinhole(
                    pano_bgr,
                    yaw=yaw_deg,
                    pitch=pitch_deg,
                    roll=0,
                    h_fov=90,
                    v_fov=None,
                    output_width=image_size,
                    output_height=image_size,
                )

                out_name = f"cam{cam_idx:02d}_yaw{int(yaw_deg):+04d}_pitch{int(pitch_deg):+04d}_{pano_path.name}"
                out_path = output_image_dir / out_name
                if cv2.imwrite(str(out_path), perspective):
                    generated += 1

        return generated

    def _copy_sparse_model(self, src_sparse_dir: Path, dst_sparse_dir: Path) -> None:
        """Copy sparse model files from one COLMAP model directory to another."""
        dst_sparse_dir.mkdir(parents=True, exist_ok=True)
        copied = 0

        for name in ("cameras.bin", "images.bin", "points3D.bin", "cameras.txt", "images.txt", "points3D.txt"):
            src = src_sparse_dir / name
            if src.exists():
                shutil.copy2(src, dst_sparse_dir / name)
                copied += 1

        if copied == 0:
            raise RuntimeError(f"No sparse model files found in {src_sparse_dir}")

    def _create_minimal_sparse_model(self, images_dir: Path, sparse_dir: Path) -> None:
        """Create a minimal COLMAP text sparse model for fallback execution."""
        sparse_dir.mkdir(parents=True, exist_ok=True)

        image_files = sorted(
            list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.png")),
            key=lambda p: _natural_frame_key(p.name),
        )
        if not image_files:
            raise RuntimeError(f"No images available to build minimal sparse model in {images_dir}")

        first_img = cv2.imread(str(image_files[0]))
        if first_img is None:
            raise RuntimeError(f"Could not read first fallback image: {image_files[0]}")

        height, width = first_img.shape[:2]
        focal = (width / 2.0) / np.tan(np.radians(90.0 / 2.0))
        cx = width / 2.0
        cy = height / 2.0

        cameras_txt = sparse_dir / "cameras.txt"
        with open(cameras_txt, "w", encoding="utf-8") as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            f.write(f"1 PINHOLE {width} {height} {focal:.6f} {focal:.6f} {cx:.6f} {cy:.6f}\n")

        images_txt = sparse_dir / "images.txt"
        with open(images_txt, "w", encoding="utf-8") as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n")
            f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
            for idx, img in enumerate(image_files, start=1):
                tx = float(idx - 1) * 0.05
                f.write(f"{idx} 1 0 0 0 {tx:.6f} 0 0 1 {img.name}\n")
                f.write("\n")

        points3d_txt = sparse_dir / "points3D.txt"
        with open(points3d_txt, "w", encoding="utf-8") as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
    
    def _run_spheresfm(
        self,
        frames_dir: Path,
        masks_dir: Optional[Path],
        output_dir: Path,
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        Run SphereSfM on equirectangular images (WITHOUT cubemap conversion).
        
        For Mode C, we only want the spherical reconstruction.
        We'll do our own 9-camera rig perspective extraction.
        """
        from src.premium.sphere_sfm_integration import SphereSfMIntegrator
        
        integrator = SphereSfMIntegrator(self.settings)
        
        if not integrator.is_available():
            return {
                'success': False,
                'error': f'SphereSfM binary not found at {integrator.spheresfm_path}'
            }
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup paths
        database_path = output_dir / "database.db"
        sparse_path = output_dir / "sparse"
        sparse_path.mkdir(exist_ok=True)
        
        if database_path.exists():
            database_path.unlink()
        
        # Count input images
        image_files = list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png"))
        if not image_files:
            return {'success': False, 'error': 'No input images found'}
        
        # Detect image dimensions
        from PIL import Image
        test_img = Image.open(image_files[0])
        width, height = test_img.size
        
        logger.info(f"[Mode C SphereSfM] {len(image_files)} images at {width}x{height}")
        
        try:
            # Step 1: Create database
            if progress_callback:
                progress_callback("SphereSfM: Creating database...")
            
            result = integrator._run_command([
                "database_creator",
                "--database_path", str(database_path)
            ], progress_callback)
            
            if result.returncode != 0:
                return {'success': False, 'error': f'Database creation failed: {result.stderr}'}
            
            # Step 2: Feature extraction with SPHERE model
            if progress_callback:
                progress_callback("SphereSfM: Extracting features...")
            
            extractor_args = [
                "feature_extractor",
                "--database_path", str(database_path),
                "--image_path", str(frames_dir),
                "--ImageReader.camera_model", "SPHERE",
                "--ImageReader.single_camera", "1",
                "--SiftExtraction.use_gpu", "1" if self.settings.use_gpu else "0",
                "--SiftExtraction.max_image_size", "3200",
                "--SiftExtraction.max_num_features", "8192",
                "--SiftExtraction.max_num_orientations", "2",
                "--SiftExtraction.peak_threshold", "0.00667",
                "--SiftExtraction.edge_threshold", "10.0",
            ]
            
            result = integrator._run_command(extractor_args, progress_callback)
            if result.returncode != 0 and self.settings.use_gpu:
                logger.warning("[Mode C SphereSfM] GPU feature extraction failed, retrying on CPU")
                extractor_args_cpu = list(extractor_args)
                gpu_idx = extractor_args_cpu.index("--SiftExtraction.use_gpu") + 1
                extractor_args_cpu[gpu_idx] = "0"
                result = integrator._run_command(extractor_args_cpu, progress_callback)
            if result.returncode != 0:
                return {'success': False, 'error': f'Feature extraction failed: {result.stderr}'}
            
            # Step 3: Sequential matching
            if progress_callback:
                progress_callback("SphereSfM: Matching features...")
            
            matcher_args = [
                "sequential_matcher",
                "--database_path", str(database_path),
                "--SequentialMatching.overlap", "10",
                "--SequentialMatching.quadratic_overlap", "1",
                "--SequentialMatching.loop_detection", "0",
                "--SiftMatching.use_gpu", "1" if self.settings.use_gpu else "0",
            ]
            
            result = integrator._run_command(matcher_args, progress_callback)
            if result.returncode != 0 and self.settings.use_gpu:
                logger.warning("[Mode C SphereSfM] GPU feature matching failed, retrying on CPU")
                matcher_args_cpu = list(matcher_args)
                gpu_idx = matcher_args_cpu.index("--SiftMatching.use_gpu") + 1
                matcher_args_cpu[gpu_idx] = "0"
                result = integrator._run_command(matcher_args_cpu, progress_callback)
            if result.returncode != 0:
                logger.warning("[Mode C SphereSfM] Sequential matching failed, retrying with exhaustive matcher")
                exhaustive_args = [
                    "exhaustive_matcher",
                    "--database_path", str(database_path),
                    "--SiftMatching.use_gpu", "1" if self.settings.use_gpu else "0",
                ]
                result = integrator._run_command(exhaustive_args, progress_callback)
                if result.returncode != 0 and self.settings.use_gpu:
                    exhaustive_args_cpu = list(exhaustive_args)
                    gpu_idx = exhaustive_args_cpu.index("--SiftMatching.use_gpu") + 1
                    exhaustive_args_cpu[gpu_idx] = "0"
                    result = integrator._run_command(exhaustive_args_cpu, progress_callback)
            if result.returncode != 0:
                return {'success': False, 'error': f'Feature matching failed: {result.stderr}'}
            
            # Step 4: Mapper (incremental SfM with spherical geometry)
            if progress_callback:
                progress_callback("SphereSfM: Running reconstruction...")
            
            mapper_args = [
                "mapper",
                "--database_path", str(database_path),
                "--image_path", str(frames_dir),
                "--output_path", str(sparse_path),
                "--Mapper.ba_refine_focal_length", "0",
                "--Mapper.ba_refine_principal_point", "0",
                "--Mapper.ba_refine_extra_params", "0",
                "--Mapper.init_min_num_inliers", "100",
                "--Mapper.init_num_trials", "200",
                "--Mapper.init_max_error", "4",
                "--Mapper.init_max_forward_motion", "0.95",
                "--Mapper.init_min_tri_angle", "16",
                "--Mapper.abs_pose_min_num_inliers", "30",
                "--Mapper.abs_pose_max_error", "12",
                "--Mapper.abs_pose_min_inlier_ratio", "0.25",
                "--Mapper.max_reg_trials", "3",
                "--Mapper.tri_min_angle", "1.5",
                "--Mapper.tri_max_transitivity", "1",
                "--Mapper.tri_ignore_two_view_tracks", "1",
                "--Mapper.filter_max_reproj_error", "4",
                "--Mapper.filter_min_tri_angle", "1.5",
                "--Mapper.multiple_models", "1",
                "--Mapper.min_num_matches", "15",
            ]

            if getattr(integrator, "supports_sphere_camera_mapper", False):
                mapper_args.extend(["--Mapper.sphere_camera", "1"])
            else:
                logger.warning("[Mode C] Using COLMAP fallback without --Mapper.sphere_camera")
            
            result = integrator._run_command(mapper_args, progress_callback)
            if result.returncode != 0:
                return {'success': False, 'error': f'Mapper failed: {result.stderr}'}
            
            # CRITICAL: Check if reconstruction was actually created
            model_path = sparse_path / "0"
            if not model_path.exists():
                logger.error(f"[Mode C] Mapper completed but no reconstruction found at {model_path}")
                logger.error(f"[Mode C] Mapper stdout: {result.stdout}")
                logger.error(f"[Mode C] Mapper stderr: {result.stderr}")
                return {
                    'success': False,
                    'error': 'Mapper completed but no reconstruction model created. Check if images have enough features/matches.'
                }
            
            # Count registered images
            images_file = model_path / "images.txt"
            if images_file.exists():
                # Try binary format first
                images_file = model_path / "images.bin"
            
            num_registered = 0
            if (model_path / "images.txt").exists():
                with open(model_path / "images.txt", 'r') as f:
                    lines = [l for l in f if not l.startswith('#') and l.strip()]
                    num_registered = len(lines) // 2  # 2 lines per image
            elif (model_path / "images.bin").exists():
                # Convert to text format for counting
                try:
                    integrator._run_command([
                        "model_converter",
                        "--input_path", str(model_path),
                        "--output_path", str(model_path),
                        "--output_type", "TXT"
                    ], None)
                    with open(model_path / "images.txt", 'r') as f:
                        lines = [l for l in f if not l.startswith('#') and l.strip()]
                        num_registered = len(lines) // 2
                except Exception as e:
                    logger.warning(f"Could not count registered images: {e}")
            
            # NOTE: NO cubemap conversion - we'll do our own 9-camera rig extraction
            logger.info(f"[Mode C] SphereSfM reconstruction complete: {num_registered} images registered")
            
            return {
                'success': True,
                'sparse_dir': str(model_path),
                'registered_images': num_registered
            }
            
        except Exception as e:
            logger.error(f"[Mode C SphereSfM] Error: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }
    
    def _extract_perspectives(
        self,
        frames_dir: Path,
        output_image_dir: Path,
        progress_callback=None
    ) -> 'pycolmap.RigConfig':
        """
        Extract 9 perspective crops per equirectangular frame.
        
        All images are saved in a FLAT folder with descriptive names:
        cam00_yaw+090_pitch+000_frame0001.jpg
        
        The rig_config image_prefix (e.g. 'cam00_yaw+090_pitch+000_') maps each
        filename to the correct rig camera. The 'frame_name' after stripping
        the prefix (e.g. 'frame0001.jpg') groups cameras into rig frames.
        """
        output_image_dir.mkdir(parents=True, exist_ok=True)

        def _natural_frame_key(path: Path):
            stem = path.stem
            match = re.search(r"(-?\d+(?:\.\d+)?)$", stem)
            if match:
                try:
                    return (0, float(match.group(1)), stem)
                except ValueError:
                    pass
            return (1, stem.lower(), stem)
        
        # Get equirectangular images
        pano_images = sorted(
            list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png")),
            key=_natural_frame_key,
        )
        
        if not pano_images:
            raise ValueError(f"No images found in {frames_dir}")
        
        logger.info(f"[Mode C] Extracting perspectives from {len(pano_images)} frames")
        
        # Build camera rotations
        pitch_yaw_pairs = get_virtual_camera_rotations()
        cams_from_pano_rotation = []
        for pitch_deg, yaw_deg in pitch_yaw_pairs:
            R = Rotation.from_euler("YX", [yaw_deg, pitch_deg], degrees=True).as_matrix()
            cams_from_pano_rotation.append(R)
        
        ref_idx = 0  # Reference camera index
        rig_config = create_pano_rig_config(cams_from_pano_rotation, ref_idx=ref_idx)
        
        camera = pano_size = rays_in_cam = None
        
        for idx, pano_path in enumerate(pano_images):
            if progress_callback:
                progress_callback(f"Extracting perspectives: {idx+1}/{len(pano_images)}")
            
            try:
                pano_image = PIL.Image.open(pano_path)
            except Exception as e:
                logger.warning(f"Cannot read {pano_path}: {e}")
                continue
            
            pano_image = np.asarray(pano_image)
            pano_height, pano_width = pano_image.shape[:2]
            
            if pano_width != pano_height * 2:
                logger.warning(f"Skipping non-360° image: {pano_path}")
                continue
            
            if camera is None:
                camera = create_virtual_camera(pano_height)
                for rig_camera in rig_config.cameras:
                    rig_camera.camera = camera
                pano_size = (pano_width, pano_height)
                rays_in_cam = get_virtual_camera_rays(camera)
            
            # Frame identifier: use original filename stem for grouping
            # All cameras for the same frame share this identifier
            frame_id = pano_path.name  # e.g. "1068.jpg"
            
            # Extract each perspective view
            for cam_idx, cam_from_pano_r in enumerate(cams_from_pano_rotation):
                rays_in_pano = rays_in_cam @ cam_from_pano_r
                xy_in_pano = spherical_img_from_cam(pano_size, rays_in_pano)
                xy_in_pano = xy_in_pano.reshape(camera.width, camera.height, 2).astype(np.float32)
                xy_in_pano -= 0.5
                
                image = cv2.remap(
                    pano_image,
                    *np.moveaxis(xy_in_pano, -1, 0),
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_WRAP,
                )
                
                # Build flat filename: prefix + frame_id
                # e.g. "cam00_yaw+090_pitch+000_1068.jpg"
                image_name = rig_config.cameras[cam_idx].image_prefix + frame_id
                image_path = output_image_dir / image_name
                PIL.Image.fromarray(image).save(image_path)
        
        total_images = len(pano_images) * len(cams_from_pano_rotation)
        logger.info(f"[Mode C] Extracted {len(pano_images)} × {len(cams_from_pano_rotation)} = {total_images} perspective images")
        logger.info(f"[Mode C] All images in flat folder: {output_image_dir}")
        return rig_config
    
    def _generate_yolo_sam_masks(
        self,
        images_dir: Path,
        masks_dir: Path,
        progress_callback=None
    ) -> int:
        """
        Generate YOLO+SAM masks for images.
        
        Masks are saved with the naming convention:
        - Image: cam00_yaw+090_pitch+000_1068.jpg
        - Mask:  cam00_yaw+090_pitch+000_1068.jpg.png
        
        This is the COLMAP mask naming convention (image_name + .png).
        
        Returns:
            Number of masks generated
        """
        masks_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect all images
        image_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
        if not image_files:
            logger.warning(f"[Mode C] No images found for masking in {images_dir}")
            return 0
        
        logger.info(f"[Mode C] Generating YOLO+SAM masks for {len(image_files)} images")
        
        try:
            from src.masking.hybrid_yolo_sam_masker import HybridYOLOSAMMasker
            
            # HybridYOLOSAMMasker._select_device() tests actual CUDA kernel execution
            # and falls back to CPU automatically if GPU architecture is incompatible
            masker = HybridYOLOSAMMasker(
                yolo_model='yolov8m.pt',
                sam_checkpoint='sam_vit_b_01ec64.pth',
                use_gpu=self.settings.use_gpu,
                mask_dilation_pixels=15,
                yolo_confidence=0.5
            )
            
            masks_generated = 0
            for idx, img_path in enumerate(image_files):
                if progress_callback and idx % 10 == 0:
                    progress_callback(f"Generating masks: {idx+1}/{len(image_files)}")
                
                # COLMAP mask naming: image_name.ext.png
                mask_name = f"{img_path.name}.png"
                mask_path = masks_dir / mask_name
                
                mask = masker.generate_mask(img_path, mask_path)
                if mask is not None:
                    masks_generated += 1
            
            masker.cleanup()
            logger.info(f"[Mode C] Generated {masks_generated} YOLO+SAM masks (skipped {len(image_files) - masks_generated} without detections)")
            return masks_generated
            
        except ImportError as e:
            logger.warning(f"[Mode C] YOLO+SAM masker not available: {e}")
            logger.warning("[Mode C] Proceeding without object masks")
            return 0
        except Exception as e:
            logger.warning(f"[Mode C] Mask generation failed: {e}")
            logger.warning("[Mode C] Proceeding without object masks")
            return 0

    def _transfer_poses(
        self,
        sphere_sparse_dir: Path,
        perspective_images_dir: Path,
        rig_config: 'pycolmap.RigConfig',
        output_sparse_dir: Path,
        masks_dir: Optional[Path] = None
    ):
        """
        Complete pose transfer workflow with COLMAP rig pipeline.
        Based on Kevin's panorama_sfm.py implementation.
        
        Images are in a flat folder with descriptive names:
        cam00_yaw+090_pitch+000_1068.jpg
        
        The rig_config uses filename prefixes to group cameras.
        CameraMode.SINGLE is used since all cameras share intrinsics.
        """
        
        output_sparse_dir.mkdir(parents=True, exist_ok=True)
        database_path = output_sparse_dir.parent / "database.db"
        
        # Delete existing database
        if database_path.exists():
            database_path.unlink()
        
        # Lazy import pycolmap at function call time
        pycolmap = _get_pycolmap()
        if pycolmap is None:
            raise ImportError("pycolmap required for pose transfer")
        
        logger.info("[Mode C] Step 1: Feature extraction on perspective images")
        logger.info(f"[Mode C] Database path: {database_path}")
        logger.info(f"[Mode C] Perspective images: {perspective_images_dir}")
        logger.info(f"[Mode C] Output sparse: {output_sparse_dir}")
        
        pycolmap.set_random_seed(0)
        
        # Configure hardware
        try:
            if self.settings.use_gpu:
                device = pycolmap.Device.cuda
                logger.info("[Mode C] Using CUDA for feature extraction/matching")
            else:
                device = pycolmap.Device.cpu
                logger.info("[Mode C] Using CPU for feature extraction/matching")
        except AttributeError:
            device = None
            logger.info("[Mode C] Device enum not available in pycolmap, using defaults")
        
        # Reader options: include YOLO+SAM masks if available
        reader_options = {}
        if masks_dir and masks_dir.exists():
            mask_files = list(masks_dir.glob("*.png"))
            if mask_files:
                image_files = sorted(list(perspective_images_dir.glob("*.jpg")) + list(perspective_images_dir.glob("*.png")))
                expected_masks = {f"{img.name}.png" for img in image_files}
                available_masks = {mask.name for mask in mask_files}
                missing_masks = expected_masks - available_masks

                if not missing_masks:
                    reader_options["mask_path"] = str(masks_dir)
                    logger.info(f"[Mode C] Using YOLO+SAM masks from: {masks_dir} ({len(mask_files)} masks)")
                else:
                    logger.warning(
                        "[Mode C] Incomplete mask set for COLMAP (%d missing of %d images). "
                        "Proceeding without masks to keep rig camera coverage.",
                        len(missing_masks),
                        len(image_files),
                    )
            else:
                logger.info("[Mode C] No mask files found, proceeding without masks")
        
        # Feature extraction - SINGLE camera mode since all perspectives share intrinsics
        # (flat folder, no subfolders)
        extract_params = {
            'database_path': str(database_path),
            'image_path': str(perspective_images_dir),
            'reader_options': reader_options,
            'camera_mode': pycolmap.CameraMode.SINGLE
        }
        
        if device is not None:
            extract_params['device'] = device
        
        pycolmap.extract_features(**extract_params)
        
        logger.info("[Mode C] Step 2: Apply rig configuration to database")
        
        # Apply rig config to database
        db = _open_database_compat(pycolmap, database_path)
        pycolmap.apply_rig_config([rig_config], db)
        db.close()
        _enforce_rig_sensor_assignments(database_path, rig_config)

        logger.info("[Mode C] Step 2.5: Injecting SphereSfM pose priors into perspective DB")
        prior_info = self._inject_spheresfm_position_priors(
            sphere_sparse_dir=sphere_sparse_dir,
            database_path=database_path,
            rig_config=rig_config,
        )
        
        image_count = len(list(perspective_images_dir.glob("*.jpg")) + list(perspective_images_dir.glob("*.png")))

        logger.info("[Mode C] Step 3: Sequential matching with loop detection (%d images)", image_count)
        if hasattr(pycolmap, "SequentialPairingOptions"):
            seq_opts = pycolmap.SequentialPairingOptions(loop_detection=True)
            if hasattr(seq_opts, "overlap"):
                seq_opts.overlap = 12
            match_params = {
                'database_path': str(database_path),
                'pairing_options': seq_opts
            }
        else:
            seq_opts = pycolmap.SequentialMatchingOptions(loop_detection=True)
            if hasattr(seq_opts, "overlap"):
                seq_opts.overlap = 12
            match_params = {
                'database_path': str(database_path),
                'matching_options': seq_opts
            }
        if device is not None:
            match_params['device'] = device

        pycolmap.match_sequential(**match_params)
        
        logger.info("[Mode C] Step 4: Incremental mapping with rig constraints")
        
        # Incremental mapping with rig constraints
        opts = pycolmap.IncrementalPipelineOptions(
            ba_refine_sensor_from_rig=False,
            ba_refine_focal_length=False,
            ba_refine_principal_point=False,
            ba_refine_extra_params=False,
        )

        prior_count = prior_info.get('prior_count', 0)
        init_image_id1 = prior_info.get('init_image_id1')
        init_image_id2 = prior_info.get('init_image_id2')

        if init_image_id1 and hasattr(opts, "init_image_id1"):
            opts.init_image_id1 = int(init_image_id1)
        if init_image_id2 and hasattr(opts, "init_image_id2"):
            opts.init_image_id2 = int(init_image_id2)

        if prior_count > 0:
            if hasattr(opts, "use_prior_position"):
                opts.use_prior_position = True
            if hasattr(opts, "use_robust_loss_on_prior_position"):
                opts.use_robust_loss_on_prior_position = True
            if hasattr(opts, "prior_position_loss_scale"):
                opts.prior_position_loss_scale = 1.0

            logger.info(
                "[Mode C] Using SphereSfM position priors for initialization (%d images, init pair: %s, %s)",
                prior_count,
                init_image_id1,
                init_image_id2,
            )
        else:
            if init_image_id1 and init_image_id2:
                logger.info(
                    "[Mode C] Position prior columns unavailable; using SphereSfM-derived init pair only (%s, %s)",
                    init_image_id1,
                    init_image_id2,
                )
            else:
                logger.warning("[Mode C] No SphereSfM priors or init pair available; mapping will initialize without pose priors")
        
        mapping_backend = str(getattr(self.settings, 'mapping_backend', 'glomap')).strip().lower()
        if mapping_backend in {'glomap', 'global', 'global_mapper', 'colmap_global'}:
            colmap_path = getattr(self.settings, 'colmap_path', None) or getattr(self.settings, 'sphere_alignment_path', None) or "colmap"

            cmd = [
                str(colmap_path),
                "global_mapper",
                "--database_path", str(database_path),
                "--image_path", str(perspective_images_dir),
                "--output_path", str(output_sparse_dir.parent),
            ]
            logger.info("[Mode C] Running COLMAP global mapper: %s", " ".join(cmd))
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"COLMAP global mapper failed (code {result.returncode})\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
                )

            direct_files = ["cameras.bin", "images.bin", "points3D.bin", "cameras.txt", "images.txt", "points3D.txt"]
            if (output_sparse_dir.parent / "cameras.bin").exists():
                output_sparse_dir.mkdir(parents=True, exist_ok=True)
                for name in direct_files:
                    src = output_sparse_dir.parent / name
                    if src.exists():
                        shutil.copy2(src, output_sparse_dir / name)
            if not output_sparse_dir.exists() or not ((output_sparse_dir / "images.bin").exists() or (output_sparse_dir / "images.txt").exists()):
                raise RuntimeError("COLMAP global mapper completed but sparse/0 model was not created")

            logger.info("[Mode C] Pose transfer complete via COLMAP global mapper backend")
            return

        recs = pycolmap.incremental_mapping(
            str(database_path), 
            str(perspective_images_dir), 
            str(output_sparse_dir.parent),
            options=opts
        )
        
        # Process reconstructions
        for idx, rec in recs.items():
            logger.info(f"[Mode C] Reconstruction #{idx}: {rec.summary()}")
            
            out_dir = output_sparse_dir.parent / str(idx)
            out_dir.mkdir(exist_ok=True, parents=True)
            
            logger.info(f"[Mode C] Writing COLMAP model to {out_dir}")
            rec.write_text(str(out_dir))
        
        logger.info(f"[Mode C] Pose transfer complete: {len(recs)} reconstructions")

    def _load_spheresfm_camera_centers(
        self,
        sphere_sparse_dir: Path,
    ) -> Dict[str, np.ndarray]:
        """Load camera centers from SphereSfM reconstruction keyed by image filename."""
        pycolmap = _get_pycolmap()
        if pycolmap is None:
            return {}

        reconstruction = pycolmap.Reconstruction(str(sphere_sparse_dir))
        centers_by_name: Dict[str, np.ndarray] = {}

        for image in reconstruction.images.values():
            cam_from_world = image.cam_from_world() if callable(image.cam_from_world) else image.cam_from_world
            if cam_from_world is None:
                continue

            matrix_attr = cam_from_world.matrix
            mat = np.asarray(matrix_attr() if callable(matrix_attr) else matrix_attr, dtype=np.float64)
            rotation = mat[:, :3]
            translation = mat[:, 3]
            camera_center = -rotation.T @ translation
            centers_by_name[Path(image.name).name] = camera_center

        return centers_by_name

    def _inject_spheresfm_position_priors(
        self,
        sphere_sparse_dir: Path,
        database_path: Path,
        rig_config: 'pycolmap.RigConfig',
    ) -> Dict[str, Optional[int]]:
        """
        Inject SphereSfM-derived camera-center priors into COLMAP DB for Mode C.

        For each perspective image `camXX_..._<frame_name>`, this maps `<frame_name>`
        to SphereSfM's equirect frame pose and writes `prior_tx/ty/tz`.
        """
        centers_by_name = self._load_spheresfm_camera_centers(sphere_sparse_dir)
        if not centers_by_name:
            logger.warning("[Mode C] Could not load SphereSfM camera centers from %s", sphere_sparse_dir)
            return {'prior_count': 0, 'init_image_id1': None, 'init_image_id2': None}

        centers_by_stem = {Path(name).stem: center for name, center in centers_by_name.items()}

        prefixes = [
            getattr(rig_camera, "image_prefix", "")
            for rig_camera in rig_config.cameras
            if getattr(rig_camera, "image_prefix", "")
        ]
        if not prefixes:
            logger.warning("[Mode C] No rig prefixes available for prior mapping")
            return {'prior_count': 0, 'init_image_id1': None, 'init_image_id2': None}

        con = sqlite3.connect(str(database_path))
        cur = con.cursor()
        try:
            image_rows = cur.execute("SELECT image_id, name FROM images").fetchall()

            frame_to_ref_image_id: Dict[str, int] = {}
            reference_prefix = prefixes[0]

            columns = {row[1] for row in cur.execute("PRAGMA table_info(images)")}
            required_columns = {"prior_tx", "prior_ty", "prior_tz"}
            has_prior_columns = required_columns.issubset(columns)
            if not has_prior_columns:
                logger.warning("[Mode C] Database schema missing prior columns (%s)", sorted(required_columns - columns))

            updates = []

            for image_id, image_name in image_rows:
                image_name_base = Path(image_name).name
                frame_name = image_name_base
                for prefix in prefixes:
                    if image_name_base.startswith(prefix):
                        frame_name = image_name_base[len(prefix):]
                        break

                center = centers_by_name.get(frame_name)
                if center is None:
                    center = centers_by_stem.get(Path(frame_name).stem)
                if center is None:
                    continue

                if has_prior_columns:
                    updates.append((float(center[0]), float(center[1]), float(center[2]), image_id))

                if image_name_base.startswith(reference_prefix):
                    frame_to_ref_image_id[frame_name] = image_id

            if has_prior_columns and updates:
                cur.executemany(
                    "UPDATE images SET prior_tx=?, prior_ty=?, prior_tz=? WHERE image_id=?",
                    updates,
                )
                con.commit()

            ordered_frames = sorted(frame_to_ref_image_id.keys(), key=_natural_frame_key)
            init_image_id1 = frame_to_ref_image_id.get(ordered_frames[0]) if len(ordered_frames) >= 1 else None
            init_image_id2 = frame_to_ref_image_id.get(ordered_frames[1]) if len(ordered_frames) >= 2 else None

            logger.info(
                "[Mode C] SphereSfM priors applied to %d/%d perspective images",
                len(updates),
                len(image_rows),
            )

            return {
                'prior_count': len(updates),
                'init_image_id1': init_image_id1,
                'init_image_id2': init_image_id2,
            }
        finally:
            con.close()


def export_for_lichtfeld(
    colmap_dir: str,
    images_dir: str,
    masks_dir: str,
    output_dir: str,
    use_equirect: bool = False,
    equirect_dir: Optional[str] = None,
    equirect_masks_dir: Optional[str] = None,
    fix_rotation: bool = True
) -> bool:
    """
    Export Mode C results for LichtFeld Studio.
    
    LichtFeld supports both equirect and perspective images.
    Masks use naming: image.jpg → image.jpg.png
    
    Args:
        colmap_dir: Path to COLMAP sparse reconstruction (sparse/0)
        images_dir: Path to perspective images (flat folder)
        masks_dir: Path to YOLO+SAM masks
        output_dir: Export output directory
        use_equirect: If True, export equirect images instead of perspectives
        equirect_dir: Path to equirect images (required if use_equirect=True)
        equirect_masks_dir: Path to equirect masks
        fix_rotation: Apply rotation fix for 360 images
    
    Returns:
        True if successful
    """
    from src.pipeline.export_formats import LichtfeldExporter
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if use_equirect and equirect_dir:
        # Export equirectangular images with their masks
        source_images_dir = equirect_dir
        source_masks_dir = equirect_masks_dir
        logger.info("[Export] LichtFeld: Exporting equirectangular images")
    else:
        # Export perspective images with their masks
        source_images_dir = images_dir
        source_masks_dir = masks_dir
        logger.info("[Export] LichtFeld: Exporting perspective images")
    
    exporter = LichtfeldExporter(
        colmap_dir=colmap_dir,
        output_dir=output_dir
    )
    
    success = exporter.export(
        images_dir=source_images_dir,
        fix_rotation=fix_rotation,
        masks_dir=source_masks_dir
    )
    
    if success:
        # Copy masks with correct naming (image.jpg.png)
        masks_source = Path(source_masks_dir) if source_masks_dir else None
        if masks_source and masks_source.exists():
            masks_out = output_path / "masks"
            masks_out.mkdir(exist_ok=True)
            
            copied = 0
            for mask_file in masks_source.glob("*.png"):
                # Masks should already be named correctly (image.jpg.png)
                dst = masks_out / mask_file.name
                if not dst.exists():
                    import shutil
                    shutil.copy2(mask_file, dst)
                    copied += 1
            
            logger.info(f"[Export] LichtFeld: Copied {copied} masks")
    
    return success


def export_for_realityscan(
    colmap_dir: Optional[str],
    images_dir: str,
    masks_dir: str,
    output_dir: str,
    database_path: Optional[str] = None,
    flat_folder: bool = False,
) -> bool:
    """
    Export images/masks for RealityScan, with optional COLMAP model.
    
    RealityScan supports importing images + masks directly.
    If COLMAP sparse files are provided, they are exported too.
    
    Output structure:
        realityscan_export/
        ├── images/
        │   ├── cam00_yaw+090_pitch+000_1068.jpg
        │   ├── cam00_yaw+090_pitch+000_1068_mask.png  ← mask alongside image
        │   └── ...
        ├── sparse/            (optional, only when colmap_dir is provided)
        │   ├── cameras.txt
        │   ├── images.txt
        │   └── points3D.txt
        └── database.db        (optional)
    
    Args:
        colmap_dir: Path to COLMAP sparse reconstruction (sparse/0), optional
        images_dir: Path to perspective images (flat folder)
        masks_dir: Path to YOLO+SAM masks
        output_dir: Export output directory
        database_path: Optional path to COLMAP database to include
        flat_folder: If True, copy images and masks directly into output_dir
                    (single-folder export). If False, use output_dir/images.
    
    Returns:
        True if successful
    """
    import shutil

    def _rewrite_images_txt_for_realityscan(src_images_txt: Path, dst_images_txt: Path) -> int:
        """
        Rewrite COLMAP images.txt NAME field to be just the filename (no path prefix).
        RealityScan expects plain filenames in images.txt — it locates images via
        the images folder you specify during import.
        """
        def _is_int_token(value: str) -> bool:
            try:
                int(value)
                return True
            except ValueError:
                return False

        def _is_float_token(value: str) -> bool:
            try:
                float(value)
                return True
            except ValueError:
                return False

        def _is_image_header(parts: List[str]) -> bool:
            # COLMAP images.txt header format:
            # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
            if len(parts) < 10:
                return False
            if not _is_int_token(parts[0]):
                return False
            if not all(_is_float_token(token) for token in parts[1:8]):
                return False
            if not _is_int_token(parts[8]):
                return False
            return True

        rewritten = 0
        out_lines = []
        with open(src_images_txt, 'r', encoding='utf-8') as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith('#'):
                    out_lines.append(line)
                    continue

                parts = stripped.split()
                if _is_image_header(parts):
                    # Strip any path prefix — RealityScan wants plain filename only
                    image_name = Path(parts[9]).name
                    if parts[9] != image_name:
                        parts[9] = image_name
                        rewritten += 1
                    out_lines.append(' '.join(parts) + '\n')
                else:
                    out_lines.append(line)

        with open(dst_images_txt, 'w', encoding='utf-8') as f:
            f.writelines(out_lines)
        return rewritten
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    images_out = output_path if flat_folder else (output_path / "images")
    sparse_out = output_path / "sparse"
    images_out.mkdir(exist_ok=True)

    if colmap_dir and Path(colmap_dir).exists():
        sparse_out.mkdir(exist_ok=True)

        # Remove stale sparse files from previous exports (especially *.bin)
        # to ensure RealityScan loads the freshly exported TXT model with
        # corrected ../images/ paths.
        for existing_file in sparse_out.glob("*"):
            if existing_file.is_file():
                existing_file.unlink()
    
    logger.info("[Export] RealityScan: Exporting perspective images + masks")
    
    try:
        # 1. Copy perspective images
        images_source = Path(images_dir)
        images_copied = 0
        valid_exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
        image_files = sorted(
            [p for p in images_source.rglob('*') if p.is_file() and p.suffix.lower() in valid_exts],
            key=lambda p: str(p).lower()
        )

        for img_file in image_files:
            dst = images_out / img_file.name
            if not dst.exists():
                shutil.copy2(img_file, dst)
            images_copied += 1
        
        logger.info(f"[Export] RealityScan: Copied {images_copied} images")
        
        # 2. Copy masks INTO the images folder with RealityScan naming
        # COLMAP mask: cam00_..._1068.jpg.png → RealityScan: cam00_..._1068_mask.png
        masks_source = Path(masks_dir) if masks_dir else None
        masks_copied = 0
        if masks_source and masks_source.exists():
            for mask_file in sorted(masks_source.glob("*.png")):
                # Mask name: "cam00_yaw+090_pitch+000_1068.jpg.png"
                # Target: "cam00_yaw+090_pitch+000_1068_mask.png"
                mask_name = mask_file.name
                if mask_name.endswith(".jpg.png"):
                    image_stem = mask_name[:-8]  # Remove ".jpg.png"
                    rs_mask_name = f"{image_stem}_mask.png"
                elif mask_name.endswith(".png.png"):
                    image_stem = mask_name[:-8]
                    rs_mask_name = f"{image_stem}_mask.png"
                else:
                    rs_mask_name = mask_name
                
                dst = images_out / rs_mask_name
                if not dst.exists():
                    shutil.copy2(mask_file, dst)
                masks_copied += 1
            
            logger.info(f"[Export] RealityScan: Copied {masks_copied} masks into images folder")
        
        # 3. Optionally copy COLMAP sparse reconstruction (TXT preferred)
        copied_sparse = []
        if colmap_dir and Path(colmap_dir).exists():
            colmap_source = Path(colmap_dir)

            # Core text model files expected by RealityScan
            for colmap_file in ['cameras.txt', 'points3D.txt']:
                src = colmap_source / colmap_file
                if src.exists():
                    shutil.copy2(src, sparse_out / colmap_file)
                    copied_sparse.append(colmap_file)

            # Rewrite images.txt with relative path to images/ folder
            src_images_txt = colmap_source / 'images.txt'
            dst_images_txt = sparse_out / 'images.txt'
            if src_images_txt.exists():
                rewritten_count = _rewrite_images_txt_for_realityscan(src_images_txt, dst_images_txt)
                copied_sparse.append('images.txt')
                logger.info(f"[Export] RealityScan: Wrote images.txt ({rewritten_count} paths normalized to plain filename)")
            else:
                logger.warning("[Export] RealityScan: images.txt not found in COLMAP source; skipping COLMAP text export")

            logger.info(f"[Export] RealityScan: Copied COLMAP sparse model files: {copied_sparse}")
        else:
            logger.info("[Export] RealityScan: No COLMAP model provided (images + masks export only)")
            if sparse_out.exists():
                try:
                    sparse_out.rmdir()
                except Exception:
                    pass
        
        # 4. Copy database if provided
        if database_path and Path(database_path).exists():
            db_target_root = output_path / "database.db"
            shutil.copy2(database_path, db_target_root)
            if sparse_out.exists():
                try:
                    shutil.copy2(database_path, sparse_out / "database.db")
                except Exception:
                    pass
            logger.info("[Export] RealityScan: Copied database.db")
        
        logger.info(f"[Export] RealityScan export complete: {images_copied} images, {masks_copied} masks")
        return True
        
    except Exception as e:
        logger.error(f"[Export] RealityScan export failed: {e}")
        return False