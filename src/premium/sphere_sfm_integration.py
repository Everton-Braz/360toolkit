"""
SphereSfM Integration Module for 360ToolKit

Provides two reconstruction modes for 360° panoramic images:

Panorama SfM (SphereSfM Native): 
  - Uses SphereSfM binary (modified COLMAP) for direct spherical feature matching
  - Matches features in equirectangular space (handles wrap-around)
  - Outputs spherical reconstruction, then converts to cubic (perspective) views
  - Best for: Urban scenes, street-level captures

Perspective Reconstruction (Rig-based SfM):
  - Renders virtual perspective cameras from panoramas
  - Uses COLMAP rig constraints for alignment
  - Proven 100% registration rate
  - Best for: Indoor scenes, small-scale captures

References:
  - SphereSfM: https://github.com/json87/SphereSfM
  - Jiang et al. "3D reconstruction of spherical images" (IJRS 2024)
"""

import logging
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Optional, Callable, List, Tuple
import os

logger = logging.getLogger(__name__)

# Preferred bundled COLMAP release location (user-provided GPU build)
DEFAULT_COLMAP_RELEASE_PATH = Path(__file__).parent.parent.parent / "bin" / "colmap" / "colmap.exe"
# Legacy SphereSfM binary location
DEFAULT_SPHERESFM_PATH = Path(__file__).parent.parent.parent / "bin" / "SphereSfM" / "colmap.exe"


def _normalize_binary_candidate(path_value) -> Optional[Path]:
    """Normalize binary candidate (supports direct exe/bat path or folder path)."""
    if not path_value:
        return None
    p = Path(path_value)
    if p.is_dir():
        for filename in ("colmap.exe", "colmap.bat", "colmap.cmd"):
            candidate = p / filename
            if candidate.exists():
                return candidate
    return p


def _compose_binary_command(binary_path: Path, args: List[str]) -> List[str]:
    """Build subprocess command, including .bat/.cmd wrappers on Windows."""
    suffix = binary_path.suffix.lower()
    if suffix in {".bat", ".cmd"}:
        return ["cmd", "/c", str(binary_path)] + args
    return [str(binary_path)] + args


def _probe_binary(binary_path: Path) -> Tuple[bool, bool]:
    """Check whether a COLMAP-compatible binary is runnable and supports sphere mapper option."""
    if not binary_path.exists():
        return False, False

    run_env = os.environ.copy()
    binary_dir = str(binary_path.parent)
    run_env["PATH"] = binary_dir + os.pathsep + run_env.get("PATH", "")

    try:
        help_cmd = _compose_binary_command(binary_path, ["help"])
        help_proc = subprocess.run(
            help_cmd,
            capture_output=True,
            text=True,
            timeout=10,
            stdin=subprocess.DEVNULL,
            cwd=binary_dir,
            env=run_env,
        )
        help_text = f"{help_proc.stdout}\n{help_proc.stderr}".lower()

        is_runnable = help_proc.returncode == 0 and (
            "available commands" in help_text or "usage:" in help_text or "colmap" in help_text
        )
        if not is_runnable:
            return False, False

        mapper_cmd = _compose_binary_command(binary_path, ["mapper", "-h"])
        mapper_proc = subprocess.run(
            mapper_cmd,
            capture_output=True,
            text=True,
            timeout=10,
            stdin=subprocess.DEVNULL,
            cwd=binary_dir,
            env=run_env,
        )
        mapper_text = f"{mapper_proc.stdout}\n{mapper_proc.stderr}".lower()
        supports_sphere = "mapper.sphere_camera" in mapper_text
        return True, supports_sphere
    except Exception:
        return False, False


def resolve_spheresfm_binary_path(settings=None, override_path: Optional[Path] = None) -> Tuple[Path, List[str]]:
    """
    Resolve SphereSfM/COLMAP binary path.

    Search order (SphereSfM-first):
    1) Explicit override path
    2) settings.sphere_alignment_path (if present)
    3) Environment variables (SPHERESFM_PATH, SPHERESFM_BIN)
    4) Default bundled SphereSfM path
    5) Common local SphereSfM installs
    6) Generic COLMAP fallbacks (COLMAP_PATH, bundled colmap, PATH lookup)
    """
    candidates: List[Path] = []

    def add_candidate(value):
        normalized = _normalize_binary_candidate(value)
        if normalized is not None and normalized not in candidates:
            candidates.append(normalized)

    add_candidate(override_path)

    settings_path = getattr(settings, "sphere_alignment_path", None) if settings is not None else None
    add_candidate(settings_path)

    for env_name in ("SPHERESFM_PATH", "SPHERESFM_BIN"):
        env_value = os.environ.get(env_name)
        if env_value:
            add_candidate(env_value)

    add_candidate(DEFAULT_SPHERESFM_PATH)

    home = Path.home()
    spheresfm_common_paths = [
        home / "Documents" / "APLICATIVOS" / "360ToolKit" / "bin" / "SphereSfM" / "colmap.exe",
        home / "Documents" / "APLICATIVOS" / "360toolkit" / "bin" / "SphereSfM" / "colmap.exe",
        home / "Documents" / "SphereSfM" / "colmap.exe",
    ]
    for path in spheresfm_common_paths:
        add_candidate(path)

    # Generic fallback binaries (only if SphereSfM-specific candidates do not work)
    colmap_env_value = os.environ.get("COLMAP_PATH")
    if colmap_env_value:
        add_candidate(colmap_env_value)

    add_candidate(DEFAULT_COLMAP_RELEASE_PATH)

    generic_colmap_paths = [
        home / "Documents" / "APLICATIVOS" / "360toolkit" / "bin" / "colmap" / "colmap.exe",
        home / "Documents" / "APLICATIVOS" / "360ToolKit" / "bin" / "colmap" / "colmap.exe",
        home / "Documents" / "colmap-x64-windows-cuda" / "colmap.bat",
        home / "Documents" / "colmap-x64-windows-cuda" / "colmap.exe",
        Path("C:/colmap/colmap.exe"),
    ]
    for path in generic_colmap_paths:
        add_candidate(path)

    for command_name in ("colmap", "colmap.exe", "colmap.bat", "colmap.cmd"):
        which_result = shutil.which(command_name)
        if which_result:
            add_candidate(which_result)

    existing_candidates = [candidate for candidate in candidates if candidate.exists()]

    for candidate in existing_candidates:
        runnable, _ = _probe_binary(candidate)
        if runnable:
            return candidate, [str(c) for c in candidates]

    if existing_candidates:
        return existing_candidates[0], [str(c) for c in candidates]

    fallback = candidates[0] if candidates else DEFAULT_SPHERESFM_PATH
    return fallback, [str(c) for c in candidates]


class SphereSfMIntegrator:
    """
    SphereSfM integration for direct spherical feature matching.
    
    Panorama SfM: Uses native SphereSfM binary for end-to-end spherical reconstruction.
    """
    
    def __init__(self, settings, spheresfm_path: Optional[Path] = None):
        self.settings = settings
        self.spheresfm_path, self.searched_paths = resolve_spheresfm_binary_path(settings, spheresfm_path)
        self.supports_sphere_camera_mapper = self._detect_mapper_sphere_support()
        
        # Validate SphereSfM binary exists
        if not self.spheresfm_path.exists():
            logger.warning(f"SphereSfM binary not found at {self.spheresfm_path}")
    
    def is_available(self) -> bool:
        """Check if SphereSfM binary is available."""
        return self.spheresfm_path.exists()

    def _detect_mapper_sphere_support(self) -> bool:
        """Detect if current binary supports --Mapper.sphere_camera option."""
        if not self.spheresfm_path.exists():
            return False
        try:
            cmd = _compose_binary_command(self.spheresfm_path, ["mapper", "-h"])
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
                stdin=subprocess.DEVNULL,
            )
            help_text = f"{proc.stdout}\n{proc.stderr}".lower()
            return "mapper.sphere_camera" in help_text
        except Exception:
            return False
    
    def get_version(self) -> Optional[str]:
        """Get SphereSfM version string."""
        if not self.is_available():
            return None
        try:
            result = subprocess.run(
                [str(self.spheresfm_path), "--version"],
                capture_output=True, text=True, timeout=10,
                stdin=subprocess.DEVNULL
            )
            return result.stdout.strip() or result.stderr.strip()
        except Exception as e:
            logger.debug(f"Could not get SphereSfM version: {e}")
            return "Unknown"
    
    def _run_command(self, args: List[str], progress_callback: Optional[Callable] = None) -> subprocess.CompletedProcess:
        """Run SphereSfM command with logging."""
        cmd = _compose_binary_command(self.spheresfm_path, args)
        logger.info(f"Running: {' '.join(cmd)}")
        
        if progress_callback:
            progress_callback(f"SphereSfM: {args[0]}...")
        
        # Add SphereSfM directory to PATH for DLL dependencies
        env = os.environ.copy()
        spheresfm_dir = str(self.spheresfm_path.parent)
        env["PATH"] = spheresfm_dir + os.pathsep + env.get("PATH", "")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            cwd=spheresfm_dir,  # Run from SphereSfM directory
            stdin=subprocess.DEVNULL
        )
        
        if result.returncode != 0:
            logger.error(f"SphereSfM command failed: {result.stderr}")
        
        return result

    def _cleanup_partial_outputs(self, output_dir: Path) -> None:
        """Best-effort cleanup for failed Panorama SfM runs."""
        cleanup_targets = [
            output_dir / "sparse-cubic",
            output_dir / "sparse",
            output_dir / "database.db",
        ]
        for target in cleanup_targets:
            try:
                if target.is_dir():
                    shutil.rmtree(target, ignore_errors=True)
                elif target.exists():
                    target.unlink()
            except Exception as e:
                logger.debug(f"Cleanup skipped for {target}: {e}")
    
    def run_alignment_mode_a(
        self,
        frames_dir: Path,
        masks_dir: Optional[Path],
        output_dir: Path,
        progress_callback: Optional[Callable] = None
    ) -> Dict:
        """
        Panorama SfM: Native SphereSfM reconstruction.
        
        Workflow:
        1. database_creator - Create COLMAP database
        2. feature_extractor - Extract features with SPHERE camera model
        3. sequential_matcher / spatial_matcher - Match features
        4. mapper - Incremental SfM with spherical geometry
        5. sphere_cubic_reprojector - Convert to perspective views
        
        Args:
            frames_dir: Directory containing equirectangular images
            masks_dir: Optional directory with camera masks
            output_dir: Output directory for reconstruction
            progress_callback: Optional progress callback
            
        Returns:
            Dictionary with success status and output paths
        """
        if not self.is_available():
            return {
                'success': False,
                'error': f'SphereSfM binary not found at {self.spheresfm_path}'
            }
        
        output_dir.mkdir(parents=True, exist_ok=True)

        def fail_with_cleanup(error_message: str) -> Dict:
            self._cleanup_partial_outputs(output_dir)
            return {'success': False, 'error': error_message}
        
        # Setup paths
        database_path = output_dir / "database.db"
        sparse_path = output_dir / "sparse"
        sparse_path.mkdir(exist_ok=True)
        
        if database_path.exists():
            database_path.unlink()
        
        # Count input images
        image_files = list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png"))
        if not image_files:
            return fail_with_cleanup('No input images found')
        
        # Detect image dimensions for camera params
        from PIL import Image
        test_img = Image.open(image_files[0])
        width, height = test_img.size
        
        # SPHERE camera params: "1, width, height" (f=1, cx=width/2, cy=height/2)
        camera_params = f"1,{width},{height}"
        
        logger.info(f"SphereSfM Panorama: {len(image_files)} images at {width}x{height}")
        
        try:
            # Step 1: Create database
            if progress_callback:
                progress_callback("Creating database...")
            
            result = self._run_command([
                "database_creator",
                "--database_path", str(database_path)
            ], progress_callback)
            
            if result.returncode != 0:
                return fail_with_cleanup(f'Database creation failed: {result.stderr}')
            
            # Step 2: Feature extraction with SPHERE model
            if progress_callback:
                progress_callback("Extracting features (spherical)...")
            
            # NOTE: GUI uses camera_params="" (empty) to auto-detect
            # SPHERE model: f=1, cx=width/2, cy=height/2 (auto-calculated)
            extractor_args = [
                "feature_extractor",
                "--database_path", str(database_path),
                "--image_path", str(frames_dir),
                "--ImageReader.camera_model", "SPHERE",
                "--ImageReader.single_camera", "1"
                # camera_params left empty for auto-detection (matches GUI)
            ]
            
            # Add mask if available
            if masks_dir and masks_dir.exists():
                mask_file = masks_dir / "camera_mask.png"
                if mask_file.exists():
                    extractor_args.extend(["--ImageReader.camera_mask_path", str(mask_file)])
            
            # GPU acceleration disabled - SphereSfM binary has CUDA kernel mismatch
            # Force CPU mode (proven to work in GUI tests)
            # Settings from successful GUI run:
            extractor_args.extend([
                "--SiftExtraction.use_gpu", "0",
                "--SiftExtraction.max_image_size", "3200",        # From GUI settings
                "--SiftExtraction.max_num_features", "8192",      # From GUI settings
                "--SiftExtraction.max_num_orientations", "2",
                "--SiftExtraction.peak_threshold", "0.00667",
                "--SiftExtraction.edge_threshold", "10.0",
            ])
            
            result = self._run_command(extractor_args, progress_callback)
            
            if result.returncode != 0:
                return fail_with_cleanup(f'Feature extraction failed: {result.stderr}')
            
            # Step 3: Feature matching (sequential for video sequences)
            if progress_callback:
                progress_callback("Matching features...")
            
            matching_args = [
                "sequential_matcher",
                "--database_path", str(database_path),
                "--SequentialMatching.overlap", "10",              # From successful GUI test
                "--SequentialMatching.quadratic_overlap", "1",     # Enabled in GUI
                "--SequentialMatching.loop_detection", "0",        # Disabled in GUI
                "--SiftMatching.use_gpu", "0",                     # Force CPU (kernel mismatch)
                "--SiftMatching.max_ratio", "0.8",
                "--SiftMatching.max_distance", "0.7",
                "--SiftMatching.cross_check", "1",
                "--SiftMatching.max_num_matches", "8192",
                "--SiftMatching.max_error", "4.0",
                "--SiftMatching.confidence", "0.999",
                "--SiftMatching.max_num_trials", "10000",
                "--SiftMatching.min_inlier_ratio", "0.25",
                "--SiftMatching.min_num_inliers", "15"             # From GUI settings
            ]
            
            result = self._run_command(matching_args, progress_callback)
            
            if result.returncode != 0:
                # Try exhaustive matching as fallback
                logger.warning("Sequential matching failed, trying exhaustive...")
                matching_args[0] = "exhaustive_matcher"
                result = self._run_command(matching_args, progress_callback)
                
                if result.returncode != 0:
                    return fail_with_cleanup(f'Feature matching failed: {result.stderr}')
            
            # Step 4: Incremental mapping with sphere camera
            if progress_callback:
                progress_callback("Running spherical reconstruction...")
            
            # Mapper settings from successful GUI run (100% registration)
            mapper_args = [
                "mapper",
                "--database_path", str(database_path),
                "--image_path", str(frames_dir),
                "--output_path", str(sparse_path),
                # BA settings from GUI (refine focal for SPHERE)
                "--Mapper.ba_refine_focal_length", "0",           # Fixed f=1 for SPHERE
                "--Mapper.ba_refine_principal_point", "0",
                "--Mapper.ba_refine_extra_params", "0",
                # Initialization settings from GUI
                "--Mapper.init_min_num_inliers", "100",
                "--Mapper.init_num_trials", "200",
                "--Mapper.init_max_error", "4",
                "--Mapper.init_max_forward_motion", "0.95",
                "--Mapper.init_min_tri_angle", "16",
                # Registration settings
                "--Mapper.abs_pose_min_num_inliers", "30",
                "--Mapper.abs_pose_max_error", "12",
                "--Mapper.abs_pose_min_inlier_ratio", "0.25",
                "--Mapper.max_reg_trials", "3",
                # Triangulation settings
                "--Mapper.tri_min_angle", "1.5",
                "--Mapper.tri_max_transitivity", "1",
                "--Mapper.tri_ignore_two_view_tracks", "1",
                # Filter settings
                "--Mapper.filter_max_reproj_error", "4",
                "--Mapper.filter_min_tri_angle", "1.5",
                # Multiple models (allow incomplete reconstructions)
                "--Mapper.multiple_models", "1",
                "--Mapper.min_num_matches", "15"
            ]

            if self.supports_sphere_camera_mapper:
                mapper_args.extend(["--Mapper.sphere_camera", "1"])
            else:
                logger.warning("Current COLMAP binary does not expose --Mapper.sphere_camera; continuing without it.")
            
            result = self._run_command(mapper_args, progress_callback)
            
            if result.returncode != 0:
                return fail_with_cleanup(f'Mapper failed: {result.stderr}')
            
            # Check if reconstruction was created
            model_path = sparse_path / "0"
            if not model_path.exists():
                return fail_with_cleanup('No reconstruction model created')
            
            # Convert binary model to text format for easier analysis
            cameras_bin = model_path / "cameras.bin"
            cameras_txt = model_path / "cameras.txt"
            if cameras_bin.exists() and not cameras_txt.exists():
                try:
                    self._run_command([
                        "model_converter",
                        "--input_path", str(model_path),
                        "--output_path", str(model_path),
                        "--output_type", "TXT"
                    ], progress_callback)
                except Exception as e:
                    logger.debug(f"Model conversion to TXT failed: {e}")
            
            # Count results
            images_file = model_path / "images.txt"
            num_registered = 0
            if images_file.exists():
                with open(images_file, 'r') as f:
                    lines = [l for l in f if not l.startswith('#') and l.strip()]
                    num_registered = len(lines) // 2  # 2 lines per image
            
            points_file = model_path / "points3D.txt"
            num_points = 0
            if points_file.exists():
                with open(points_file, 'r') as f:
                    num_points = sum(1 for l in f if not l.startswith('#') and l.strip())
            
            logger.info(f"SphereSfM Panorama complete: {num_registered} images, {num_points} points")
            
            return {
                'success': True,
                'mode': 'Panorama SfM (SphereSfM)',
                'colmap_output': model_path,
                'cubic_output': None,
                'frames_dir': frames_dir,
                'num_aligned': num_registered,
                'num_points': num_points
            }
            
        except Exception as e:
            logger.error(f"SphereSfM Panorama Error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self._cleanup_partial_outputs(output_dir)
            return {'success': False, 'error': str(e)}
    
class DualModeAlignmentIntegrator:
    """
    Dual-mode alignment integrator providing both Panorama SfM and Perspective Reconstruction.
    
    Panorama SfM: SphereSfM (native spherical matching)
    Perspective Reconstruction: Rig-based SfM (virtual perspectives + rig constraints)
    """
    
    MODE_A = "sphere_sfm"      # SphereSfM native
    MODE_B = "rig_sfm"         # Rig-based (hybrid)
    
    def __init__(self, settings):
        self.settings = settings
        self.sphere_sfm = SphereSfMIntegrator(settings)
        
        # Import rig SfM integrator
        from .rig_colmap_integration import RigColmapIntegrator
        self.rig_sfm = RigColmapIntegrator(settings)
    
    def is_mode_a_available(self) -> bool:
        """Check if SphereSfM (Panorama SfM) is available."""
        return self.sphere_sfm.is_available()
    
    def get_available_modes(self) -> List[str]:
        """Get list of available alignment modes."""
        modes = [self.MODE_B]  # Rig SfM always available (uses pycolmap)
        if self.is_mode_a_available():
            modes.insert(0, self.MODE_A)
        return modes
    
    def get_mode_description(self, mode: str) -> str:
        """Get human-readable description of a mode."""
        descriptions = {
            self.MODE_A: "SphereSfM (Native Spherical) - Direct equirectangular matching",
            self.MODE_B: "Rig-based SfM (Recommended) - Virtual perspectives with rig constraints"
        }
        return descriptions.get(mode, "Unknown mode")
    
    def run_alignment(
        self,
        mode: str,
        frames_dir: Path,
        masks_dir: Optional[Path],
        output_dir: Path,
        progress_callback: Optional[Callable] = None
    ) -> Dict:
        """
        Run alignment with specified mode.
        
        Args:
            mode: Either MODE_A ('sphere_sfm') or MODE_B ('rig_sfm')
            frames_dir: Directory containing input images
            masks_dir: Optional mask directory
            output_dir: Output directory
            progress_callback: Optional progress callback
            
        Returns:
            Dictionary with reconstruction results
        """
        if mode == self.MODE_A:
            if not self.is_mode_a_available():
                return {
                    'success': False,
                    'error': 'SphereSfM binary not available. Please use Rig-based SfM.'
                }
            logger.info("Running Panorama SfM: SphereSfM (native spherical)")
            return self.sphere_sfm.run_alignment_mode_a(
                frames_dir, masks_dir, output_dir, progress_callback
            )
        
        elif mode == self.MODE_B:
            logger.info("Running Perspective Reconstruction: Rig-based SfM (virtual perspectives)")
            return self.rig_sfm.run_alignment(
                frames_dir, masks_dir, output_dir, progress_callback
            )
        
        else:
            return {'success': False, 'error': f'Unknown alignment mode: {mode}'}


def create_pole_mask(width: int, height: int, pole_degrees: float = 15) -> 'np.ndarray':
    """
    Create a mask to exclude polar regions from equirectangular images.
    
    Polar regions in ERP have high distortion and unreliable features.
    
    Args:
        width: Image width
        height: Image height  
        pole_degrees: Degrees from pole to mask (0-90)
        
    Returns:
        Binary mask (255 = valid, 0 = masked)
    """
    import numpy as np
    
    mask = np.ones((height, width), dtype=np.uint8) * 255
    
    # Calculate pixel rows to mask
    pole_fraction = pole_degrees / 90.0
    pole_pixels = int(height * pole_fraction / 2)
    
    # Mask top and bottom poles
    mask[:pole_pixels, :] = 0
    mask[-pole_pixels:, :] = 0
    
    return mask


def verify_spheresfm_installation(settings=None) -> Dict:
    """
    Verify SphereSfM installation and return status.
    
    Returns:
        Dictionary with installation status and details
    """
    spheresfm_path, searched_paths = resolve_spheresfm_binary_path(settings)
    
    result = {
        'installed': False,
        'path': str(spheresfm_path),
        'searched_paths': searched_paths,
        'version': None,
        'error': None,
        'supports_sphere_camera': False,
    }
    
    if not spheresfm_path.exists():
        result['error'] = f"SphereSfM binary not found at {spheresfm_path}"
        return result
    
    # Try to execute binary/help
    try:
        runnable, supports_sphere = _probe_binary(spheresfm_path)
        result['installed'] = runnable
        result['version'] = "SphereSfM/COLMAP-compatible" if runnable else None
        result['supports_sphere_camera'] = supports_sphere
        if not runnable:
            result['error'] = f"Binary exists but failed to run correctly: {spheresfm_path}"
        
    except subprocess.TimeoutExpired:
        result['error'] = "SphereSfM command timed out"
    except Exception as e:
        result['error'] = str(e)
    
    return result
