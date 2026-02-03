"""
COLMAP alignment bridge for 360FrameTools Pro Edition.
DO NOT COMMIT TO GITHUB!

Direct SphereSfM/COLMAP integration (built-in).

Workflow:
1. Take equirectangular frames + masks
2. Run SphereSfM COLMAP directly (feature extraction, matching, mapper)
3. Parse COLMAP output files (images.txt)
4. Extract positions for each frame
5. Embed in frame metadata
6. Split propagates positions to perspectives
"""

import subprocess
import logging
from pathlib import Path
from typing import Dict, Optional, Callable, Tuple
import json
from src.pipeline.export_formats import LichtfeldExporter
import time

logger = logging.getLogger(__name__)


class ColmapIntegrator:
    """
    Direct SphereSfM/COLMAP integration for sphere camera alignment.
    Calls COLMAP executable directly with proper SPHERE camera parameters.
    
    Now with GloMAP support for 10-100x faster global SfM!
    GloMAP is preferred when available, with COLMAP incremental as fallback.
    """
    
    def __init__(self, settings):
        self.settings = settings
        self.spheresfm_exe_path = self._find_spheresfm_exe()
        self.glomap_exe_path = self._find_glomap_exe()
        
        if not self.spheresfm_exe_path:
            logger.error("SphereSFM COLMAP executable not found")
        
        if self.glomap_exe_path:
            logger.info(f"✓ GloMAP found: {self.glomap_exe_path} (10-100x faster mapper)")
        else:
            logger.info("GloMAP not found - will use COLMAP incremental mapper (slower)")
    
    def _find_spheresfm_exe(self) -> Optional[Path]:
        """Find SphereSfM COLMAP executable"""
        # Try user-specified path first
        if hasattr(self.settings, 'sphere_alignment_path') and self.settings.sphere_alignment_path:
            colmap_exe = self.settings.sphere_alignment_path / "SphereSfM-2024-12-14" / "colmap.exe"
            if colmap_exe.exists():
                return colmap_exe
        
        # Try default location
        default_base = Path(r"C:\Users\Everton-PC\Documents\APLICATIVOS\SphereAlignment")
        if default_base.exists():
            colmap_exe = default_base / "SphereSfM-2024-12-14" / "colmap.exe"
            if colmap_exe.exists():
                return colmap_exe
        
        return None
    
    def _find_glomap_exe(self) -> Optional[Path]:
        """
        Find GloMAP executable.
        
        GloMAP provides global SfM that is 10-100x faster than COLMAP incremental.
        Search order:
        1. Settings (glomap_path from SettingsManager)
        2. Standard location in Documents/APLICATIVOS/GloMAP
        3. System PATH
        """
        # Try settings first
        if hasattr(self.settings, 'glomap_path') and self.settings.glomap_path:
            glomap_exe = Path(self.settings.glomap_path)
            if glomap_exe.exists():
                logger.debug(f"GloMAP from settings: {glomap_exe}")
                return glomap_exe
        
        # Try SettingsManager
        try:
            from src.config.settings import get_settings
            settings_mgr = get_settings()
            glomap_path = settings_mgr.get_glomap_path()
            if glomap_path and glomap_path.exists():
                logger.debug(f"GloMAP from SettingsManager: {glomap_path}")
                return glomap_path
        except Exception as e:
            logger.debug(f"Could not get GloMAP from SettingsManager: {e}")
        
        # Try standard locations
        standard_paths = [
            Path(r"C:\Users\Everton-PC\Documents\APLICATIVOS\GloMAP\bin\glomap.exe"),
            Path(r"C:\Users\Everton-PC\Documents\APLICATIVOS\GloMAP_GUI\glomap\build\bin\Release\glomap.exe"),
            Path(r"C:\Program Files\GloMAP\bin\glomap.exe"),
        ]
        for path in standard_paths:
            if path.exists():
                logger.debug(f"GloMAP from standard location: {path}")
                return path
        
        # Try system PATH
        import shutil
        glomap_in_path = shutil.which('glomap')
        if glomap_in_path:
            return Path(glomap_in_path)
        
        return None
    
    def run_alignment(
        self,
        frames_dir: Path,
        masks_dir: Path,
        output_dir: Path,
        progress_callback: Optional[Callable] = None
    ) -> Dict:
        """
        Run COLMAP alignment on equirectangular frames.
        
        Direct COLMAP execution with proper camera_params for SPHERE model.
        
        Steps:
        1. Detect image dimensions and calculate camera_params
        2. Create database
        3. Extract features (with SPHERE model + camera_params)
        4. Match features (sequential or exhaustive)
        5. Run mapper (incremental SfM)
        6. Parse camera positions from images.txt
        """
        if not self.spheresfm_exe_path:
            return {'success': False, 'error': 'SphereSFM executable not found'}
        
        # Detect image dimensions from first frame
        image_files = list(frames_dir.glob('*.png')) + list(frames_dir.glob('*.jpg'))
        if not image_files:
            return {'success': False, 'error': 'No images found in input directory'}
        
        from PIL import Image
        first_image = Image.open(image_files[0])
        width, height = first_image.size
        first_image.close()
        
        # SPHERE camera model - DO NOT use camera_params (SphereAlignment working approach)
        logger.info(f"Image dimensions: {width}x{height}")
        logger.info(f"Camera model: SPHERE (no camera_params - following SphereAlignment)")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        db_path = output_dir / "database.db"
        sparse_path = output_dir / "sparse"
        sparse_path.mkdir(exist_ok=True)
        
        matching_method = getattr(self.settings, 'matching_method', 'sequential')
        use_gpu = getattr(self.settings, 'use_gpu', True)
        
        logger.info(f"Running COLMAP alignment on equirectangular frames: {frames_dir}")
        logger.info(f"  Masks: {masks_dir}")
        logger.info(f"  Output: {output_dir}")
        logger.info(f"  SphereSFM: {self.spheresfm_exe_path}")
        logger.info(f"  Matching method: {matching_method}")
        
        try:
            # Step 1: Create database
            if progress_callback:
                progress_callback("Creating COLMAP database...")
            
            self._run_colmap_command(
                ["database_creator", "--database_path", str(db_path)],
                "Database creation"
            )
            
            # Step 2: Extract features with SPHERE camera model (SphereAlignment approach)
            # KEY: Use SPHERE model WITH calculated camera_params
            if progress_callback:
                progress_callback("Extracting features...")
            
            # Use SphereAlignment's exact approach: SPHERE model, minimal SIFT params
            use_gpu = "0"  # CPU mode - RTX 5070 Ti sm_120 not supported by SphereSfM CUDA build
            camera_params = f"1,{width/2},{height/2}"
            
            feature_cmd = [
                "feature_extractor",
                "--database_path", str(db_path),
                "--image_path", str(frames_dir),
                "--ImageReader.camera_model", "SPHERE",
                "--ImageReader.camera_params", camera_params,
                "--ImageReader.single_camera", "1",      # Assume single camera for video
                "--SiftExtraction.use_gpu", use_gpu
            ]
            
            # Add mask path if available
            if masks_dir and masks_dir.exists():
                feature_cmd.extend(["--ImageReader.mask_path", str(masks_dir)])
            
            logger.info(f"Feature extraction command: {' '.join(feature_cmd)}")
            
            self._run_colmap_command(feature_cmd, "Feature extraction")
            
            # Step 3: Match features
            if progress_callback:
                progress_callback(f"Matching features ({matching_method})...")
            
            if matching_method == 'sequential':
                overlap = getattr(self.settings, 'sequential_overlap', 10)  # SphereAlignment default
                match_cmd = [
                    "sequential_matcher",
                    "--database_path", str(db_path),
                    "--SequentialMatching.overlap", str(overlap),
                    "--SequentialMatching.loop_detection", "0",  # SphereAlignment disables loop detection
                    "--SiftMatching.max_error", "4.0",
                    "--SiftMatching.min_num_inliers", "15",
                    "--SiftMatching.use_gpu", use_gpu
                ]
            else:  # exhaustive
                match_cmd = [
                    "exhaustive_matcher",
                    "--database_path", str(db_path),
                    "--SiftMatching.max_error", "4.0",
                    "--SiftMatching.min_num_inliers", "15",
                    "--SiftMatching.use_gpu", use_gpu
                ]
            
            self._run_colmap_command(match_cmd, "Feature matching")
            
            # Step 4: Run mapper (GloMAP global SfM if available, else COLMAP incremental)
            if progress_callback:
                progress_callback("Building 3D reconstruction...")
            
            # Try GloMAP first (10-100x faster global SfM)
            use_glomap = self.glomap_exe_path is not None
            glomap_success = False
            
            if use_glomap:
                logger.info("Using GloMAP global mapper (10-100x faster than COLMAP incremental)")
                try:
                    glomap_success = self._run_glomap_mapper(
                        db_path, frames_dir, sparse_path, progress_callback
                    )
                except Exception as e:
                    logger.warning(f"GloMAP failed: {e}. Falling back to COLMAP incremental mapper.")
                    glomap_success = False
            
            # Fallback to COLMAP incremental mapper if GloMAP not available or failed
            if not glomap_success:
                if use_glomap:
                    logger.info("Falling back to COLMAP incremental mapper...")
                else:
                    logger.info("Using COLMAP incremental mapper (GloMAP not available)")
                
                mapper_cmd = [
                    "mapper",
                    "--database_path", str(db_path),
                    "--image_path", str(frames_dir),
                    "--output_path", str(sparse_path),
                    "--Mapper.sphere_camera", "1",  # Enable spherical camera mode (Critical)
                    "--Mapper.init_min_tri_angle", "0.1", # Extremely low angle for video
                    "--Mapper.ba_refine_focal_length", "0",
                    "--Mapper.ba_refine_principal_point", "0",
                    "--Mapper.ba_refine_extra_params", "0",
                    "--Mapper.abs_pose_min_num_inliers", "10", # Reduced from 15
                    "--Mapper.init_min_num_inliers", "25",     # Reduced from 50
                    "--Mapper.init_max_error", "12.0",         # Relaxed
                    "--Mapper.abs_pose_max_error", "12.0",     # Relaxed 
                    "--Mapper.min_num_matches", "15",
                    "--Mapper.min_model_size", "3",            # Keep tiny reconstructions
                    "--Mapper.filter_max_reproj_error", "12.0", # Relaxed
                    "--Mapper.max_reg_trials", "50",
                ]
                
                self._run_colmap_command(mapper_cmd, "COLMAP Mapper")
            
            # Check for output (COLMAP/GloMAP creates binary files by default)
            colmap_dir = sparse_path / "0"
            if not colmap_dir.exists():
                return {'success': False, 'error': 'COLMAP reconstruction failed - no output model'}
            
            # Check for images file (binary or text format)
            images_bin = colmap_dir / "images.bin"
            images_txt = colmap_dir / "images.txt"
            
            if not images_bin.exists() and not images_txt.exists():
                return {'success': False, 'error': 'No images file found - reconstruction incomplete'}
            
            # Convert binary to text format if needed
            if images_bin.exists() and not images_txt.exists():
                logger.info("Converting COLMAP binary model to text format...")
                convert_cmd = [
                    "model_converter",
                    "--input_path", str(colmap_dir),
                    "--output_path", str(colmap_dir),
                    "--output_type", "TXT"
                ]
                self._run_colmap_command(convert_cmd, "Model converter")
            
            # Parse positions
            positions = self._parse_images_txt(images_txt)
            
            if not positions:
                return {'success': False, 'error': 'No camera positions extracted'}
            
            logger.info(f"✓ COLMAP aligned {len(positions)} frames")
            
            return {
                'success': True,
                'num_aligned': len(positions),
                'positions': positions,
                'colmap_output': colmap_dir,
                'frames_dir': frames_dir,
                'confidence': self._calculate_confidence(positions)
            }
        
        except subprocess.CalledProcessError as e:
            error_msg = f"COLMAP command failed: {e.stderr if e.stderr else str(e)}"
            
            # Handle specific error codes
            if e.returncode == 3221226505 or '3221226505' in str(e):
                error_msg = (
                    "Feature matching crashed (error 3221226505). This usually means:\n"
                    "1. Not enough matching features between frames (try closer frame intervals)\n"
                    "2. GPU memory issue (try reducing resolution or using sequential matching)\n"
                    "3. CUDA/GPU driver issue\n\n"
                    "Suggestions:\n"
                    "• Increase FPS (extract more frames, e.g., 5-10 FPS instead of 2-4 FPS)\n"
                    "• Use sequential matching instead of exhaustive\n"
                    "• Reduce image resolution in Stage 1 (try 4K instead of 8K)\n"
                    "• Update GPU drivers"
                )
            
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}
        
        except Exception as e:
            logger.error(f"Reconstruction error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'success': False, 'error': str(e)}
    
    def _run_colmap_command(self, args: list, step_name: str):
        """Run a COLMAP command and handle output"""
        cmd = [str(self.spheresfm_exe_path)] + args
        logger.info(f"  {step_name}...")
        logger.debug(f"  Command: {' '.join(cmd)}")
        
        start_time = time.time()
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        elapsed = time.time() - start_time
        logger.info(f"  ✓ {step_name} completed in {elapsed:.1f}s")
        
        # Log output for debugging
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    logger.debug(f"  STDOUT: {line}")
        if result.stderr:
            for line in result.stderr.strip().split('\n'):
                if line.strip():
                    logger.warning(f"  STDERR: {line}")
        
        return result
    
    def _run_glomap_mapper(
        self, 
        db_path: Path, 
        frames_dir: Path, 
        sparse_path: Path,
        progress_callback: Optional[Callable] = None
    ) -> bool:
        """
        Run GloMAP global mapper.
        
        GloMAP is 10-100x faster than COLMAP incremental mapper and often
        produces better results for large-scale scenes.
        
        Args:
            db_path: Path to COLMAP database (with features and matches)
            frames_dir: Path to images
            sparse_path: Output path for reconstruction
            progress_callback: Optional progress callback
            
        Returns:
            True if successful, False otherwise
        """
        if not self.glomap_exe_path:
            return False
        
        logger.info(f"  GloMAP mapper...")
        logger.info(f"    Database: {db_path}")
        logger.info(f"    Images: {frames_dir}")
        logger.info(f"    Output: {sparse_path}")
        
        if progress_callback:
            progress_callback("Running GloMAP global mapper...")
        
        start_time = time.time()
        
        # GloMAP command - uses COLMAP database format
        # Note: GloMAP outputs to sparse/0/ folder automatically
        glomap_cmd = [
            str(self.glomap_exe_path),
            "mapper",
            "--database_path", str(db_path),
            "--image_path", str(frames_dir),
            "--output_path", str(sparse_path),
        ]
        
        logger.debug(f"  GloMAP command: {' '.join(glomap_cmd)}")
        
        try:
            result = subprocess.run(
                glomap_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            elapsed = time.time() - start_time
            logger.info(f"  ✓ GloMAP mapper completed in {elapsed:.1f}s")
            
            # Log output for debugging
            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        logger.debug(f"  GLOMAP STDOUT: {line}")
            if result.stderr:
                for line in result.stderr.strip().split('\n'):
                    if line.strip() and 'error' in line.lower():
                        logger.warning(f"  GLOMAP STDERR: {line}")
            
            # Verify output exists
            output_model = sparse_path / "0"
            if output_model.exists():
                logger.info(f"  ✓ GloMAP output verified at {output_model}")
                return True
            else:
                logger.warning(f"  GloMAP did not create output at {output_model}")
                return False
                
        except subprocess.CalledProcessError as e:
            logger.warning(f"  GloMAP mapper failed (exit code {e.returncode})")
            if e.stderr:
                logger.warning(f"  Error: {e.stderr[:500]}")
            return False
        except Exception as e:
            logger.warning(f"  GloMAP mapper error: {e}")
            return False
    
    def _parse_images_txt(self, images_txt: Path) -> Dict[str, Tuple[float, float, float]]:
        """
        Parse COLMAP images.txt from SphereAlignment output.
        
        Format:
        # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        1 0.1 0.2 0.3 0.4 1.234 5.678 0.123 1 frame_001.jpg
        
        Returns:
            {frame_001.jpg: (1.234, 5.678, 0.123), ...}
        """
        positions = {}
        
        with open(images_txt, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) < 10:
                    continue
                
                try:
                    # TX, TY, TZ at indices 5, 6, 7
                    tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
                    frame_name = parts[-1]  # Last element is image name
                    
                    positions[frame_name] = (tx, ty, tz)
                    logger.debug(f"{frame_name}: ({tx:.3f}, {ty:.3f}, {tz:.3f})")
                
                except (ValueError, IndexError) as e:
                    logger.warning(f"Parse error: {e}")
        
        return positions
    
    def _calculate_confidence(self, positions: Dict) -> float:
        """Calculate alignment confidence based on position spread"""
        if not positions:
            return 0.0
        
        import numpy as np
        coords = np.array(list(positions.values()))
        std = np.std(coords, axis=0).mean()
        return min(1.0, std / 10.0)
    
    def extract_positions(self, alignment_results: Dict) -> Dict[str, Tuple[float, float, float]]:
        """Extract positions from alignment results"""
        return alignment_results.get('positions', {})
    
    def embed_positions_in_frames(
        self,
        positions: Dict[str, Tuple[float, float, float]],
        frames_dir: Path
    ):
        """
        Embed COLMAP positions into equirectangular frame metadata.
        
        When frames are later split to perspectives, perspectives will
        inherit positions from their parent frame.
        """
        from src.pipeline.metadata_handler import MetadataHandler
        
        metadata_handler = MetadataHandler()
        
        logger.info(f"Embedding positions in {len(positions)} frames")
        
        for frame_name, position in positions.items():
            frame_path = frames_dir / frame_name
            if not frame_path.exists():
                logger.warning(f"Frame not found: {frame_name}")
                continue
            
            metadata = {
                'colmap_position': {
                    'x': float(position[0]),
                    'y': float(position[1]),
                    'z': float(position[2])
                },
                'colmap_aligned': True,
                'position_source': 'sphere_alignment_colmap'
            }
            
            metadata_handler.embed_metadata(frame_path, metadata)
            logger.debug(f"Embedded position in {frame_name}")
    
    def export_for_3dgs(self, alignment_results: Dict, output_dir: Path):
        """Export for 3D Gaussian Splatting (Lichtfield Studio)"""
        if not alignment_results.get('success'):
            logger.error("Cannot export - alignment failed")
            return
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        colmap_output = alignment_results.get('colmap_output')
        if not colmap_output or not colmap_output.exists():
            logger.error("No COLMAP output found for export")
            return
            
        images_dir = alignment_results.get('frames_dir')
        if not images_dir:
            logger.error("Source frames directory missing from alignment results")
            return
            
        logger.info(f"Exporting to Lichtfeld Studio format at {output_dir}")
        
        try:
            exporter = LichtfeldExporter(str(colmap_output), str(output_dir))
            success = exporter.export(str(images_dir), fix_rotation=True)
            
            if success:
                logger.info("Lichtfeld Studio export complete")
            else:
                logger.error("Lichtfeld Studio export failed")
                
        except Exception as e:
            logger.error(f"Error during Lichtfeld export: {e}")
