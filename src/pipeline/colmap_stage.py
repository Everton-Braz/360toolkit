"""
COLMAP Stage Integration for 360FrameTools Pipeline.
Provides ColmapSettings dataclass and pipeline integration.

Supports two reconstruction workflows:
- Panorama SfM: Direct spherical feature matching on equirectangular images (SphereSfM)
- Perspective Reconstruction: Split to perspectives, then COLMAP GPU SfM (incremental or global_mapper)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Reconstruction workflows
ALIGNMENT_MODE_PANORAMA_SFM = 'panorama_sfm'                 # SphereSfM on equirectangular images
ALIGNMENT_MODE_PERSPECTIVE = 'perspective_reconstruction'     # COLMAP GPU on perspective images

# Legacy aliases (backward compatibility with saved configs)
ALIGNMENT_MODE_SPHERE_SFM = ALIGNMENT_MODE_PANORAMA_SFM      # old "Mode A"
ALIGNMENT_MODE_RIG_SFM = ALIGNMENT_MODE_PERSPECTIVE           # old "Mode B"
ALIGNMENT_MODE_POSE_TRANSFER = ALIGNMENT_MODE_PERSPECTIVE     # old "Mode C" → merged into Perspective


@dataclass
class ColmapSettings:
    """Settings for COLMAP/SphereSfM reconstruction."""
    
    # Reconstruction workflow
    alignment_mode: str = ALIGNMENT_MODE_PERSPECTIVE  # Default to Perspective Reconstruction
    
    # Paths
    sphere_alignment_path: Optional[Path] = None
    colmap_path: Optional[Path] = None
    mapping_backend: str = 'glomap'  # 'colmap' | 'glomap' (legacy key; uses COLMAP global_mapper)
    
    # Quality settings
    quality: str = 'medium'  # 'fast', 'medium', 'high'
    
    # Matching settings
    matching_method: str = 'sequential'  # 'sequential', 'exhaustive', 'vocab_tree'
    use_lightglue_aliked: bool = True
    camera_grouping: str = 'per_folder'  # 'single' | 'per_folder' | 'per_image'
    prefer_colmap_learned: bool = False
    require_learned_pipeline: bool = False
    enable_hloc_fallback: bool = True
    reuse_colmap_database: bool = True
    
    # Pipeline steps
    extract_features: bool = True
    match_features: bool = True
    build_reconstruction: bool = True
    dense_model: bool = False
    
    # Advanced settings
    enable_propagation: bool = True  # Propagate positions to perspectives
    export_3dgs: bool = False  # Export for 3D Gaussian Splatting
    
    # GPU settings
    use_gpu: bool = True
    gpu_index: int = 0
    
    # SphereSfM-specific parameters (used by sphere_sfm_integration.py)
    spheresfm_camera_model: str = 'SPHERE'          # SPHERE | SIMPLE_SPHERE | FULL_OPENCV | OPENCV_FISHEYE
    spheresfm_feature_extractor: str = 'SIFT'        # SIFT | ALIKED
    spheresfm_use_gpu: bool = False                  # False = CPU (bundled binary has CUDA mismatch)
    spheresfm_matching_method: str = 'sequential'    # sequential | exhaustive | vocab_tree
    spheresfm_max_image_size: int = 3200
    spheresfm_max_num_features: int = 8192
    spheresfm_sequential_overlap: int = 10
    spheresfm_min_num_matches: int = 15
    spheresfm_extra_args: str = ''

    # Legacy compatibility
    sphere_camera_model: bool = True  # Use SPHERE camera model (Panorama SfM)
    use_rig_sfm: bool = True  # Legacy field — kept for config compat
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return {
            'alignment_mode': self.alignment_mode,
            'sphere_alignment_path': str(self.sphere_alignment_path) if self.sphere_alignment_path else None,
            'colmap_path': str(self.colmap_path) if self.colmap_path else None,
            'mapping_backend': self.mapping_backend,
            'quality': self.quality,
            'matching_method': self.matching_method,
            'use_lightglue_aliked': self.use_lightglue_aliked,
            'camera_grouping': self.camera_grouping,
            'prefer_colmap_learned': self.prefer_colmap_learned,
            'require_learned_pipeline': self.require_learned_pipeline,
            'enable_hloc_fallback': self.enable_hloc_fallback,
            'reuse_colmap_database': self.reuse_colmap_database,
            'extract_features': self.extract_features,
            'match_features': self.match_features,
            'build_reconstruction': self.build_reconstruction,
            'dense_model': self.dense_model,
            'enable_propagation': self.enable_propagation,
            'export_3dgs': self.export_3dgs,
            'use_gpu': self.use_gpu,
            'gpu_index': self.gpu_index,
            'sphere_camera_model': self.sphere_camera_model,
            'use_rig_sfm': self.use_rig_sfm,
            # SphereSfM-specific params
            'spheresfm_camera_model': self.spheresfm_camera_model,
            'spheresfm_feature_extractor': self.spheresfm_feature_extractor,
            'spheresfm_use_gpu': self.spheresfm_use_gpu,
            'spheresfm_matching_method': self.spheresfm_matching_method,
            'spheresfm_max_image_size': self.spheresfm_max_image_size,
            'spheresfm_max_num_features': self.spheresfm_max_num_features,
            'spheresfm_sequential_overlap': self.spheresfm_sequential_overlap,
            'spheresfm_min_num_matches': self.spheresfm_min_num_matches,
            'spheresfm_extra_args': self.spheresfm_extra_args,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ColmapSettings':
        """Create settings from dictionary."""
        if data.get('sphere_alignment_path'):
            data['sphere_alignment_path'] = Path(data['sphere_alignment_path'])
        if data.get('colmap_path'):
            data['colmap_path'] = Path(data['colmap_path'])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class ColmapStage:
    """
    Pipeline stage for 3D Reconstruction (COLMAP/SphereSfM).
    
    Supports two workflows:
    - Panorama SfM (panorama_sfm): Direct spherical feature matching using SphereSfM (equirect output)
    - Perspective Reconstruction (perspective_reconstruction): COLMAP GPU on perspective images
    """
    
    def __init__(self, settings: ColmapSettings):
        self.settings = settings
        self._integrator = None
        self._mode = self._normalize_mode(settings.alignment_mode)
    
    @staticmethod
    def _normalize_mode(mode: str) -> str:
        """Normalize legacy mode names to current workflow names."""
        legacy_map = {
            'sphere_sfm': ALIGNMENT_MODE_PANORAMA_SFM,
            'rig_sfm': ALIGNMENT_MODE_PERSPECTIVE,
            'pose_transfer': ALIGNMENT_MODE_PERSPECTIVE,  # Mode C → Perspective
        }
        return legacy_map.get(mode, mode)
    
    @property
    def integrator(self):
        """Lazy load appropriate integrator based on reconstruction workflow."""
        if self._integrator is None:
            try:
                if self._mode == ALIGNMENT_MODE_PANORAMA_SFM:
                    # Panorama SfM: SphereSfM native (equirect only)
                    from src.premium.sphere_sfm_integration import SphereSfMIntegrator
                    self._integrator = SphereSfMIntegrator(self.settings)
                    logger.info("Using Panorama SfM: SphereSfM (equirectangular images)")
                else:
                    # Perspective Reconstruction: Rig-based SfM (default)
                    from src.premium.rig_colmap_integration import RigColmapIntegrator
                    self._integrator = RigColmapIntegrator(self.settings)
                    logger.info("Using Perspective Reconstruction: COLMAP (perspective images)")
            except (ImportError, NameError) as e:
                logger.warning(f"Reconstruction integration not available: {e}")
                self._integrator = None
        return self._integrator
    
    def is_available(self) -> bool:
        """Check if reconstruction is available for current workflow."""
        if self._mode == ALIGNMENT_MODE_PANORAMA_SFM:
            # Panorama SfM requires SphereSfM binary
            try:
                from src.premium.sphere_sfm_integration import verify_spheresfm_installation
                status = verify_spheresfm_installation()
                return status.get('installed', False)
            except Exception:
                return False
        else:
            # Perspective Reconstruction needs rig integrator
            return self.integrator is not None
    
    def run(
        self,
        frames_dir: Path,
        masks_dir: Optional[Path],
        output_dir: Path,
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        Run alignment on equirectangular frames using selected mode.
        
        Args:
            frames_dir: Directory containing equirectangular frames
            masks_dir: Directory containing mask images (optional)
            output_dir: Output directory for COLMAP results
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dict with alignment results including camera positions
        """
        if not self.is_available():
            return {
                'success': False,
                'error': f'Alignment mode "{self._mode}" not available',
                'positions': {}
            }

        if self.integrator is None:
            return {
                'success': False,
                'error': f'Failed to initialize integrator for mode "{self._mode}"',
                'positions': {}
            }
        
        # Route to appropriate method based on workflow
        if self._mode == ALIGNMENT_MODE_PANORAMA_SFM:
            # Panorama SfM: SphereSfM native (equirect only)
            return self.integrator.run_alignment_mode_a(
                frames_dir=frames_dir,
                masks_dir=masks_dir,
                output_dir=output_dir,
                progress_callback=progress_callback
            )
        else:
            # Perspective Reconstruction
            return self.integrator.run_alignment(
                frames_dir=frames_dir,
                masks_dir=masks_dir,
                output_dir=output_dir,
                progress_callback=progress_callback
            )


# Convenience function to check SphereSfM availability
def is_spheresfm_available() -> bool:
    """Check if SphereSfM COLMAP binary is installed and available."""
    try:
        from src.premium.sphere_sfm_integration import verify_spheresfm_installation
        status = verify_spheresfm_installation()
        return status.get('installed', False)
    except Exception:
        return False


def get_default_colmap_settings(alignment_mode: str = ALIGNMENT_MODE_PERSPECTIVE) -> ColmapSettings:
    """Get default COLMAP settings for specified workflow."""
    # Normalize legacy mode names
    alignment_mode = ColmapStage._normalize_mode(alignment_mode)
    return ColmapSettings(
        alignment_mode=alignment_mode,
        mapping_backend='glomap',
        quality='medium',
        matching_method='sequential',
        camera_grouping='per_folder',
        prefer_colmap_learned=False,
        require_learned_pipeline=False,
        enable_hloc_fallback=True,
        reuse_colmap_database=True,
        extract_features=True,
        match_features=True,
        build_reconstruction=True,
        dense_model=False,
        enable_propagation=True,
        use_gpu=True,
        use_rig_sfm=(alignment_mode == ALIGNMENT_MODE_PERSPECTIVE)
    )
