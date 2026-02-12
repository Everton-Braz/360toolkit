"""
COLMAP Stage Integration for 360FrameTools Pipeline.
Provides ColmapSettings dataclass and pipeline integration.

Supports three alignment modes:
- Mode A (SphereSfM Direct): Native spherical feature matching, equirect output only
- Mode B (Rig SfM): Virtual perspective rendering with COLMAP rig constraints
- Mode C (Pose Transfer): SphereSfM alignment + 9-camera rig perspective extraction
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Alignment modes
ALIGNMENT_MODE_SPHERE_SFM = 'sphere_sfm'  # Mode A: Native SphereSfM (equirect only)
ALIGNMENT_MODE_RIG_SFM = 'rig_sfm'        # Mode B: Rig-based SfM
ALIGNMENT_MODE_POSE_TRANSFER = 'pose_transfer'  # Mode C: SphereSfM + Pose Transfer (9-camera rig)


@dataclass
class ColmapSettings:
    """Settings for COLMAP/SphereSfM alignment stage."""
    
    # Alignment mode (Mode A or Mode B)
    alignment_mode: str = ALIGNMENT_MODE_RIG_SFM  # Default to proven Rig SfM
    
    # Paths
    sphere_alignment_path: Optional[Path] = None
    glomap_path: Optional[Path] = None  # For future GloMAP integration
    
    # Quality settings
    quality: str = 'medium'  # 'fast', 'medium', 'high'
    
    # Matching settings
    matching_method: str = 'sequential'  # 'sequential', 'exhaustive', 'vocab_tree'
    
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
    
    # Legacy compatibility
    sphere_camera_model: bool = True  # Use SPHERE camera model (Mode A)
    use_rig_sfm: bool = True  # Use Rig SFM (Mode B) - kept for compatibility
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return {
            'alignment_mode': self.alignment_mode,
            'sphere_alignment_path': str(self.sphere_alignment_path) if self.sphere_alignment_path else None,
            'glomap_path': str(self.glomap_path) if self.glomap_path else None,
            'quality': self.quality,
            'matching_method': self.matching_method,
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
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ColmapSettings':
        """Create settings from dictionary."""
        if data.get('sphere_alignment_path'):
            data['sphere_alignment_path'] = Path(data['sphere_alignment_path'])
        if data.get('glomap_path'):
            data['glomap_path'] = Path(data['glomap_path'])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class ColmapStage:
    """
    Pipeline stage for COLMAP/SphereSfM alignment.
    
    Supports three modes:
    - Mode A (sphere_sfm): Direct spherical feature matching using SphereSfM binary (equirect output)
    - Mode B (rig_sfm): Virtual perspectives with COLMAP rig constraints
    - Mode C (pose_transfer): SphereSfM alignment + 9-camera rig perspective extraction with pose transfer
    """
    
    def __init__(self, settings: ColmapSettings):
        self.settings = settings
        self._integrator = None
        self._mode = settings.alignment_mode
    
    @property
    def integrator(self):
        """Lazy load appropriate integrator based on alignment mode."""
        if self._integrator is None:
            try:
                if self._mode == ALIGNMENT_MODE_SPHERE_SFM:
                    # Mode A: SphereSfM native (equirect only)
                    from src.premium.sphere_sfm_integration import SphereSfMIntegrator
                    self._integrator = SphereSfMIntegrator(self.settings)
                    logger.info("Using Mode A: SphereSfM Direct (equirectangular output)")
                elif self._mode == ALIGNMENT_MODE_POSE_TRANSFER:
                    # Mode C: SphereSfM + Pose Transfer (9-camera rig)
                    from src.premium.pose_transfer_integration import PoseTransferIntegrator
                    self._integrator = PoseTransferIntegrator(self.settings)
                    logger.info("Using Mode C: SphereSfM + Pose Transfer (9-camera rig)")
                else:
                    # Mode B: Rig-based SfM (default)
                    from src.premium.rig_colmap_integration import RigColmapIntegrator
                    self._integrator = RigColmapIntegrator(self.settings)
                    logger.info("Using Mode B: Rig-based SfM (virtual perspectives)")
            except (ImportError, NameError) as e:
                logger.warning(f"COLMAP integration not available: {e}")
                self._integrator = None
        return self._integrator
    
    def is_available(self) -> bool:
        """Check if alignment is available for current mode."""
        if self._mode == ALIGNMENT_MODE_SPHERE_SFM:
            # Both require SphereSfM binary
            try:
                from src.premium.sphere_sfm_integration import verify_spheresfm_installation
                status = verify_spheresfm_installation()
                return status.get('installed', False)
            except Exception:
                return False
        elif self._mode == ALIGNMENT_MODE_POSE_TRANSFER:
            try:
                from src.premium.sphere_sfm_integration import verify_spheresfm_installation
                status = verify_spheresfm_installation()
                if not status.get('installed', False):
                    return False
                import pycolmap
                return pycolmap is not None
            except Exception:
                return False
        else:
            # Rig SfM just needs pycolmap
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
        
        # Route to appropriate method based on mode
        if self._mode == ALIGNMENT_MODE_SPHERE_SFM:
            # Mode A: SphereSfM native (equirect only)
            return self.integrator.run_alignment_mode_a(
                frames_dir=frames_dir,
                masks_dir=masks_dir,
                output_dir=output_dir,
                progress_callback=progress_callback
            )
        elif self._mode == ALIGNMENT_MODE_POSE_TRANSFER:
            # Mode C: SphereSfM + Pose Transfer
            return self.integrator.run_alignment(
                frames_dir=frames_dir,
                masks_dir=masks_dir,
                output_dir=output_dir,
                progress_callback=progress_callback
            )
        else:
            # Mode B: Rig-based SfM
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


def get_default_colmap_settings(alignment_mode: str = ALIGNMENT_MODE_RIG_SFM) -> ColmapSettings:
    """Get default COLMAP settings for specified mode."""
    return ColmapSettings(
        alignment_mode=alignment_mode,
        quality='medium',
        matching_method='sequential',
        extract_features=True,
        match_features=True,
        build_reconstruction=True,
        dense_model=False,
        enable_propagation=True,
        use_gpu=True,
        use_rig_sfm=(alignment_mode == ALIGNMENT_MODE_RIG_SFM)
    )
