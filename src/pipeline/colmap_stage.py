"""
COLMAP Stage Integration for 360FrameTools Pipeline.
Provides ColmapSettings dataclass and pipeline integration.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class ColmapSettings:
    """Settings for COLMAP/SphereSfM alignment stage."""
    
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
    
    # SphereSfM specific
    sphere_camera_model: bool = True  # Use SPHERE camera model
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return {
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
    Wraps ColmapIntegrator for use in batch pipeline.
    """
    
    def __init__(self, settings: ColmapSettings):
        self.settings = settings
        self._integrator = None
    
    @property
    def integrator(self):
        """Lazy load COLMAP integrator."""
        if self._integrator is None:
            try:
                from src.premium.colmap_integration import ColmapIntegrator
                self._integrator = ColmapIntegrator(self.settings)
            except ImportError:
                logger.warning("Premium COLMAP integration not available")
                self._integrator = None
        return self._integrator
    
    def is_available(self) -> bool:
        """Check if COLMAP alignment is available."""
        return self.integrator is not None and self.integrator.spheresfm_exe_path is not None
    
    def run(
        self,
        frames_dir: Path,
        masks_dir: Optional[Path],
        output_dir: Path,
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        Run COLMAP alignment on equirectangular frames.
        
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
                'error': 'COLMAP/SphereSfM not available',
                'positions': {}
            }
        
        return self.integrator.run_alignment(
            frames_dir=frames_dir,
            masks_dir=masks_dir,
            output_dir=output_dir,
            progress_callback=progress_callback
        )


# Convenience function to check SphereSfM availability
def is_spheresfm_available() -> bool:
    """Check if SphereSfM COLMAP is installed and available."""
    default_path = Path(r"C:\Users\Everton-PC\Documents\APLICATIVOS\SphereAlignment\SphereSfM-2024-12-14\colmap.exe")
    return default_path.exists()


def get_default_colmap_settings() -> ColmapSettings:
    """Get default COLMAP settings."""
    return ColmapSettings(
        sphere_alignment_path=Path(r"C:\Users\Everton-PC\Documents\APLICATIVOS\SphereAlignment"),
        quality='medium',
        matching_method='sequential',
        extract_features=True,
        match_features=True,
        build_reconstruction=True,
        dense_model=False,
        enable_propagation=True,
        use_gpu=True
    )
