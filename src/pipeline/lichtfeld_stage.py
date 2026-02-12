
import logging
from pathlib import Path
import subprocess
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class LichtfeldTrainingStage:
    """
    Stage for launching Lichtfeld Studio training.
    """
    
    def __init__(self, lichtfeld_path: Optional[Path] = None):
        if lichtfeld_path:
             self.lichtfeld_path = Path(lichtfeld_path)
        else:
             # Try default locations or check system path
             self.lichtfeld_path = self._find_lichtfeld_exe()
             
    def _find_lichtfeld_exe(self) -> Optional[Path]:
        potential_paths = [
            Path(r"C:\Program Files\Lichtfeld Studio\Lichtfeld Studio.exe"),
            Path(r"C:\Users\Everton-PC\AppData\Local\Programs\Lichtfeld Studio\Lichtfeld Studio.exe")
        ]
        for p in potential_paths:
            if p.exists():
                return p
        return None

    def run(self, colmap_model_path: Path, images_path: Path) -> Dict[str, Any]:
        """
        Launch Lichtfeld Studio for training.
        """
        if not self.lichtfeld_path or not self.lichtfeld_path.exists():
            return {'success': False, 'error': 'Lichtfeld Studio executable not found'}
            
        logger.info(f"Launching Lichtfeld Studio from {self.lichtfeld_path}")
        logger.info(f"  Model: {colmap_model_path}")
        logger.info(f"  Images: {images_path}")
        
        # NOTE: Lichtfeld Studio CLI arguments are hypothetical unless documented.
        # Often GUI apps verify if a folder is passed as argument to open it.
        # We will try passing the model folder.
        
        try:
            # Launch async (don't wait for user to close it)
            subprocess.Popen([str(self.lichtfeld_path), str(colmap_model_path)])
            return {'success': True, 'message': 'Lichtfeld Studio launched'}
        except Exception as e:
            logger.error(f"Failed to launch Lichtfeld Studio: {e}")
            return {'success': False, 'error': str(e)}
