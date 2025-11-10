"""
360FrameTools - Main Application Entry Point
Unified photogrammetry preprocessing pipeline.
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('360frametools.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main application entry point"""
    
    logger.info("=" * 60)
    logger.info("Starting 360FrameTools v1.0.0")
    logger.info("=" * 60)
    
    try:
        from PyQt6.QtWidgets import QApplication
        from src.ui.main_window import MainWindow
        
        # Create Qt application
        app = QApplication(sys.argv)
        app.setApplicationName("360FrameTools")
        app.setOrganizationName("360FrameTools Development Team")
        
        # Create and show main window
        window = MainWindow()
        window.show()
        
        logger.info("Application window opened")
        
        # Run event loop
        sys.exit(app.exec())
    
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Please install dependencies: pip install -r requirements.txt")
        print("\n" + "="*60)
        print("ERROR: Missing Dependencies")
        print("="*60)
        print(f"\n{e}\n")
        print("Please install dependencies:")
        print("  pip install -r requirements.txt")
        print("\nFor GPU support:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print("="*60)
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        print(f"\nERROR: {e}\n")
        sys.exit(1)

if __name__ == '__main__':
    main()
