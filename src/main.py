"""
360toolkit - Main Application Entry Point
Unified photogrammetry preprocessing pipeline.
"""

import sys
import os
import multiprocessing

# CRITICAL: Must be called at the very start for PyInstaller on Windows
# This prevents child processes from spawning new GUI windows
if __name__ == '__main__':
    multiprocessing.freeze_support()

import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('360toolkit.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main application entry point"""
    
    logger.info("=" * 60)
    logger.info("Starting 360toolkit v1.1.0")
    logger.info("=" * 60)
    
    try:
        # Enable High-DPI scaling BEFORE creating QApplication
        # This is essential for 4K monitors
        os.environ['QT_ENABLE_HIGHDPI_SCALING'] = '1'
        os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '1'
        
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QFont
        from src.ui.main_window import MainWindow
        
        # Create Qt application
        app = QApplication(sys.argv)
        app.setApplicationName("360toolkit")
        app.setOrganizationName("360toolkit Development Team")
        
        # Set default font size for better readability on high-DPI
        font = app.font()
        font.setPointSize(10)  # Slightly larger default font
        app.setFont(font)
        
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
