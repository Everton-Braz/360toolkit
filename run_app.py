"""
Launcher script for 360toolkit
Ensures proper Python path setup
"""

import sys
import multiprocessing
from pathlib import Path

# CRITICAL: Must be at the very start for PyInstaller on Windows
# Prevents child processes from spawning new GUI windows
if __name__ == '__main__':
    multiprocessing.freeze_support()

# Add project root to Python path so 'src' can be imported as a package
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import and run main from src package
from src.main import main

if __name__ == '__main__':
    main()
