# PyInstaller hook for the subset of PyQt6 used by the desktop app.

datas = []
binaries = []
hiddenimports = [
    'PyQt6',
    'PyQt6.QtCore',
    'PyQt6.QtGui',
    'PyQt6.QtWidgets',
    'PyQt6.sip',
]
