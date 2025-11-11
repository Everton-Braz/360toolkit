# PyInstaller hook for PyQt6
from PyInstaller.utils.hooks import collect_all, collect_submodules

# Collect all PyQt6 modules
datas, binaries, hiddenimports = collect_all('PyQt6')

# Ensure all submodules are included
hiddenimports += collect_submodules('PyQt6')
hiddenimports += [
    'PyQt6.QtCore',
    'PyQt6.QtGui',
    'PyQt6.QtWidgets',
    'PyQt6.sip',
]
