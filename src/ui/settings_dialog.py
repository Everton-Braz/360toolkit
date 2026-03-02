"""
360FrameTools - Settings Dialog
UI for configuring SDK and FFmpeg paths, and other preferences.
"""

import logging
from pathlib import Path

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QLineEdit, QFileDialog, QCheckBox,
    QMessageBox, QTextEdit, QTabWidget, QWidget, QFormLayout, QComboBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


class SettingsDialog(QDialog):
    """
    Settings dialog for configuring application preferences.
    
    Features:
    - SDK path configuration with auto-detection
    - FFmpeg path configuration with auto-detection
    - Path validation and testing
    - Recent files management
    """
    
    settings_changed = pyqtSignal()  # Emitted when settings are saved
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.settings = get_settings()
        
        self.setWindowTitle("Settings - 360FrameTools")
        self.setMinimumSize(700, 600)
        self.resize(800, 700)
        
        self.init_ui()
        self.load_current_settings()
    
    def init_ui(self):
        """Initialize the settings dialog UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        
        # Tabs for different settings categories
        tabs = QTabWidget()
        tabs.addTab(self.create_paths_tab(), "Paths & Detection")
        tabs.addTab(self.create_directories_tab(), "Directories")
        tabs.addTab(self.create_info_tab(), "System Info")
        
        layout.addWidget(tabs)
        
        # Bottom buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.reset_btn = QPushButton("Reset to Defaults")
        self.reset_btn.clicked.connect(self.reset_settings)
        button_layout.addWidget(self.reset_btn)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)
        
        self.save_btn = QPushButton("Save Settings")
        self.save_btn.setDefault(True)
        self.save_btn.clicked.connect(self.save_settings)
        button_layout.addWidget(self.save_btn)
        
        layout.addLayout(button_layout)
    
    def create_paths_tab(self) -> QWidget:
        """Create the paths configuration tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        
        # Auto-detection settings
        auto_group = QGroupBox("Automatic Detection")
        auto_layout = QVBoxLayout()
        
        self.auto_detect_checkbox = QCheckBox("Auto-detect paths on startup")
        self.auto_detect_checkbox.setToolTip(
            "Automatically search for SDK and FFmpeg when the application starts"
        )
        auto_layout.addWidget(self.auto_detect_checkbox)
        
        detect_btn_layout = QHBoxLayout()
        detect_btn_layout.addStretch()
        self.detect_now_btn = QPushButton("Detect Now")
        self.detect_now_btn.setToolTip("Run path detection immediately")
        self.detect_now_btn.clicked.connect(self.detect_paths_now)
        detect_btn_layout.addWidget(self.detect_now_btn)
        auto_layout.addLayout(detect_btn_layout)
        
        auto_group.setLayout(auto_layout)
        layout.addWidget(auto_group)
        
        # SDK path configuration
        sdk_group = QGroupBox("Insta360 MediaSDK Path")
        sdk_layout = QVBoxLayout()
        
        # SDK path input
        sdk_path_layout = QHBoxLayout()
        self.sdk_path_edit = QLineEdit()
        self.sdk_path_edit.setPlaceholderText("Path to MediaSDK folder (containing bin/, lib/, models/)")
        sdk_path_layout.addWidget(self.sdk_path_edit, stretch=1)
        
        self.sdk_browse_btn = QPushButton("Browse...")
        self.sdk_browse_btn.clicked.connect(self.browse_sdk_path)
        sdk_path_layout.addWidget(self.sdk_browse_btn)
        
        self.sdk_clear_btn = QPushButton("Clear")
        self.sdk_clear_btn.clicked.connect(lambda: self.sdk_path_edit.clear())
        sdk_path_layout.addWidget(self.sdk_clear_btn)
        
        sdk_layout.addLayout(sdk_path_layout)
        
        # SDK status label
        self.sdk_status_label = QLabel()
        self.sdk_status_label.setWordWrap(True)
        sdk_layout.addWidget(self.sdk_status_label)
        
        # Test SDK button
        test_sdk_layout = QHBoxLayout()
        test_sdk_layout.addStretch()
        self.test_sdk_btn = QPushButton("Test SDK")
        self.test_sdk_btn.clicked.connect(self.test_sdk)
        test_sdk_layout.addWidget(self.test_sdk_btn)
        sdk_layout.addLayout(test_sdk_layout)
        
        sdk_group.setLayout(sdk_layout)
        layout.addWidget(sdk_group)
        
        # FFmpeg path configuration
        ffmpeg_group = QGroupBox("FFmpeg Path")
        ffmpeg_layout = QVBoxLayout()
        
        # FFmpeg path input
        ffmpeg_path_layout = QHBoxLayout()
        self.ffmpeg_path_edit = QLineEdit()
        self.ffmpeg_path_edit.setPlaceholderText("Path to ffmpeg.exe")
        ffmpeg_path_layout.addWidget(self.ffmpeg_path_edit, stretch=1)
        
        self.ffmpeg_browse_btn = QPushButton("Browse...")
        self.ffmpeg_browse_btn.clicked.connect(self.browse_ffmpeg_path)
        ffmpeg_path_layout.addWidget(self.ffmpeg_browse_btn)
        
        self.ffmpeg_clear_btn = QPushButton("Clear")
        self.ffmpeg_clear_btn.clicked.connect(lambda: self.ffmpeg_path_edit.clear())
        ffmpeg_path_layout.addWidget(self.ffmpeg_clear_btn)
        
        ffmpeg_layout.addLayout(ffmpeg_path_layout)
        
        # FFmpeg status label
        self.ffmpeg_status_label = QLabel()
        self.ffmpeg_status_label.setWordWrap(True)
        ffmpeg_layout.addWidget(self.ffmpeg_status_label)
        
        # Test FFmpeg button
        test_ffmpeg_layout = QHBoxLayout()
        test_ffmpeg_layout.addStretch()
        self.test_ffmpeg_btn = QPushButton("Test FFmpeg")
        self.test_ffmpeg_btn.clicked.connect(self.test_ffmpeg)
        test_ffmpeg_layout.addWidget(self.test_ffmpeg_btn)
        ffmpeg_layout.addLayout(test_ffmpeg_layout)
        
        ffmpeg_group.setLayout(ffmpeg_layout)
        layout.addWidget(ffmpeg_group)

        # Reconstruction binaries configuration
        recon_group = QGroupBox("Reconstruction Binaries")
        recon_layout = QVBoxLayout()

        # SphereSfM path input
        spheresfm_path_layout = QHBoxLayout()
        self.spheresfm_path_edit = QLineEdit()
        self.spheresfm_path_edit.setPlaceholderText("Path to SphereSfM colmap.exe")
        spheresfm_path_layout.addWidget(self.spheresfm_path_edit, stretch=1)

        self.spheresfm_browse_btn = QPushButton("Browse...")
        self.spheresfm_browse_btn.clicked.connect(self.browse_spheresfm_path)
        spheresfm_path_layout.addWidget(self.spheresfm_browse_btn)

        self.spheresfm_clear_btn = QPushButton("Clear")
        self.spheresfm_clear_btn.clicked.connect(lambda: self.spheresfm_path_edit.clear())
        spheresfm_path_layout.addWidget(self.spheresfm_clear_btn)
        recon_layout.addLayout(spheresfm_path_layout)

        self.spheresfm_status_label = QLabel()
        self.spheresfm_status_label.setWordWrap(True)
        recon_layout.addWidget(self.spheresfm_status_label)

        test_spheresfm_layout = QHBoxLayout()
        test_spheresfm_layout.addStretch()
        self.test_spheresfm_btn = QPushButton("Test SphereSfM")
        self.test_spheresfm_btn.clicked.connect(self.test_spheresfm)
        test_spheresfm_layout.addWidget(self.test_spheresfm_btn)
        recon_layout.addLayout(test_spheresfm_layout)

        # COLMAP GPU path input
        colmap_path_layout = QHBoxLayout()
        self.colmap_path_edit = QLineEdit()
        self.colmap_path_edit.setPlaceholderText("Path to COLMAP GPU colmap.exe / colmap.bat")
        colmap_path_layout.addWidget(self.colmap_path_edit, stretch=1)

        self.colmap_browse_btn = QPushButton("Browse...")
        self.colmap_browse_btn.clicked.connect(self.browse_colmap_path)
        colmap_path_layout.addWidget(self.colmap_browse_btn)

        self.colmap_clear_btn = QPushButton("Clear")
        self.colmap_clear_btn.clicked.connect(lambda: self.colmap_path_edit.clear())
        colmap_path_layout.addWidget(self.colmap_clear_btn)
        recon_layout.addLayout(colmap_path_layout)

        self.colmap_status_label = QLabel()
        self.colmap_status_label.setWordWrap(True)
        recon_layout.addWidget(self.colmap_status_label)

        test_colmap_layout = QHBoxLayout()
        test_colmap_layout.addStretch()
        self.test_colmap_btn = QPushButton("Test COLMAP GPU")
        self.test_colmap_btn.clicked.connect(self.test_colmap)
        test_colmap_layout.addWidget(self.test_colmap_btn)
        recon_layout.addLayout(test_colmap_layout)

        # Global Mapper path input (legacy key; uses COLMAP executable)
        glomap_path_layout = QHBoxLayout()
        self.glomap_path_edit = QLineEdit()
        self.glomap_path_edit.setPlaceholderText("Path to colmap.exe for global_mapper")
        glomap_path_layout.addWidget(self.glomap_path_edit, stretch=1)

        self.glomap_browse_btn = QPushButton("Browse...")
        self.glomap_browse_btn.clicked.connect(self.browse_glomap_path)
        glomap_path_layout.addWidget(self.glomap_browse_btn)

        self.glomap_clear_btn = QPushButton("Clear")
        self.glomap_clear_btn.clicked.connect(lambda: self.glomap_path_edit.clear())
        glomap_path_layout.addWidget(self.glomap_clear_btn)
        recon_layout.addLayout(glomap_path_layout)

        self.glomap_status_label = QLabel()
        self.glomap_status_label.setWordWrap(True)
        recon_layout.addWidget(self.glomap_status_label)

        test_glomap_layout = QHBoxLayout()
        test_glomap_layout.addStretch()
        self.test_glomap_btn = QPushButton("Test Global Mapper")
        self.test_glomap_btn.clicked.connect(self.test_glomap)
        test_glomap_layout.addWidget(self.test_glomap_btn)
        recon_layout.addLayout(test_glomap_layout)

        recon_group.setLayout(recon_layout)
        layout.addWidget(recon_group)
        
        layout.addStretch()
        return widget
    
    def create_directories_tab(self) -> QWidget:
        """Create the directories tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        
        appearance_group = QGroupBox("Appearance")
        appearance_layout = QFormLayout()
        self.theme_combo = QComboBox()
        self.theme_combo.addItem("Dark", "dark")
        self.theme_combo.addItem("Light", "light")
        self.theme_combo.addItem("System", "system")
        appearance_layout.addRow("Theme:", self.theme_combo)
        appearance_group.setLayout(appearance_layout)
        layout.addWidget(appearance_group)
        
        # Recent directories
        dirs_group = QGroupBox("Recent Directories")
        dirs_layout = QFormLayout()
        
        self.last_input_label = QLabel()
        self.last_input_label.setWordWrap(True)
        dirs_layout.addRow("Last Input:", self.last_input_label)
        
        self.last_output_label = QLabel()
        self.last_output_label.setWordWrap(True)
        dirs_layout.addRow("Last Output:", self.last_output_label)
        
        dirs_group.setLayout(dirs_layout)
        layout.addWidget(dirs_group)
        
        # Recent files
        recent_group = QGroupBox("Recent Files")
        recent_layout = QVBoxLayout()
        
        self.recent_files_edit = QTextEdit()
        self.recent_files_edit.setReadOnly(True)
        self.recent_files_edit.setMaximumHeight(200)
        recent_layout.addWidget(self.recent_files_edit)
        
        clear_recent_layout = QHBoxLayout()
        clear_recent_layout.addStretch()
        self.clear_recent_btn = QPushButton("Clear Recent Files")
        self.clear_recent_btn.clicked.connect(self.clear_recent_files)
        clear_recent_layout.addWidget(self.clear_recent_btn)
        recent_layout.addLayout(clear_recent_layout)
        
        recent_group.setLayout(recent_layout)
        layout.addWidget(recent_group)
        
        layout.addStretch()
        return widget
    
    def create_info_tab(self) -> QWidget:
        """Create the system info tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        
        info_group = QGroupBox("System Information")
        info_layout = QVBoxLayout()
        
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setFont(QFont("Consolas", 9))
        info_layout.addWidget(self.info_text)
        
        refresh_layout = QHBoxLayout()
        refresh_layout.addStretch()
        self.refresh_info_btn = QPushButton("Refresh Info")
        self.refresh_info_btn.clicked.connect(self.refresh_system_info)
        refresh_layout.addWidget(self.refresh_info_btn)
        info_layout.addLayout(refresh_layout)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        return widget
    
    def load_current_settings(self):
        """Load current settings into UI"""
        # Auto-detection
        self.auto_detect_checkbox.setChecked(self.settings.get_auto_detect_on_startup())
        
        # SDK path
        sdk_path = self.settings.get_sdk_path()
        if sdk_path:
            self.sdk_path_edit.setText(str(sdk_path))
        self.update_sdk_status()
        
        # FFmpeg path
        ffmpeg_path = self.settings.get_ffmpeg_path()
        if ffmpeg_path:
            self.ffmpeg_path_edit.setText(str(ffmpeg_path))
        self.update_ffmpeg_status()

        # SphereSfM path
        spheresfm_path = self.settings.get_spheresfm_path()
        if spheresfm_path:
            self.spheresfm_path_edit.setText(str(spheresfm_path))
        self.update_spheresfm_status()

        # COLMAP GPU path
        colmap_path = self.settings.get_colmap_gpu_path()
        if colmap_path:
            self.colmap_path_edit.setText(str(colmap_path))
        self.update_colmap_status()

        # Global mapper path mirrors COLMAP path
        glomap_path = self.settings.get_colmap_gpu_path()
        if glomap_path:
            self.glomap_path_edit.setText(str(glomap_path))
        self.update_glomap_status()
        
        theme = self.settings.get_theme()
        theme_index = self.theme_combo.findData(theme)
        if theme_index >= 0:
            self.theme_combo.setCurrentIndex(theme_index)
        
        # Directories
        last_input = self.settings.get_last_input_directory()
        self.last_input_label.setText(str(last_input) if last_input else "Not set")
        
        last_output = self.settings.get_last_output_directory()
        self.last_output_label.setText(str(last_output) if last_output else "Not set")
        
        # Recent files
        recent_files = self.settings.get_recent_files()
        if recent_files:
            self.recent_files_edit.setText('\n'.join(str(f) for f in recent_files))
        else:
            self.recent_files_edit.setText("No recent files")
        
        # System info
        self.refresh_system_info()
    
    def update_sdk_status(self):
        """Update SDK status label"""
        sdk_path_str = self.sdk_path_edit.text().strip()
        if not sdk_path_str:
            self.sdk_status_label.setText("⚠️ No SDK path configured")
            self.sdk_status_label.setStyleSheet("color: orange;")
            return
        
        sdk_path = Path(sdk_path_str)
        if self.settings.is_sdk_valid(sdk_path):
            info = self.settings.get_sdk_info(sdk_path)
            status = f"✅ Valid SDK found - Version {info.get('version', 'Unknown')}"
            if info.get('models'):
                status += f"\nModels: {', '.join(info['models'][:3])}"
                if len(info['models']) > 3:
                    status += f" (+{len(info['models'])-3} more)"
            self.sdk_status_label.setText(status)
            self.sdk_status_label.setStyleSheet("color: green;")
        else:
            self.sdk_status_label.setText("❌ Invalid SDK path - missing required files")
            self.sdk_status_label.setStyleSheet("color: red;")
    
    def update_ffmpeg_status(self):
        """Update FFmpeg status label"""
        ffmpeg_path_str = self.ffmpeg_path_edit.text().strip()
        if not ffmpeg_path_str:
            self.ffmpeg_status_label.setText("⚠️ No FFmpeg path configured")
            self.ffmpeg_status_label.setStyleSheet("color: orange;")
            return
        
        ffmpeg_path = Path(ffmpeg_path_str)
        if self.settings.is_ffmpeg_valid(ffmpeg_path):
            info = self.settings.get_ffmpeg_info(ffmpeg_path)
            status = f"✅ Valid FFmpeg found - Version {info.get('version', 'Unknown')}"
            features = []
            if info.get('has_v360_filter'):
                features.append("v360 filter")
            if info.get('has_cuda'):
                features.append("CUDA")
            if features:
                status += f"\nFeatures: {', '.join(features)}"
            self.ffmpeg_status_label.setText(status)
            self.ffmpeg_status_label.setStyleSheet("color: green;")
        else:
            self.ffmpeg_status_label.setText("❌ Invalid FFmpeg path")
            self.ffmpeg_status_label.setStyleSheet("color: red;")

    def update_colmap_status(self):
        """Update COLMAP GPU status label"""
        colmap_path_str = self.colmap_path_edit.text().strip()
        if not colmap_path_str:
            self.colmap_status_label.setText("⚠️ No COLMAP GPU path configured")
            self.colmap_status_label.setStyleSheet("color: orange;")
            return

        colmap_path = Path(colmap_path_str)
        if self.settings.is_colmap_valid(colmap_path):
            info = self.settings.get_colmap_info(colmap_path)
            status = f"✅ Valid COLMAP GPU found - {info.get('version', 'Version unknown')}"
            self.colmap_status_label.setText(status)
            self.colmap_status_label.setStyleSheet("color: green;")
        else:
            self.colmap_status_label.setText("❌ Invalid COLMAP GPU path")
            self.colmap_status_label.setStyleSheet("color: red;")

    def update_spheresfm_status(self):
        """Update SphereSfM status label"""
        spheresfm_path_str = self.spheresfm_path_edit.text().strip()
        if not spheresfm_path_str:
            self.spheresfm_status_label.setText("⚠️ No SphereSfM path configured")
            self.spheresfm_status_label.setStyleSheet("color: orange;")
            return

        spheresfm_path = Path(spheresfm_path_str)
        if self.settings.is_colmap_valid(spheresfm_path):
            info = self.settings.get_colmap_info(spheresfm_path)
            status = f"✅ Valid SphereSfM found - {info.get('version', 'Version unknown')}"
            self.spheresfm_status_label.setText(status)
            self.spheresfm_status_label.setStyleSheet("color: green;")
        else:
            self.spheresfm_status_label.setText("❌ Invalid SphereSfM path")
            self.spheresfm_status_label.setStyleSheet("color: red;")

    def browse_spheresfm_path(self):
        """Open file browser for SphereSfM executable"""
        current = self.spheresfm_path_edit.text().strip()
        start_dir = str(Path(current).parent) if current and Path(current).exists() else ""

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select SphereSfM Executable",
            start_dir,
            "Executables (*.exe *.bat *.cmd);;All Files (*.*)"
        )

        if file_path:
            self.spheresfm_path_edit.setText(file_path)
            self.update_spheresfm_status()

    def update_glomap_status(self):
        """Update global mapper status label (resolved via COLMAP)."""
        glomap_path_str = self.glomap_path_edit.text().strip()
        if not glomap_path_str:
            self.glomap_status_label.setText("⚠️ No global mapper path configured")
            self.glomap_status_label.setStyleSheet("color: orange;")
            return

        glomap_path = Path(glomap_path_str)
        if self.settings.is_colmap_valid(glomap_path):
            info = self.settings.get_colmap_info(glomap_path)
            cuda_text = "CUDA" if info.get('cuda') else "CPU"
            status = f"✅ Global mapper ready - {info.get('version', 'Version unknown')} [{cuda_text}]"
            self.glomap_status_label.setText(status)
            self.glomap_status_label.setStyleSheet("color: green;")
        else:
            self.glomap_status_label.setText("❌ Invalid global mapper path")
            self.glomap_status_label.setStyleSheet("color: red;")
    
    def browse_sdk_path(self):
        """Open folder browser for SDK path"""
        current = self.sdk_path_edit.text().strip()
        start_dir = current if current and Path(current).exists() else str(Path.home() / "Documents")
        
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select MediaSDK Folder",
            start_dir,
            QFileDialog.Option.ShowDirsOnly
        )
        
        if folder:
            self.sdk_path_edit.setText(folder)
            self.update_sdk_status()
    
    def browse_ffmpeg_path(self):
        """Open file browser for FFmpeg executable"""
        current = self.ffmpeg_path_edit.text().strip()
        start_dir = str(Path(current).parent) if current and Path(current).exists() else ""
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select FFmpeg Executable",
            start_dir,
            "Executables (*.exe);;All Files (*.*)"
        )
        
        if file_path:
            self.ffmpeg_path_edit.setText(file_path)
            self.update_ffmpeg_status()

    def browse_colmap_path(self):
        """Open file browser for COLMAP executable"""
        current = self.colmap_path_edit.text().strip()
        start_dir = str(Path(current).parent) if current and Path(current).exists() else ""

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select COLMAP Executable",
            start_dir,
            "Executables (*.exe *.bat *.cmd);;All Files (*.*)"
        )

        if file_path:
            self.colmap_path_edit.setText(file_path)
            self.update_colmap_status()

    def browse_glomap_path(self):
        """Open file browser for COLMAP executable used by global_mapper."""
        current = self.glomap_path_edit.text().strip()
        start_dir = str(Path(current).parent) if current and Path(current).exists() else ""

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select COLMAP Executable (global_mapper)",
            start_dir,
            "Executables (*.exe *.bat *.cmd);;All Files (*.*)"
        )

        if file_path:
            self.glomap_path_edit.setText(file_path)
            self.update_glomap_status()
    
    def detect_paths_now(self):
        """Run auto-detection for important dependencies."""
        self.detect_now_btn.setEnabled(False)
        self.detect_now_btn.setText("Detecting...")
        
        # Detect SDK
        sdk_path = self.settings.auto_detect_sdk()
        if sdk_path:
            self.sdk_path_edit.setText(str(sdk_path))
            self.update_sdk_status()
        
        # Detect FFmpeg
        ffmpeg_path = self.settings.auto_detect_ffmpeg()
        if ffmpeg_path:
            self.ffmpeg_path_edit.setText(str(ffmpeg_path))
            self.update_ffmpeg_status()

        # Detect SphereSfM
        spheresfm_path = self.settings.auto_detect_spheresfm()
        if spheresfm_path:
            self.spheresfm_path_edit.setText(str(spheresfm_path))
            self.update_spheresfm_status()

        # Detect COLMAP GPU
        colmap_path = self.settings.auto_detect_colmap()
        if colmap_path:
            self.colmap_path_edit.setText(str(colmap_path))
            self.update_colmap_status()

        # Global mapper uses the same COLMAP executable
        glomap_path = colmap_path
        if glomap_path:
            self.glomap_path_edit.setText(str(glomap_path))
            self.update_glomap_status()
        
        self.detect_now_btn.setEnabled(True)
        self.detect_now_btn.setText("Detect Now")
        
        if sdk_path or ffmpeg_path or spheresfm_path or colmap_path or glomap_path:
            QMessageBox.information(
                self,
                "Detection Complete",
                f"Found:\n" +
                (f"✓ SDK: {sdk_path}\n" if sdk_path else "✗ SDK not found\n") +
                (f"✓ FFmpeg: {ffmpeg_path}\n" if ffmpeg_path else "✗ FFmpeg not found\n") +
                (f"✓ SphereSfM: {spheresfm_path}\n" if spheresfm_path else "✗ SphereSfM not found\n") +
                (f"✓ COLMAP GPU: {colmap_path}\n" if colmap_path else "✗ COLMAP GPU not found\n") +
                (f"✓ Global Mapper (via COLMAP): {glomap_path}" if glomap_path else "✗ Global Mapper path not found")
            )
        else:
            QMessageBox.warning(
                self,
                "Detection Failed",
                "Could not auto-detect SDK/FFmpeg/SphereSfM/COLMAP GPU/global mapper path.\nPlease browse manually."
            )

    def test_spheresfm(self):
        """Test SphereSfM configuration"""
        spheresfm_path_str = self.spheresfm_path_edit.text().strip()
        if not spheresfm_path_str:
            QMessageBox.warning(self, "No Path", "Please enter SphereSfM path first")
            return

        spheresfm_path = Path(spheresfm_path_str)
        info = self.settings.get_colmap_info(spheresfm_path)

        if info['valid']:
            msg = f"✅ SphereSfM is valid and ready to use!\n\n"
            msg += f"Version: {info.get('version', 'Unknown')}\n"
            msg += f"Path: {info.get('path', 'N/A')}\n"
            QMessageBox.information(self, "SphereSfM Test - Success", msg)
        else:
            QMessageBox.critical(
                self,
                "SphereSfM Test - Failed",
                f"❌ SphereSfM validation failed:\n{info.get('error', 'Unknown error')}"
            )
    
    def test_sdk(self):
        """Test SDK configuration"""
        sdk_path_str = self.sdk_path_edit.text().strip()
        if not sdk_path_str:
            QMessageBox.warning(self, "No Path", "Please enter SDK path first")
            return
        
        sdk_path = Path(sdk_path_str)
        info = self.settings.get_sdk_info(sdk_path)
        
        if info['valid']:
            msg = f"✅ SDK is valid and ready to use!\n\n"
            msg += f"Version: {info.get('version', 'Unknown')}\n"
            msg += f"Executable: {info['executable']}\n"
            msg += f"Models: {len(info.get('models', []))}\n\n"
            if info.get('models'):
                msg += "Available models:\n" + '\n'.join(f"  • {m}" for m in info['models'])
            QMessageBox.information(self, "SDK Test - Success", msg)
        else:
            QMessageBox.critical(
                self,
                "SDK Test - Failed",
                f"❌ SDK validation failed:\n{info.get('error', 'Unknown error')}"
            )
    
    def test_ffmpeg(self):
        """Test FFmpeg configuration"""
        ffmpeg_path_str = self.ffmpeg_path_edit.text().strip()
        if not ffmpeg_path_str:
            QMessageBox.warning(self, "No Path", "Please enter FFmpeg path first")
            return
        
        ffmpeg_path = Path(ffmpeg_path_str)
        info = self.settings.get_ffmpeg_info(ffmpeg_path)
        
        if info['valid']:
            msg = f"✅ FFmpeg is valid and ready to use!\n\n"
            msg += f"Version: {info.get('version', 'Unknown')}\n"
            msg += f"Path: {info['path']}\n\n"
            msg += "Features:\n"
            msg += f"  • v360 filter: {'✓' if info.get('has_v360_filter') else '✗'}\n"
            msg += f"  • CUDA support: {'✓' if info.get('has_cuda') else '✗'}\n"
            QMessageBox.information(self, "FFmpeg Test - Success", msg)
        else:
            QMessageBox.critical(
                self,
                "FFmpeg Test - Failed",
                f"❌ FFmpeg validation failed:\n{info.get('error', 'Unknown error')}"
            )

    def test_colmap(self):
        """Test COLMAP GPU configuration"""
        colmap_path_str = self.colmap_path_edit.text().strip()
        if not colmap_path_str:
            QMessageBox.warning(self, "No Path", "Please enter COLMAP GPU path first")
            return

        colmap_path = Path(colmap_path_str)
        info = self.settings.get_colmap_info(colmap_path)

        if info['valid']:
            msg = f"✅ COLMAP GPU is valid and ready to use!\n\n"
            msg += f"Version: {info.get('version', 'Unknown')}\n"
            msg += f"Path: {info.get('path', 'N/A')}\n"
            QMessageBox.information(self, "COLMAP GPU Test - Success", msg)
        else:
            QMessageBox.critical(
                self,
                "COLMAP GPU Test - Failed",
                f"❌ COLMAP GPU validation failed:\n{info.get('error', 'Unknown error')}"
            )

    def test_glomap(self):
        """Test global mapper configuration (via COLMAP)."""
        glomap_path_str = self.glomap_path_edit.text().strip()
        if not glomap_path_str:
            QMessageBox.warning(self, "No Path", "Please enter COLMAP/global mapper path first")
            return

        glomap_path = Path(glomap_path_str)
        info = self.settings.get_colmap_info(glomap_path)

        if info['valid']:
            msg = f"✅ Global mapper is valid and ready to use!\n\n"
            msg += f"Version: {info.get('version', 'Unknown')}\n"
            msg += f"Path: {info.get('path', 'N/A')}\n"
            msg += f"CUDA support: {'✓' if info.get('cuda') else '✗'}\n"
            QMessageBox.information(self, "Global Mapper Test - Success", msg)
        else:
            QMessageBox.critical(
                self,
                "Global Mapper Test - Failed",
                f"❌ Global mapper validation failed:\n{info.get('error', 'Unknown error')}"
            )
    
    def refresh_system_info(self):
        """Refresh system information display"""
        info_lines = []
        
        # SDK info
        info_lines.append("=== INSTA360 MEDIASDK ===")
        sdk_path = self.settings.get_sdk_path()
        if sdk_path:
            sdk_info = self.settings.get_sdk_info(sdk_path)
            info_lines.append(f"Status: {'✓ Valid' if sdk_info['valid'] else '✗ Invalid'}")
            info_lines.append(f"Version: {sdk_info.get('version', 'Unknown')}")
            info_lines.append(f"Path: {sdk_info.get('path', 'N/A')}")
            info_lines.append(f"Executable: {sdk_info.get('executable', 'N/A')}")
            info_lines.append(f"Models: {len(sdk_info.get('models', []))}")
            info_lines.append(f"Auto-detected: {'Yes' if self.settings.settings.get('sdk_auto_detected') else 'No'}")
        else:
            info_lines.append("Status: Not configured")

        info_lines.append("")

        # SphereSfM info
        info_lines.append("=== SPHERESFM ===")
        spheresfm_path = self.settings.get_spheresfm_path()
        if spheresfm_path:
            spheresfm_info = self.settings.get_colmap_info(spheresfm_path)
            info_lines.append(f"Status: {'✓ Valid' if spheresfm_info['valid'] else '✗ Invalid'}")
            info_lines.append(f"Version: {spheresfm_info.get('version', 'Unknown')}")
            info_lines.append(f"Path: {spheresfm_info.get('path', 'N/A')}")
            info_lines.append(f"Auto-detected: {'Yes' if self.settings.settings.get('spheresfm_auto_detected') else 'No'}")
        else:
            info_lines.append("Status: Not configured")

        info_lines.append("")

        # COLMAP GPU info
        info_lines.append("=== COLMAP GPU ===")
        colmap_path = self.settings.get_colmap_gpu_path()
        if colmap_path:
            colmap_info = self.settings.get_colmap_info(colmap_path)
            info_lines.append(f"Status: {'✓ Valid' if colmap_info['valid'] else '✗ Invalid'}")
            info_lines.append(f"Version: {colmap_info.get('version', 'Unknown')}")
            info_lines.append(f"Path: {colmap_info.get('path', 'N/A')}")
            info_lines.append(f"Auto-detected: {'Yes' if self.settings.settings.get('colmap_auto_detected') else 'No'}")
        else:
            info_lines.append("Status: Not configured")

        info_lines.append("")

        # Global mapper info (legacy GloMAP key)
        info_lines.append("=== GLOBAL MAPPER (COLMAP) ===")
        glomap_path = self.settings.get_colmap_gpu_path()
        if glomap_path:
            glomap_info = self.settings.get_colmap_info(glomap_path)
            info_lines.append(f"Status: {'✓ Valid' if glomap_info['valid'] else '✗ Invalid'}")
            info_lines.append(f"Version: {glomap_info.get('version', 'Unknown')}")
            info_lines.append(f"Path: {glomap_info.get('path', 'N/A')}")
            info_lines.append(f"CUDA support: {'✓ Available' if glomap_info.get('cuda') else '✗ Not available'}")
            info_lines.append(f"Auto-detected: {'Yes' if self.settings.settings.get('colmap_auto_detected') else 'No'}")
        else:
            info_lines.append("Status: Not configured")
        
        info_lines.append("")
        
        # FFmpeg info
        info_lines.append("=== FFMPEG ===")
        ffmpeg_path = self.settings.get_ffmpeg_path()
        if ffmpeg_path:
            ffmpeg_info = self.settings.get_ffmpeg_info(ffmpeg_path)
            info_lines.append(f"Status: {'✓ Valid' if ffmpeg_info['valid'] else '✗ Invalid'}")
            info_lines.append(f"Version: {ffmpeg_info.get('version', 'Unknown')}")
            info_lines.append(f"Path: {ffmpeg_info.get('path', 'N/A')}")
            info_lines.append(f"v360 filter: {'✓ Available' if ffmpeg_info.get('has_v360_filter') else '✗ Not available'}")
            info_lines.append(f"CUDA support: {'✓ Available' if ffmpeg_info.get('has_cuda') else '✗ Not available'}")
            info_lines.append(f"Auto-detected: {'Yes' if self.settings.settings.get('ffmpeg_auto_detected') else 'No'}")
        else:
            info_lines.append("Status: Not configured")
        
        info_lines.append("")
        info_lines.append("=== APPLICATION ===")
        info_lines.append(f"Settings file: {self.settings.settings_file}")
        info_lines.append(f"Auto-detect on startup: {'Enabled' if self.settings.get_auto_detect_on_startup() else 'Disabled'}")
        
        self.info_text.setText('\n'.join(info_lines))
    
    def clear_recent_files(self):
        """Clear recent files list"""
        reply = QMessageBox.question(
            self,
            "Clear Recent Files",
            "Are you sure you want to clear the recent files list?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.settings.clear_recent_files()
            self.recent_files_edit.setText("No recent files")
    
    def reset_settings(self):
        """Reset all settings to defaults"""
        reply = QMessageBox.question(
            self,
            "Reset Settings",
            "Are you sure you want to reset all settings to defaults?\nThis will clear all configured paths.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.settings.reset_to_defaults()
            self.load_current_settings()
            QMessageBox.information(self, "Reset Complete", "Settings have been reset to defaults")
    
    def save_settings(self):
        """Save all settings"""
        # Save auto-detection preference
        self.settings.set_auto_detect_on_startup(self.auto_detect_checkbox.isChecked())
        
        self.settings.set_theme(self.theme_combo.currentData())
        
        # Save SDK path
        sdk_path_str = self.sdk_path_edit.text().strip()
        if sdk_path_str:
            sdk_path = Path(sdk_path_str)
            if not self.settings.set_sdk_path(sdk_path, auto_detected=False):
                QMessageBox.warning(
                    self,
                    "Invalid SDK Path",
                    "The SDK path is invalid. Please check the path and try again."
                )
                return
        else:
            self.settings.set_sdk_path(None)
        
        # Save FFmpeg path
        ffmpeg_path_str = self.ffmpeg_path_edit.text().strip()
        if ffmpeg_path_str:
            ffmpeg_path = Path(ffmpeg_path_str)
            if not self.settings.set_ffmpeg_path(ffmpeg_path, auto_detected=False):
                QMessageBox.warning(
                    self,
                    "Invalid FFmpeg Path",
                    "The FFmpeg path is invalid. Please check the path and try again."
                )
                return
        else:
            self.settings.set_ffmpeg_path(None)

        # Save SphereSfM path
        spheresfm_path_str = self.spheresfm_path_edit.text().strip()
        if spheresfm_path_str:
            spheresfm_path = Path(spheresfm_path_str)
            if not self.settings.set_spheresfm_path(spheresfm_path, auto_detected=False):
                QMessageBox.warning(
                    self,
                    "Invalid SphereSfM Path",
                    "The SphereSfM path is invalid. Please check the path and try again."
                )
                return
        else:
            self.settings.set_spheresfm_path(None)

        # Save COLMAP GPU path
        colmap_path_str = self.colmap_path_edit.text().strip()
        if colmap_path_str:
            colmap_path = Path(colmap_path_str)
            if not self.settings.set_colmap_gpu_path(colmap_path, auto_detected=False):
                QMessageBox.warning(
                    self,
                    "Invalid COLMAP GPU Path",
                    "The COLMAP GPU path is invalid. Please check the path and try again."
                )
                return
        else:
            self.settings.set_colmap_gpu_path(None)

        # Global mapper path is no longer persisted separately; it follows COLMAP path.
        
        # Emit signal and close
        self.settings_changed.emit()
        QMessageBox.information(self, "Settings Saved", "Settings have been saved successfully")
        self.accept()
