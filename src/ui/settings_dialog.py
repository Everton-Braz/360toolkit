"""
360FrameTools - Settings Dialog
UI for configuring SDK and FFmpeg paths, and other preferences.
"""

import logging
from pathlib import Path

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QLineEdit, QFileDialog, QCheckBox,
    QMessageBox, QTextEdit, QTabWidget, QWidget, QFormLayout
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
        
        layout.addStretch()
        return widget
    
    def create_directories_tab(self) -> QWidget:
        """Create the directories tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        
        # Performance options
        perf_group = QGroupBox("Performance Options")
        perf_layout = QVBoxLayout()
        
        self.skip_intermediate_checkbox = QCheckBox("Skip saving Stage 1 frames (faster)")
        self.skip_intermediate_checkbox.setToolTip(
            "When enabled, equirectangular frames are saved to a temp folder\n"
            "and automatically deleted after Stage 2 completes.\n"
            "This saves disk space and can be faster on SSDs.\n\n"
            "Disable this if you want to keep the equirectangular frames."
        )
        perf_layout.addWidget(self.skip_intermediate_checkbox)
        
        perf_group.setLayout(perf_layout)
        layout.addWidget(perf_group)
        
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
        
        # Performance options
        self.skip_intermediate_checkbox.setChecked(self.settings.get_skip_intermediate_save())
        
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
    
    def detect_paths_now(self):
        """Run auto-detection for both SDK and FFmpeg"""
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
        
        self.detect_now_btn.setEnabled(True)
        self.detect_now_btn.setText("Detect Now")
        
        if sdk_path or ffmpeg_path:
            QMessageBox.information(
                self,
                "Detection Complete",
                f"Found:\n" +
                (f"✓ SDK: {sdk_path}\n" if sdk_path else "✗ SDK not found\n") +
                (f"✓ FFmpeg: {ffmpeg_path}" if ffmpeg_path else "✗ FFmpeg not found")
            )
        else:
            QMessageBox.warning(
                self,
                "Detection Failed",
                "Could not auto-detect SDK or FFmpeg.\nPlease browse manually."
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
        
        # Save performance options
        self.settings.set_skip_intermediate_save(self.skip_intermediate_checkbox.isChecked())
        
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
        
        # Emit signal and close
        self.settings_changed.emit()
        QMessageBox.information(self, "Settings Saved", "Settings have been saved successfully")
        self.accept()
