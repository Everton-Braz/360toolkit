"""
360FrameTools - Main Window
PyQt6-based minimalist interface for the 3-stage pipeline.
"""

import sys
import logging
from pathlib import Path

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QProgressBar, QTabWidget,
    QFileDialog, QMessageBox, QTextEdit, QGroupBox,
    QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
    QLineEdit, QSplitter, QScrollArea, QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QAction

from src.pipeline.batch_orchestrator import BatchOrchestrator
from src.config.defaults import (
    APP_NAME, APP_VERSION,
    DEFAULT_FPS, DEFAULT_H_FOV, DEFAULT_SPLIT_COUNT,
    EXTRACTION_METHODS, TRANSFORM_TYPES, YOLOV8_MODELS,
    DEFAULT_SDK_QUALITY
)
from src.config.settings import get_settings
from src.config.config_manager import get_config_manager
from src.ui.settings_dialog import SettingsDialog
from src.ui.config_dialog import ConfigManagementDialog, SaveConfigDialog

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Main application window for 360toolkit"""
    
    def __init__(self):
        super().__init__()
        
        self.settings = get_settings()
        self.config_manager = get_config_manager()
        self.orchestrator = BatchOrchestrator()
        self.pipeline_config = {}
        
        self.init_ui()
        self.create_menu_bar()
        self.apply_dark_theme()
        
        # Trigger initial visibility of SDK controls
        self.on_extraction_method_changed(0)
    
    def init_ui(self):
        """Initialize the user interface - HYBRID APPROACH with responsive design"""
        
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        
        # Detect screen size and set appropriate defaults
        screen = self.screen()
        screen_geometry = screen.geometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()
        
        logger.info(f"Screen detected: {screen_width}×{screen_height}")
        
        # Responsive sizing based on screen resolution
        if screen_width >= 3840:  # 4K or higher
            self.setMinimumSize(1600, 1000)
            self.resize(2400, 1400)
            self.base_font_size = 12
        elif screen_width >= 2560:  # 2K
            self.setMinimumSize(1400, 900)
            self.resize(2000, 1200)
            self.base_font_size = 11
        elif screen_width >= 1920:  # Full HD
            self.setMinimumSize(1200, 800)
            self.resize(1600, 1000)
            self.base_font_size = 10
        else:  # HD or lower
            self.setMinimumSize(1024, 768)
            self.resize(1280, 900)
            self.base_font_size = 9
        
        logger.info(f"Window size: {self.width()}×{self.height()}, Font size: {self.base_font_size}pt")
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)
        
        # TOP SECTION: Input/Output + Pipeline Overview (fixed at top)
        top_section = self.create_top_section()
        main_layout.addWidget(top_section, stretch=0)  # Don't stretch - fixed size
        
        # MIDDLE + BOTTOM: Use QSplitter for resizable tabs/log sections
        splitter = self.create_splitter_section()
        main_layout.addWidget(splitter, stretch=1)  # Allow this to expand
        
        # Status bar
        self.statusBar().showMessage("Ready to start")
    
    def create_menu_bar(self):
        """Create menu bar with File and Settings menus"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        open_action = QAction("&Open File...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.browse_input_file)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Config menu
        config_menu = menubar.addMenu("&Configuration")
        
        save_config_action = QAction("💾 &Save Configuration...", self)
        save_config_action.setShortcut("Ctrl+S")
        save_config_action.triggered.connect(self.save_configuration)
        config_menu.addAction(save_config_action)
        
        load_config_action = QAction("📂 &Load Configuration...", self)
        load_config_action.setShortcut("Ctrl+L")
        load_config_action.triggered.connect(self.load_configuration)
        config_menu.addAction(load_config_action)
        
        config_menu.addSeparator()
        
        manage_config_action = QAction("🗂️ &Manage Configurations...", self)
        manage_config_action.triggered.connect(self.manage_configurations)
        config_menu.addAction(manage_config_action)
        
        # Settings menu
        settings_menu = menubar.addMenu("&Settings")
        
        preferences_action = QAction("&Preferences...", self)
        preferences_action.setShortcut("Ctrl+P")
        preferences_action.triggered.connect(self.open_settings)
        settings_menu.addAction(preferences_action)
        
        settings_menu.addSeparator()
        
        detect_paths_action = QAction("&Detect SDK/FFmpeg Paths", self)
        detect_paths_action.triggered.connect(self.detect_paths)
        settings_menu.addAction(detect_paths_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def open_settings(self):
        """Open settings dialog"""
        dialog = SettingsDialog(self)
        dialog.settings_changed.connect(self.on_settings_changed)
        dialog.exec()
    
    def detect_paths(self):
        """Run automatic path detection"""
        sdk = self.settings.auto_detect_sdk()
        ffmpeg = self.settings.auto_detect_ffmpeg()
        
        msg = "Path Detection Results:\n\n"
        
        if sdk:
            self.settings.set_sdk_path(sdk, auto_detected=True)
            msg += f"[OK] SDK Found: {sdk}\n"
        else:
            msg += "[X] SDK not found\n"
        
        if ffmpeg:
            self.settings.set_ffmpeg_path(ffmpeg, auto_detected=True)
            msg += f"[OK] FFmpeg Found: {ffmpeg}\n"
        else:
            msg += "[X] FFmpeg not found\n"
        
        QMessageBox.information(self, "Path Detection", msg)
        self.on_settings_changed()
    
    def on_settings_changed(self):
        """Handle settings changes - update UI and log"""
        sdk_path = self.settings.get_sdk_path()
        ffmpeg_path = self.settings.get_ffmpeg_path()
        
        status_parts = []
        if sdk_path:
            status_parts.append(f"SDK: {sdk_path.name}")
        if ffmpeg_path:
            status_parts.append(f"FFmpeg: {ffmpeg_path.name}")
        
        if status_parts:
            self.statusBar().showMessage(" | ".join(status_parts))
        else:
            self.statusBar().showMessage("[!] SDK/FFmpeg not configured - check Settings")
        
        logger.info(f"Settings updated - SDK: {sdk_path}, FFmpeg: {ffmpeg_path}")
    
    def show_about(self):
        """Show about dialog"""
        about_text = f"""{APP_NAME} v{APP_VERSION}

Unified photogrammetry preprocessing pipeline.
Extract -> Split -> Mask in one streamlined workflow.

Copyright (c) 2026
License: MIT"""
        QMessageBox.about(self, f"About {APP_NAME}", about_text)
    
    def save_configuration(self):
        """Save current pipeline configuration"""
        config = self.get_current_config()
        
        dialog = SaveConfigDialog(config, self)
        dialog.exec()
    
    def load_configuration(self):
        """Load a saved configuration"""
        dialog = ConfigManagementDialog(self)
        dialog.config_loaded.connect(self.apply_loaded_config)
        dialog.exec()
    
    def manage_configurations(self):
        """Open configuration management dialog"""
        dialog = ConfigManagementDialog(self)
        dialog.config_loaded.connect(self.apply_loaded_config)
        dialog.exec()
    
    def get_current_config(self) -> dict:
        """Get current UI configuration as dictionary"""
        return {
            # I/O
            'input_file': self.input_file_edit.text(),
            'output_dir': self.output_dir_edit.text(),
            
            # Stage 1
            'stage1_enabled': self.stage1_enable.isChecked(),
            'fps_interval': self.fps_spin.value(),
            'extraction_method': list(EXTRACTION_METHODS.keys())[self.extraction_method_combo.currentIndex()],
            'sdk_quality': self.sdk_quality_combo.currentText().split(' (')[0].lower(),
            'output_format': 'png',
            
            # Stage 2
            'stage2_enabled': self.stage2_enable.isChecked(),
            'transform_type': self.stage2_method_combo.currentData(),
            'split_count': self.split_count_spin.value(),
            'h_fov': self.fov_spin.value(),  # FIXED: h_fov_spin -> fov_spin
            'output_width': self.stage2_width_spin.value(),  # FIXED: output_width_spin -> stage2_width_spin
            'output_height': self.stage2_height_spin.value(),  # FIXED: output_height_spin -> stage2_height_spin
            'cubemap_face_size': getattr(self, 'cubemap_face_spin', None).value() if hasattr(self, 'cubemap_face_spin') else 1920,
            'cubemap_overlap': getattr(self, 'cubemap_overlap_spin', None).value() if hasattr(self, 'cubemap_overlap_spin') else 10,
            'cubemap_fov': getattr(self, 'cubemap_fov_spin', None).value() if hasattr(self, 'cubemap_fov_spin') else 110,
            'skip_transform': self.skip_transform_check.isChecked(),
            
            # Stage 3
            'stage3_enabled': self.stage3_enable.isChecked(),
            'model_size': list(YOLOV8_MODELS.keys())[self.model_size_combo.currentIndex()],
            'confidence_threshold': self.confidence_spin.value(),
            'use_gpu': self.use_gpu_check.isChecked(),
            'masking_categories': {
                'persons': self.persons_group.isChecked(),
                'personal_objects': self.objects_group.isChecked(),
                'animals': self.animals_group.isChecked()
            },
            
            # Stage 4
            'stage4_enabled': getattr(self, 'stage4_enable', QCheckBox()).isChecked(),
            'alignment_tool': 'glomap',
            'use_gpu_colmap': True,
            
            # Stage 5
            'stage5_enabled': getattr(self, 'stage5_enable', QCheckBox()).isChecked(),
            'export_lithcfeld': True,
            'export_realityscan': True,
            'export_colmap': False,
        }
    
    def apply_loaded_config(self, config: dict):
        """Apply a loaded configuration to the UI"""
        try:
            # I/O
            if 'input_file' in config:
                self.input_file_edit.setText(config['input_file'])
            if 'output_dir' in config:
                self.output_dir_edit.setText(config['output_dir'])
            
            # Stage 1
            if 'stage1_enabled' in config:
                self.stage1_enable.setChecked(config['stage1_enabled'])
            if 'fps_interval' in config:
                self.fps_spin.setValue(config['fps_interval'])
            if 'extraction_method' in config:
                methods = list(EXTRACTION_METHODS.keys())
                if config['extraction_method'] in methods:
                    self.extraction_method_combo.setCurrentIndex(methods.index(config['extraction_method']))
            
            # Stage 2
            if 'stage2_enabled' in config:
                self.stage2_enable.setChecked(config['stage2_enabled'])
            if 'split_count' in config:
                self.split_count_spin.setValue(config['split_count'])
            if 'h_fov' in config:
                self.fov_spin.setValue(config['h_fov'])  # FIXED: h_fov_spin -> fov_spin
            if 'output_width' in config:
                self.stage2_width_spin.setValue(config['output_width'])  # FIXED: output_width_spin -> stage2_width_spin
            if 'output_height' in config:
                self.stage2_height_spin.setValue(config['output_height'])  # FIXED: output_height_spin -> stage2_height_spin
            if 'skip_transform' in config:
                self.skip_transform_check.setChecked(config['skip_transform'])
            
            # Stage 3
            if 'stage3_enabled' in config:
                self.stage3_enable.setChecked(config['stage3_enabled'])
            if 'model_size' in config:
                models = list(YOLOV8_MODELS.keys())
                if config['model_size'] in models:
                    self.model_size_combo.setCurrentIndex(models.index(config['model_size']))
            if 'confidence_threshold' in config:
                self.confidence_spin.setValue(config['confidence_threshold'])
            if 'use_gpu' in config:
                self.use_gpu_check.setChecked(config['use_gpu'])
            if 'masking_categories' in config:
                cats = config['masking_categories']
                self.persons_group.setChecked(cats.get('persons', True))
                self.objects_group.setChecked(cats.get('personal_objects', True))
                self.animals_group.setChecked(cats.get('animals', True))
            
            logger.info("Configuration applied successfully")
            QMessageBox.information(
                self,
                "Configuration Loaded",
                "Configuration has been applied to all settings."
            )
            
        except Exception as e:
            logger.error(f"Failed to apply configuration: {e}")
            QMessageBox.critical(
                self,
                "Load Failed",
                f"Failed to apply configuration: {str(e)}"
            )
    
    def create_splitter_section(self) -> QSplitter:
        """Create splitter with tabs (top) and log panel (bottom) - resizable"""
        
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # MIDDLE SECTION: Configuration Tabs (expandable)
        self.tab_widget = QTabWidget()
        
        # Wrap each tab in a scroll area for responsiveness
        stage1_scroll = self._wrap_in_scroll_area(self.create_stage1_config_tab())
        stage2_persp_scroll = self._wrap_in_scroll_area(self.create_stage2_perspective_tab())
        stage2_cube_scroll = self._wrap_in_scroll_area(self.create_stage2_cubemap_tab())
        stage3_scroll = self._wrap_in_scroll_area(self.create_stage3_config_tab())
        
        self.tab_widget.addTab(stage1_scroll, "Stage 1: Extraction Settings")
        self.tab_widget.addTab(stage2_persp_scroll, "Stage 2: Perspective (E2P)")
        self.tab_widget.addTab(stage2_cube_scroll, "Stage 2: Cubemap (E2C)")
        self.tab_widget.addTab(stage3_scroll, "Stage 3: Masking Settings")
        
        # BOTTOM SECTION: Log Panel (collapsible)
        log_panel = self.create_log_panel()
        
        splitter.addWidget(self.tab_widget)
        splitter.addWidget(log_panel)
        
        # Set initial sizes: 60% for tabs, 40% for log
        splitter.setStretchFactor(0, 3)  # Tabs get more space
        splitter.setStretchFactor(1, 2)  # Log gets less but still visible
        
        return splitter
    
    def _wrap_in_scroll_area(self, widget: QWidget) -> QScrollArea:
        """Wrap a widget in a scroll area for responsive design"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(widget)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        return scroll
    
    def create_top_section(self) -> QWidget:
        """Create top section: Input/Output + Pipeline Overview + Action Buttons"""
        
        section = QWidget()
        section.setObjectName("topSection")
        
        layout = QVBoxLayout(section)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        
        # === INPUT/OUTPUT CONFIGURATION ===
        io_group = QGroupBox("Input / Output Configuration")
        io_layout = QVBoxLayout()
        
        # Input file
        input_layout = QHBoxLayout()
        input_label = QLabel("Input File:")
        input_label.setMinimumWidth(100)
        input_layout.addWidget(input_label)
        self.input_file_edit = QLineEdit()
        self.input_file_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        input_layout.addWidget(self.input_file_edit, stretch=1)
        input_browse_btn = QPushButton("Browse...")
        input_browse_btn.setMinimumWidth(80)
        input_browse_btn.clicked.connect(self.browse_input_file)
        input_layout.addWidget(input_browse_btn, stretch=0)
        io_layout.addLayout(input_layout)
        
        # Output directory
        output_layout = QHBoxLayout()
        output_label = QLabel("Output Directory:")
        output_label.setMinimumWidth(100)
        output_layout.addWidget(output_label)
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        output_layout.addWidget(self.output_dir_edit, stretch=1)
        output_browse_btn = QPushButton("Browse...")
        output_browse_btn.setMinimumWidth(80)
        output_browse_btn.clicked.connect(self.browse_output_dir)
        output_layout.addWidget(output_browse_btn, stretch=0)
        io_layout.addLayout(output_layout)
        
        io_group.setLayout(io_layout)
        layout.addWidget(io_group)
        
        # === PIPELINE OVERVIEW ===
        pipeline_group = QGroupBox("Pipeline Stages")
        pipeline_layout = QHBoxLayout()
        
        # Stage 1
        stage1_layout = QVBoxLayout()
        self.stage1_enable = QCheckBox("Stage 1: Extract Frames")
        self.stage1_enable.setChecked(True)
        stage1_layout.addWidget(self.stage1_enable)
        self.run_stage1_btn = QPushButton("▶ Run Stage 1")
        self.run_stage1_btn.setMinimumHeight(32)
        self.run_stage1_btn.clicked.connect(self.run_stage_1_only)
        stage1_layout.addWidget(self.run_stage1_btn)
        pipeline_layout.addLayout(stage1_layout, stretch=1)
        
        pipeline_layout.addWidget(self._create_separator())
        
        # Stage 2
        stage2_layout = QVBoxLayout()
        stage2_header = QHBoxLayout()
        self.stage2_enable = QCheckBox("Stage 2: Split Views")
        self.stage2_enable.setChecked(True)
        stage2_header.addWidget(self.stage2_enable)
        self.stage2_method_combo = QComboBox()
        self.stage2_method_combo.addItem("Perspective (E2P)", "perspective")
        self.stage2_method_combo.addItem("Cubemap (E2C)", "cubemap")
        self.stage2_method_combo.setMinimumWidth(150)
        self.stage2_method_combo.currentIndexChanged.connect(self.on_stage2_method_changed)
        stage2_header.addWidget(self.stage2_method_combo)
        stage2_layout.addLayout(stage2_header)
        
        # Skip Transform checkbox (Direct Masking Mode)
        self.skip_transform_check = QCheckBox("⏩ Skip Transform (Direct Mask)")
        self.skip_transform_check.setChecked(False)
        self.skip_transform_check.setToolTip(
            "Skip perspective splitting and mask equirectangular/fisheye images directly.\n"
            "Faster workflow for 360° VR or native photogrammetry."
        )
        self.skip_transform_check.setStyleSheet("color: #4a9eff; font-weight: bold;")
        self.skip_transform_check.toggled.connect(self.on_skip_transform_toggled)
        stage2_layout.addWidget(self.skip_transform_check)
        
        self.run_stage2_btn = QPushButton("▶ Run Stage 2")
        self.run_stage2_btn.setMinimumHeight(32)
        self.run_stage2_btn.clicked.connect(self.run_stage_2_only)
        stage2_layout.addWidget(self.run_stage2_btn)
        pipeline_layout.addLayout(stage2_layout, stretch=1)
        
        pipeline_layout.addWidget(self._create_separator())
        
        # Stage 3
        stage3_layout = QVBoxLayout()
        self.stage3_enable = QCheckBox("Stage 3: Generate Masks")
        self.stage3_enable.setChecked(True)
        stage3_layout.addWidget(self.stage3_enable)
        self.run_stage3_btn = QPushButton("▶ Run Stage 3")
        self.run_stage3_btn.setMinimumHeight(32)
        self.run_stage3_btn.clicked.connect(self.run_stage_3_only)
        stage3_layout.addWidget(self.run_stage3_btn)
        pipeline_layout.addLayout(stage3_layout, stretch=1)
        
        pipeline_group.setLayout(pipeline_layout)
        layout.addWidget(pipeline_group)
        
        # === ACTION BUTTONS + PROGRESS ===
        action_layout = QHBoxLayout()
        
        # Start button - green with play icon
        self.start_button = QPushButton("▶ Start Pipeline")
        self.start_button.setMinimumSize(150, 45)
        self.start_button.setMaximumSize(200, 55)
        self.start_button.setObjectName("startButton")
        self.start_button.clicked.connect(self.start_pipeline)
        action_layout.addWidget(self.start_button, stretch=0)
        
        # Pause button - yellow/orange with pause icon
        self.pause_button = QPushButton("⏸ Pause")
        self.pause_button.setMinimumSize(100, 45)
        self.pause_button.setMaximumSize(140, 55)
        self.pause_button.setObjectName("pauseButton")
        self.pause_button.setEnabled(False)
        self.pause_button.clicked.connect(self.toggle_pause)
        action_layout.addWidget(self.pause_button, stretch=0)
        
        # Stop button - red with stop icon
        self.stop_button = QPushButton("⏹ Stop")
        self.stop_button.setMinimumSize(100, 45)
        self.stop_button.setMaximumSize(140, 55)
        self.stop_button.setObjectName("stopButton")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_pipeline)
        action_layout.addWidget(self.stop_button, stretch=0)
        
        action_layout.addSpacing(20)
        
        # Progress bar (expands to fill space)
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setMinimumHeight(30)
        self.progress_bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        action_layout.addWidget(self.progress_bar, stretch=1)
        
        action_layout.addSpacing(10)
        
        self.status_label = QLabel("Ready to start")
        self.status_label.setMinimumWidth(100)
        action_layout.addWidget(self.status_label)
        
        layout.addLayout(action_layout)
        
        return section
    
    def _create_separator(self) -> QWidget:
        """Create vertical separator line"""
        sep = QWidget()
        sep.setFixedWidth(2)
        sep.setStyleSheet("background-color: #555;")
        return sep
    
    def create_stage1_config_tab(self) -> QWidget:
        """Create Stage 1 configuration tab (Extraction Settings)"""
        
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(16, 16, 16, 16)
        
        # File Analysis
        analysis_group = QGroupBox("File Analysis")
        analysis_layout = QVBoxLayout()
        
        analyze_layout = QHBoxLayout()
        analyze_btn = QPushButton("Analyze Input File")
        analyze_btn.clicked.connect(self.analyze_video_file)
        analyze_layout.addWidget(analyze_btn)
        analyze_layout.addStretch()
        analysis_layout.addLayout(analyze_layout)
        
        # File metadata display
        self.file_metadata_label = QLabel("No file analyzed")
        self.file_metadata_label.setWordWrap(True)
        self.file_metadata_label.setStyleSheet("padding: 8px; background: #3c3c3c; border-radius: 4px;")
        analysis_layout.addWidget(self.file_metadata_label)
        
        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group)
        
        # Time Range Selection
        time_group = QGroupBox("Time Range")
        time_layout = QVBoxLayout()
        
        self.full_video_check = QCheckBox("Extract full video")
        self.full_video_check.setChecked(True)
        self.full_video_check.toggled.connect(self.toggle_time_range)
        time_layout.addWidget(self.full_video_check)
        
        time_controls = QHBoxLayout()
        time_controls.addWidget(QLabel("Start Time (s):"))
        self.start_time_spin = QDoubleSpinBox()
        self.start_time_spin.setRange(0, 999999)
        self.start_time_spin.setValue(0)
        self.start_time_spin.setEnabled(False)
        time_controls.addWidget(self.start_time_spin)
        
        time_controls.addWidget(QLabel("End Time (s):"))
        self.end_time_spin = QDoubleSpinBox()
        self.end_time_spin.setRange(0, 999999)
        self.end_time_spin.setValue(0)
        self.end_time_spin.setEnabled(False)
        time_controls.addWidget(self.end_time_spin)
        time_controls.addStretch()
        time_layout.addLayout(time_controls)
        
        time_group.setLayout(time_layout)
        layout.addWidget(time_group)
        
        # Extraction settings
        extract_group = QGroupBox("Extraction Settings")
        extract_layout = QVBoxLayout()
        
        # FPS
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel("Frame Rate (FPS):"))
        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setRange(0.1, 30.0)
        self.fps_spin.setValue(DEFAULT_FPS)
        self.fps_spin.setSingleStep(0.1)
        fps_layout.addWidget(self.fps_spin)
        fps_layout.addStretch()
        extract_layout.addLayout(fps_layout)
        
        # Method
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Extraction Method:"))
        self.extraction_method_combo = QComboBox()
        for key, value in EXTRACTION_METHODS.items():
            self.extraction_method_combo.addItem(value, key)
        self.extraction_method_combo.currentIndexChanged.connect(self.on_extraction_method_changed)
        method_layout.addWidget(self.extraction_method_combo)
        method_layout.addStretch()
        extract_layout.addLayout(method_layout)
        
        # SDK Quality (visible when SDK selected)
        self.sdk_quality_widget = QWidget()
        sdk_quality_layout = QHBoxLayout(self.sdk_quality_widget)
        sdk_quality_layout.setContentsMargins(0, 0, 0, 0)
        self.sdk_quality_label = QLabel("SDK Quality:")
        sdk_quality_layout.addWidget(self.sdk_quality_label)
        self.sdk_quality_combo = QComboBox()
        # Use SDK_QUALITY_OPTIONS from config (now includes method names)
        from src.config.defaults import SDK_QUALITY_OPTIONS
        for key, label in SDK_QUALITY_OPTIONS.items():
            self.sdk_quality_combo.addItem(label, key)
        default_quality_index = self.sdk_quality_combo.findData(DEFAULT_SDK_QUALITY)
        if default_quality_index >= 0:
            self.sdk_quality_combo.setCurrentIndex(default_quality_index)
        sdk_quality_layout.addWidget(self.sdk_quality_combo)
        sdk_quality_layout.addStretch()
        extract_layout.addWidget(self.sdk_quality_widget)
        self.sdk_quality_widget.setVisible(False)  # Hidden by default
        
        # SDK Resolution
        self.sdk_res_widget = QWidget()
        sdk_res_layout = QHBoxLayout(self.sdk_res_widget)
        sdk_res_layout.setContentsMargins(0, 0, 0, 0)
        self.sdk_res_label = QLabel("SDK Resolution:")
        sdk_res_layout.addWidget(self.sdk_res_label)
        self.sdk_resolution_combo = QComboBox()
        self.sdk_resolution_combo.addItem("Original", "original")
        self.sdk_resolution_combo.addItem("8K (7680×3840)", "8k")
        self.sdk_resolution_combo.addItem("6K (6144×3072)", "6k")
        self.sdk_resolution_combo.addItem("4K (3840×1920)", "4k")
        self.sdk_resolution_combo.addItem("2K (1920×960)", "2k")
        sdk_res_layout.addWidget(self.sdk_resolution_combo)
        sdk_res_layout.addStretch()
        extract_layout.addWidget(self.sdk_res_widget)
        self.sdk_res_widget.setVisible(False)  # Hidden by default
        
        # Output Format
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Output Format:"))
        self.output_format_combo = QComboBox()
        self.output_format_combo.addItem("PNG (Lossless)", "png")
        self.output_format_combo.addItem("JPEG (Compressed)", "jpeg")
        format_layout.addWidget(self.output_format_combo)
        format_layout.addStretch()
        extract_layout.addLayout(format_layout)
        
        extract_group.setLayout(extract_layout)
        layout.addWidget(extract_group)
        
        layout.addStretch()
        
        return tab
    
    def create_stage2_perspective_tab(self) -> QWidget:
        """Create Stage 2 Perspective (E2P) configuration tab"""
        
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(16, 16, 16, 16)
        
        # Camera configuration
        camera_group = QGroupBox("Camera Configuration")
        camera_layout = QVBoxLayout()
        
        # Split count
        split_layout = QHBoxLayout()
        split_layout.addWidget(QLabel("Camera Count:"))
        self.split_count_spin = QSpinBox()
        self.split_count_spin.setRange(1, 12)
        self.split_count_spin.setValue(DEFAULT_SPLIT_COUNT)
        self.split_count_spin.valueChanged.connect(self.on_split_count_changed)
        split_layout.addWidget(self.split_count_spin)
        split_layout.addStretch()
        camera_layout.addLayout(split_layout)
        
        # FOV
        fov_layout = QHBoxLayout()
        fov_layout.addWidget(QLabel("Horizontal FOV (°):"))
        self.fov_spin = QSpinBox()
        self.fov_spin.setRange(30, 150)
        self.fov_spin.setValue(DEFAULT_H_FOV)
        fov_layout.addWidget(self.fov_spin)
        fov_layout.addStretch()
        camera_layout.addLayout(fov_layout)
        
        camera_group.setLayout(camera_layout)
        self.stage2_camera_group = camera_group  # Store reference for enabling/disabling
        layout.addWidget(camera_group)
        
        # Output Settings for Stage 2
        output_group = QGroupBox("Output Settings")
        output_layout = QVBoxLayout()
        
        # Resolution
        res_layout = QHBoxLayout()
        res_layout.addWidget(QLabel("Output Resolution:"))
        res_layout.addWidget(QLabel("Width:"))
        self.stage2_width_spin = QSpinBox()
        self.stage2_width_spin.setRange(640, 7680)
        self.stage2_width_spin.setValue(1920)  # CHANGED: Default 1920 for square images
        self.stage2_width_spin.setSingleStep(128)
        res_layout.addWidget(self.stage2_width_spin)
        res_layout.addWidget(QLabel("Height:"))
        self.stage2_height_spin = QSpinBox()
        self.stage2_height_spin.setRange(480, 3840)
        self.stage2_height_spin.setValue(1920)  # CHANGED: Default 1920 for square images
        self.stage2_height_spin.setSingleStep(128)
        res_layout.addWidget(self.stage2_height_spin)
        res_layout.addStretch()
        output_layout.addLayout(res_layout)
        
        # Format
        format2_layout = QHBoxLayout()
        format2_layout.addWidget(QLabel("Image Format:"))
        self.stage2_format_combo = QComboBox()
        self.stage2_format_combo.addItem("PNG (Lossless)", "png")
        self.stage2_format_combo.addItem("JPEG (Compressed)", "jpeg")
        self.stage2_format_combo.addItem("TIFF (High Quality)", "tiff")
        format2_layout.addWidget(self.stage2_format_combo)
        format2_layout.addStretch()
        output_layout.addLayout(format2_layout)
        
        output_group.setLayout(output_layout)
        self.stage2_output_group = output_group  # Store reference
        layout.addWidget(output_group)
        
        # Perspective-specific parameters
        perspective_params_group = QGroupBox("Perspective Parameters (E2P)")
        perspective_params_layout = QVBoxLayout()
        
        # Pitch offset
        pitch_layout = QHBoxLayout()
        pitch_layout.addWidget(QLabel("Pitch Offset (°):"))
        self.pitch_offset_spin = QSpinBox()
        self.pitch_offset_spin.setRange(-90, 90)
        self.pitch_offset_spin.setValue(0)
        pitch_layout.addWidget(self.pitch_offset_spin)
        pitch_layout.addStretch()
        perspective_params_layout.addLayout(pitch_layout)
        
        # Roll offset
        roll_layout = QHBoxLayout()
        roll_layout.addWidget(QLabel("Roll Offset (°):"))
        self.roll_offset_spin = QSpinBox()
        self.roll_offset_spin.setRange(-180, 180)
        self.roll_offset_spin.setValue(0)
        roll_layout.addWidget(self.roll_offset_spin)
        roll_layout.addStretch()
        perspective_params_layout.addLayout(roll_layout)
        
        perspective_params_group.setLayout(perspective_params_layout)
        self.stage2_perspective_params_group = perspective_params_group  # Store reference
        layout.addWidget(perspective_params_group)
        
        layout.addStretch()
        return tab
    
    def create_stage2_cubemap_tab(self) -> QWidget:
        """Create Stage 2 Cubemap (E2C) configuration tab
        
        SPECIFICATIONS:
        - 6-tile: Only "Separate Files" output (no layouts needed)
        - 8-tile: FOV OR overlap percentage (user chooses which to control)
        - Resolution: Input field with auto-calculated default (input_height / 2)
        """
        
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(16, 16, 16, 16)
        
        # Output Settings
        output_group = QGroupBox("Output Settings")
        output_layout = QVBoxLayout()
        
        # Face/Tile Size (editable field with auto-calculated default)
        res_layout = QHBoxLayout()
        res_layout.addWidget(QLabel("Face/Tile Size (px):"))
        self.cubemap_face_size_spin = QSpinBox()
        self.cubemap_face_size_spin.setRange(512, 8192)
        self.cubemap_face_size_spin.setValue(1920)  # Default, will be auto-calculated on file load
        self.cubemap_face_size_spin.setSingleStep(128)
        self.cubemap_face_size_spin.setToolTip("Default: input_height / 2. Each tile is square.")
        res_layout.addWidget(self.cubemap_face_size_spin)
        res_info = QLabel("(Auto-calculated from input, editable)")
        res_info.setStyleSheet("color: gray; font-size: 10px;")
        res_layout.addWidget(res_info)
        res_layout.addStretch()
        output_layout.addLayout(res_layout)
        
        # Format
        format2_layout = QHBoxLayout()
        format2_layout.addWidget(QLabel("Image Format:"))
        self.cubemap_format_combo = QComboBox()
        self.cubemap_format_combo.addItem("PNG (Lossless)", "png")
        self.cubemap_format_combo.addItem("JPEG (Compressed)", "jpeg")
        self.cubemap_format_combo.addItem("TIFF (High Quality)", "tiff")
        format2_layout.addWidget(self.cubemap_format_combo)
        format2_layout.addStretch()
        output_layout.addLayout(format2_layout)
        
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        # Cubemap-specific parameters
        cubemap_params_group = QGroupBox("Cubemap Parameters (E2C)")
        cubemap_params_layout = QVBoxLayout()
        
        # Info label
        info_label = QLabel(
            "<b>Cubemap Types:</b><br>"
            "• <b>6-Tile Standard</b>: Cubemap for VR/rendering - 90° FOV fixed, separate files only<br>"
            "• <b>8-Tile Grid</b>: For photogrammetry - 4×2 grid with adjustable FOV or overlap"
        )
        info_label.setWordWrap(True)
        cubemap_params_layout.addWidget(info_label)
        
        # Cubemap type
        cubemap_type_layout = QHBoxLayout()
        cubemap_type_layout.addWidget(QLabel("Cubemap Type:"))
        self.cubemap_type_combo = QComboBox()
        self.cubemap_type_combo.addItem("6-Tile Standard Cubemap (90° FOV, Separate Files)", "6-face")
        self.cubemap_type_combo.addItem("8-Tile Grid (Photogrammetry/Gaussian Splatting)", "8-tile")
        self.cubemap_type_combo.setCurrentIndex(1)  # CHANGED: Default to 8-tile
        self.cubemap_type_combo.currentIndexChanged.connect(self.on_cubemap_type_changed)
        cubemap_type_layout.addWidget(self.cubemap_type_combo)
        cubemap_type_layout.addStretch()
        cubemap_params_layout.addLayout(cubemap_type_layout)
        
        # 8-Tile Grid Controls (only for 8-tile mode)
        self.tile_8_controls_widget = QWidget()
        tile_8_layout = QVBoxLayout(self.tile_8_controls_widget)
        tile_8_layout.setContentsMargins(0, 0, 0, 0)
        
        # Option to use FOV or Overlap
        control_mode_layout = QHBoxLayout()
        control_mode_layout.addWidget(QLabel("Control Method:"))
        self.tile_control_mode_combo = QComboBox()
        self.tile_control_mode_combo.addItem("Set FOV (Auto-calculate overlap)", "fov")
        self.tile_control_mode_combo.addItem("Set Overlap % (Auto-calculate FOV)", "overlap")
        self.tile_control_mode_combo.currentIndexChanged.connect(self.on_tile_control_mode_changed)
        control_mode_layout.addWidget(self.tile_control_mode_combo)
        control_mode_layout.addStretch()
        tile_8_layout.addLayout(control_mode_layout)
        
        # FOV control (when control_mode = 'fov')
        self.fov_control_widget = QWidget()
        fov_control_layout = QHBoxLayout(self.fov_control_widget)
        fov_control_layout.setContentsMargins(0, 0, 0, 0)
        fov_control_layout.addWidget(QLabel("Horizontal FOV (°):"))
        self.cubemap_fov_spin = QSpinBox()
        self.cubemap_fov_spin.setRange(45, 150)
        self.cubemap_fov_spin.setValue(110)  # CHANGED: Default 110° FOV
        self.cubemap_fov_spin.valueChanged.connect(self.on_fov_changed)
        fov_control_layout.addWidget(self.cubemap_fov_spin)
        self.fov_overlap_label = QLabel("→ Overlap: ~55%")  # Will update dynamically
        self.fov_overlap_label.setStyleSheet("color: gray;")
        fov_control_layout.addWidget(self.fov_overlap_label)
        fov_control_layout.addStretch()
        tile_8_layout.addWidget(self.fov_control_widget)
        
        # Overlap control (when control_mode = 'overlap')
        self.overlap_control_widget = QWidget()
        overlap_control_layout = QHBoxLayout(self.overlap_control_widget)
        overlap_control_layout.setContentsMargins(0, 0, 0, 0)
        overlap_control_layout.addWidget(QLabel("Overlap (%):"))
        self.overlap_spin = QSpinBox()
        self.overlap_spin.setRange(0, 75)
        self.overlap_spin.setValue(25)
        self.overlap_spin.valueChanged.connect(self.on_overlap_changed)
        overlap_control_layout.addWidget(self.overlap_spin)
        self.overlap_fov_label = QLabel("→ FOV: ~67°")
        self.overlap_fov_label.setStyleSheet("color: gray;")
        overlap_control_layout.addWidget(self.overlap_fov_label)
        overlap_control_layout.addStretch()
        tile_8_layout.addWidget(self.overlap_control_widget)
        
        # Initially show FOV control, hide overlap control
        self.overlap_control_widget.setVisible(False)
        
        cubemap_params_layout.addWidget(self.tile_8_controls_widget)
        
        # CHANGED: Show 8-tile controls by default (8-tile is now default)
        self.tile_8_controls_widget.setVisible(True)
        
        cubemap_params_group.setLayout(cubemap_params_layout)
        layout.addWidget(cubemap_params_group)
        
        layout.addStretch()
        return tab
    
    def create_stage3_config_tab(self) -> QWidget:
        """Create Stage 3 configuration tab (Masking Settings with YOLO26)"""
        
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        # === YOLO26 HEADER ===
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 8)
        
        yolo_label = QLabel("🚀 YOLO26 AI Masking")
        yolo_label.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
            color: #4a9eff;
        """)
        header_layout.addWidget(yolo_label)
        
        yolo_info = QLabel("(NMS-Free • 3-4x Faster • GPU Accelerated)")
        yolo_info.setStyleSheet("color: #888; font-size: 11px;")
        header_layout.addWidget(yolo_info)
        header_layout.addStretch()
        layout.addWidget(header_widget)
        
        # Enable masking
        self.enable_masking_check = QCheckBox("Enable AI Masking")
        self.enable_masking_check.setChecked(True)
        self.enable_masking_check.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.enable_masking_check)
        
        # === PERSONS CATEGORY ===
        persons_group = QGroupBox("👤 Persons")
        persons_group.setCheckable(True)
        persons_group.setChecked(True)
        persons_layout = QVBoxLayout()
        persons_layout.setSpacing(4)
        
        self.person_check = QCheckBox("Person (class 0)")
        self.person_check.setChecked(True)
        persons_layout.addWidget(self.person_check)
        
        persons_group.setLayout(persons_layout)
        layout.addWidget(persons_group)
        self.persons_group = persons_group  # Store reference
        
        # === PERSONAL OBJECTS CATEGORY ===
        objects_group = QGroupBox("🎒 Personal Objects")
        objects_group.setCheckable(True)
        objects_group.setChecked(True)
        objects_layout = QVBoxLayout()
        objects_layout.setSpacing(4)
        
        # Individual object checkboxes (COCO classes)
        self.backpack_check = QCheckBox("Backpack (class 24)")
        self.backpack_check.setChecked(True)
        objects_layout.addWidget(self.backpack_check)
        
        self.umbrella_check = QCheckBox("Umbrella (class 25)")
        self.umbrella_check.setChecked(True)
        objects_layout.addWidget(self.umbrella_check)
        
        self.handbag_check = QCheckBox("Handbag (class 26)")
        self.handbag_check.setChecked(True)
        objects_layout.addWidget(self.handbag_check)
        
        self.tie_check = QCheckBox("Tie (class 27)")
        self.tie_check.setChecked(True)
        objects_layout.addWidget(self.tie_check)
        
        self.suitcase_check = QCheckBox("Suitcase (class 28)")
        self.suitcase_check.setChecked(True)
        objects_layout.addWidget(self.suitcase_check)
        
        self.cell_phone_check = QCheckBox("Cell Phone (class 67)")
        self.cell_phone_check.setChecked(True)
        objects_layout.addWidget(self.cell_phone_check)
        
        objects_group.setLayout(objects_layout)
        layout.addWidget(objects_group)
        self.objects_group = objects_group  # Store reference
        
        # === ANIMALS CATEGORY ===
        animals_group = QGroupBox("🐕 Animals")
        animals_group.setCheckable(True)
        animals_group.setChecked(False)  # Disabled by default
        animals_layout = QVBoxLayout()
        animals_layout.setSpacing(4)
        
        self.bird_check = QCheckBox("Bird (class 14)")
        self.bird_check.setChecked(True)
        animals_layout.addWidget(self.bird_check)
        
        self.cat_check = QCheckBox("Cat (class 15)")
        self.cat_check.setChecked(True)
        animals_layout.addWidget(self.cat_check)
        
        self.dog_check = QCheckBox("Dog (class 16)")
        self.dog_check.setChecked(True)
        animals_layout.addWidget(self.dog_check)
        
        self.horse_check = QCheckBox("Horse (class 17)")
        self.horse_check.setChecked(True)
        animals_layout.addWidget(self.horse_check)
        
        # Other animals in one row
        other_animals_label = QLabel("Other: sheep, cow, elephant, bear, zebra, giraffe")
        other_animals_label.setStyleSheet("color: #888; font-size: 10px; margin-left: 20px;")
        animals_layout.addWidget(other_animals_label)
        
        animals_group.setLayout(animals_layout)
        layout.addWidget(animals_group)
        self.animals_group = animals_group  # Store reference
        
        # === MODEL SETTINGS ===
        model_group = QGroupBox("⚙️ Model Settings")
        model_layout = QVBoxLayout()
        model_layout.setSpacing(8)
        
        # Model size
        size_layout = QHBoxLayout()
        size_label = QLabel("Model Size:")
        size_label.setMinimumWidth(120)
        size_layout.addWidget(size_label)
        self.model_size_combo = QComboBox()
        # YOLO26 models
        self.model_size_combo.addItem("Nano (10MB, ~0.2s) - Fastest", "nano")
        self.model_size_combo.addItem("Small (40MB, ~0.4s) - Balanced", "small")
        self.model_size_combo.addItem("Medium (90MB, ~0.7s) - Best Quality", "medium")
        self.model_size_combo.setCurrentIndex(2)  # Default to 'medium'
        self.model_size_combo.setMinimumWidth(250)
        size_layout.addWidget(self.model_size_combo)
        size_layout.addStretch()
        model_layout.addLayout(size_layout)
        
        # Confidence
        conf_layout = QHBoxLayout()
        conf_label = QLabel("Confidence:")
        conf_label.setMinimumWidth(120)
        conf_layout.addWidget(conf_label)
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.0, 1.0)
        self.confidence_spin.setValue(0.6)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setMinimumWidth(80)
        conf_layout.addWidget(self.confidence_spin)
        
        conf_info = QLabel("(0.5-0.7 recommended)")
        conf_info.setStyleSheet("color: #888;")
        conf_layout.addWidget(conf_info)
        conf_layout.addStretch()
        model_layout.addLayout(conf_layout)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # === GPU ACCELERATION ===
        gpu_group = QGroupBox("🚀 GPU Acceleration")
        gpu_layout = QVBoxLayout()
        
        self.use_gpu_check = QCheckBox("Enable GPU Acceleration (CUDA)")
        self.use_gpu_check.setChecked(True)
        self.use_gpu_check.setToolTip("Use GPU for faster masking (requires CUDA)")
        gpu_layout.addWidget(self.use_gpu_check)
        
        gpu_info = QLabel("⚡ 3-4x faster with GPU enabled")
        gpu_info.setStyleSheet("color: #4a9eff; font-size: 10px; margin-left: 20px;")
        gpu_layout.addWidget(gpu_info)
        
        gpu_group.setLayout(gpu_layout)
        layout.addWidget(gpu_group)
        
        layout.addStretch()
        
        return tab
    
    def create_log_panel(self) -> QWidget:
        """Create responsive log output panel (no fixed height)"""
        
        panel = QWidget()
        # Remove fixed height - let splitter control size
        panel.setMinimumHeight(120)  # Minimum readable height
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        
        layout.addWidget(QLabel("Processing Log:"))
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        # Make log expand to fill available space
        self.log_text.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.log_text)
        
        return panel
    
    def browse_input_file(self):
        """Browse for input video file"""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select Input Video",
            "",
            "Video Files (*.insv *.mp4 *.mov);;All Files (*.*)"
        )
        if filename:
            self.input_file_edit.setText(filename)
            # Auto-analyze the file
            self.analyze_video_file()
    
    def analyze_video_file(self):
        """Analyze video file and display metadata"""
        from src.extraction import FrameExtractor
        
        input_file = self.input_file_edit.text()
        if not input_file or not Path(input_file).exists():
            return
        
        extractor = FrameExtractor()
        info = extractor.get_video_info(input_file)
        
        if info.get('success'):
            # Display metadata
            metadata_text = f"""<b>File Type:</b> {info.get('file_type_desc', 'Unknown')}<br>
<b>Duration:</b> {info.get('duration_formatted', 'N/A')} ({info.get('duration_seconds', 0)} seconds)<br>
<b>Resolution:</b> {info.get('resolution', 'N/A')}<br>
<b>FPS:</b> {info.get('fps', 0):.2f}<br>
<b>Frame Count:</b> {info.get('frame_count', 0):,}<br>
<b>Camera Model:</b> {info.get('camera_model', 'Unknown')}<br>
<b>File Size:</b> {info.get('file_size_mb', 0):.2f} MB"""
            
            self.file_metadata_label.setText(metadata_text)
            
            # Update end time spin with video duration
            duration = info.get('duration', 0)
            self.end_time_spin.setMaximum(duration)
            self.end_time_spin.setValue(duration)
            
            # Auto-calculate cubemap face size (input_height / 2)
            height = info.get('height', 0)
            if height > 0:
                auto_face_size = height // 2
                # Round to nearest 128
                auto_face_size = ((auto_face_size + 64) // 128) * 128
                self.cubemap_face_size_spin.setValue(auto_face_size)
                self.log_message(f"Analyzed: {Path(input_file).name} | Auto-set face size: {auto_face_size}px (height/2)")
            else:
                self.log_message(f"Analyzed: {Path(input_file).name}")
        else:
            self.file_metadata_label.setText(f"<b>Error:</b> {info.get('error', 'Unknown error')}")
    
    def toggle_time_range(self, checked: bool):
        """Toggle time range controls"""
        self.start_time_spin.setEnabled(not checked)
        self.end_time_spin.setEnabled(not checked)
    
    def on_extraction_method_changed(self, index: int):
        """Show/hide SDK-specific controls based on extraction method"""
        method = self.extraction_method_combo.currentData()
        is_sdk = method in ['sdk', 'sdk_stitching']
        
        # Show/hide entire widget containers (includes labels + combos)
        self.sdk_quality_widget.setVisible(is_sdk)
        self.sdk_res_widget.setVisible(is_sdk)
    
    def on_split_count_changed(self, value: int):
        """Update compass when split count changes"""
        # Compass functionality removed (preview disabled)
        pass
    
    def on_skip_transform_toggled(self, checked: bool):
        """Handle skip transform checkbox toggle"""
        # Enable/disable Stage 2 transform controls in tabs
        self.stage2_camera_group.setEnabled(not checked)
        self.stage2_output_group.setEnabled(not checked)
        self.stage2_perspective_params_group.setEnabled(not checked)
        
        # Also disable the Run Stage 2 button when skip is enabled
        self.run_stage2_btn.setEnabled(not checked)
        
        if checked:
            self.log_message("⏩ Direct Masking Mode enabled - Stage 2 will be skipped")
        else:
            self.log_message("ℹ️ Direct Masking Mode disabled - Stage 2 transform enabled")
    
    def on_stage2_method_changed(self, index: int):
        """Handle Stage 2 method selection change"""
        method = self.stage2_method_combo.currentData()
        
        # Switch to appropriate tab
        if method == 'perspective':
            self.tab_widget.setCurrentIndex(1)  # Perspective tab
            self.log_message("Stage 2: Perspective Transform (E2P) selected")
        elif method == 'cubemap':
            self.tab_widget.setCurrentIndex(2)  # Cubemap tab
            self.log_message("Stage 2: Cubemap Transform (E2C) selected")
    
    def on_cubemap_type_changed(self, index: int):
        """Handle cubemap type selection change (6-tile vs 8-tile)"""
        cubemap_type = self.cubemap_type_combo.currentData()
        
        if cubemap_type == '6-face':
            # 6-tile: Hide all controls (90° FOV fixed, separate files only)
            self.tile_8_controls_widget.setVisible(False)
            self.log_message("Cubemap: 6-Tile Standard (90° FOV fixed, separate files)")
        else:  # 8-tile
            # 8-tile: Show FOV/overlap controls
            self.tile_8_controls_widget.setVisible(True)
            self.log_message("Cubemap: 8-Tile Grid (photogrammetry mode)")
    
    def on_tile_control_mode_changed(self, index: int):
        """Handle 8-tile control mode change (FOV vs Overlap)"""
        mode = self.tile_control_mode_combo.currentData()
        
        if mode == 'fov':
            self.fov_control_widget.setVisible(True)
            self.overlap_control_widget.setVisible(False)
            self.on_fov_changed()  # Update overlap calculation
        else:  # overlap
            self.fov_control_widget.setVisible(False)
            self.overlap_control_widget.setVisible(True)
            self.on_overlap_changed()  # Update FOV calculation
    
    def on_fov_changed(self):
        """Calculate and display overlap when FOV changes (8-tile grid)"""
        # 8 tiles around 360° = 45° step
        step_size = 360.0 / 8
        fov = self.cubemap_fov_spin.value()
        overlap = fov - step_size
        overlap_percent = (overlap / step_size) * 100
        self.fov_overlap_label.setText(f"→ Overlap: ~{overlap_percent:.0f}%")
    
    def on_overlap_changed(self):
        """Calculate and display FOV when overlap changes (8-tile grid)"""
        # 8 tiles around 360° = 45° step
        step_size = 360.0 / 8
        overlap_percent = self.overlap_spin.value()
        overlap_degrees = (overlap_percent / 100.0) * step_size
        fov = step_size + overlap_degrees
        self.overlap_fov_label.setText(f"→ FOV: ~{fov:.0f}°")
    
    def browse_output_dir(self):
        """Browse for output directory"""
        dirname = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory"
        )
        if dirname:
            self.output_dir_edit.setText(dirname)
    
    def run_stage_1_only(self):
        """Run only Stage 1 (Extraction)"""
        self.log_message("Running Stage 1 only: Frame Extraction")
        
        # Set flag to enable auto-advance after stage completes
        self._auto_advance_enabled = True
        
        # Temporarily disable other stages
        stage2_state = self.stage2_enable.isChecked()
        stage3_state = self.stage3_enable.isChecked()
        
        self.stage2_enable.setChecked(False)
        self.stage3_enable.setChecked(False)
        
        # Run pipeline
        self.start_pipeline()
        
        # Restore states
        self.stage2_enable.setChecked(stage2_state)
        self.stage3_enable.setChecked(stage3_state)
    
    def run_stage_2_only(self):
        """Run only Stage 2 (Split Views) - Auto-discovers input from Stage 1 output"""
        self.log_message("Running Stage 2 only: Split Perspectives/Cubemap")
        
        output_dir = self.output_dir_edit.text()
        
        if not output_dir:
            QMessageBox.warning(self, "Missing Output Dir", "Please configure output directory first")
            return
        
        # Auto-discover Stage 1 output
        from src.pipeline.batch_orchestrator import PipelineWorker
        worker = PipelineWorker({})  # Dummy worker just for discovery method
        
        stage1_folder = worker.discover_stage_input_folder(stage=2, output_dir=output_dir)
        
        if not stage1_folder:
            # Not found - ask user to select manually
            self.log_message("[!] Stage 1 output folder not found. Selecting manually...")
            folder = QFileDialog.getExistingDirectory(
                self,
                "Select Stage 1 Output Folder (equirectangular images)",
                str(Path(output_dir))
            )
            if not folder:
                self.log_message("Stage 2 cancelled - no input folder selected")
                return
            stage1_folder = Path(folder)
        else:
            self.log_message(f"[OK] Auto-discovered Stage 1 output: {stage1_folder}")
        
        # Verify folder has images (case-insensitive)
        images = []
        for ext in ['*.png', '*.PNG', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG']:
            images.extend(stage1_folder.glob(ext))
        if not images:
            QMessageBox.warning(self, "No Images Found", f"No images in: {stage1_folder}")
            return
        
        self.log_message(f"[OK] Found {len(images)} equirectangular images")
        
        # Set flag to enable auto-advance after stage completes
        self._auto_advance_enabled = True
        
        # Temporarily disable other stages and set input
        stage1_state = self.stage1_enable.isChecked()
        stage3_state = self.stage3_enable.isChecked()
        
        self.stage1_enable.setChecked(False)
        self.stage3_enable.setChecked(False)
        
        # Set Stage 2 input and run
        self.pipeline_config['stage2_input_dir'] = str(stage1_folder)
        self.start_pipeline()
        
        # Restore states
        self.stage1_enable.setChecked(stage1_state)
        self.stage3_enable.setChecked(stage3_state)
    
    def run_stage_3_only(self):
        """Run only Stage 3 (Masking) - Auto-discovers input from Stage 2 output"""
        self.log_message("Running Stage 3 only: Generate Masks")
        
        output_dir = self.output_dir_edit.text()
        
        if not output_dir:
            QMessageBox.warning(self, "Missing Output Dir", "Please configure output directory first")
            return
        
        # Auto-discover Stage 2 output
        from src.pipeline.batch_orchestrator import PipelineWorker
        worker = PipelineWorker({})  # Dummy worker just for discovery method
        
        stage2_folder = worker.discover_stage_input_folder(stage=3, output_dir=output_dir)
        
        if not stage2_folder:
            # Not found - ask user to select manually
            self.log_message("[!] Stage 2 output folder not found. Selecting manually...")
            folder = QFileDialog.getExistingDirectory(
                self,
                "Select Stage 2 Output Folder (perspective images)",
                str(Path(output_dir))
            )
            if not folder:
                self.log_message("Stage 3 cancelled - no input folder selected")
                return
            stage2_folder = Path(folder)
        else:
            self.log_message(f"[OK] Auto-discovered Stage 2 output: {stage2_folder}")
        
        # Verify folder has images (case-insensitive)
        images = []
        for ext in ['*.png', '*.PNG', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG']:
            images.extend(stage2_folder.glob(ext))
        if not images:
            QMessageBox.warning(self, "No Images Found", f"No images in: {stage2_folder}")
            return
        
        self.log_message(f"[OK] Found {len(images)} perspective images")
        
        # Set flag to enable auto-advance (though Stage 3 is final, this keeps consistency)
        self._auto_advance_enabled = True
        
        # Temporarily disable other stages and set input
        stage1_state = self.stage1_enable.isChecked()
        stage2_state = self.stage2_enable.isChecked()
        
        self.stage1_enable.setChecked(False)
        self.stage2_enable.setChecked(False)
        
        # Set Stage 3 input and run
        self.pipeline_config['stage3_input_dir'] = str(stage2_folder)
        self.start_pipeline()
        
        # Restore states
        self.stage1_enable.setChecked(stage1_state)
        self.stage2_enable.setChecked(stage2_state)
    
    def start_pipeline(self):
        """Start the pipeline execution"""
        
        # Disable auto-advance for Full Pipeline mode (user clicked "Start Pipeline" button)
        # Auto-advance is only enabled when running stage-only methods (run_stage_X_only)
        if not hasattr(self, '_auto_advance_enabled') or not self._auto_advance_enabled:
            self._auto_advance_enabled = False
        
        # Validate inputs
        input_file = self.input_file_edit.text()
        output_dir = self.output_dir_edit.text()
        
        if not input_file:
            QMessageBox.warning(self, "Input Required", "Please select an input video file.")
            return
        
        if not output_dir:
            QMessageBox.warning(self, "Output Required", "Please select an output directory.")
            return
        
        if not Path(input_file).exists():
            QMessageBox.warning(self, "File Not Found", f"Input file does not exist:\n{input_file}")
            return
        
        # Get Stage 2 method
        stage2_method = self.stage2_method_combo.currentData()
        
        # Get skip intermediate setting from settings manager
        from src.config.settings import get_settings
        settings = get_settings()
        skip_intermediate = settings.get_skip_intermediate_save()
        
        # Build configuration
        self.pipeline_config = {
            'input_file': input_file,
            'output_dir': output_dir,
            'enable_stage1': self.stage1_enable.isChecked(),
            'skip_transform': self.skip_transform_check.isChecked(),
            'enable_stage2': self.stage2_enable.isChecked() and not self.skip_transform_check.isChecked(),
            'enable_stage3': self.stage3_enable.isChecked(),
            'skip_intermediate_save': skip_intermediate,  # Performance: Use temp folder for Stage 1
            
            # Stage 1
            'fps': self.fps_spin.value(),
            'extraction_method': self.extraction_method_combo.currentData(),
            'start_time': 0.0 if self.full_video_check.isChecked() else self.start_time_spin.value(),
            'end_time': None if self.full_video_check.isChecked() else self.end_time_spin.value(),
            'sdk_quality': self.sdk_quality_combo.currentData(),
            'sdk_resolution': self.sdk_resolution_combo.currentData(),
            'output_format': self.output_format_combo.currentData(),
            
            # Stage 2 - Common settings
            'transform_type': stage2_method,
            'camera_config': {
                'cameras': self._generate_camera_positions()
            },
        }
        
        # Add Stage 2 method-specific parameters
        if stage2_method == 'perspective':
            self.pipeline_config.update({
                'output_width': self.stage2_width_spin.value(),
                'output_height': self.stage2_height_spin.value(),
                'stage2_format': self.stage2_format_combo.currentData(),
                'perspective_params': {
                    'pitch_offset': self.pitch_offset_spin.value(),
                    'roll_offset': self.roll_offset_spin.value()
                }
            })
        elif stage2_method == 'cubemap':
            # Get face/tile size from spinner
            face_size = self.cubemap_face_size_spin.value()
            cubemap_type = self.cubemap_type_combo.currentData()
            
            # For 8-tile, get FOV and overlap based on control mode
            if cubemap_type == '8-tile':
                control_mode = self.tile_control_mode_combo.currentData()
                if control_mode == 'fov':
                    fov = self.cubemap_fov_spin.value()
                    step_size = 360.0 / 8
                    overlap_percent = ((fov - step_size) / step_size) * 100
                else:  # overlap
                    overlap_percent = self.overlap_spin.value()
                    step_size = 360.0 / 8
                    fov = step_size + ((overlap_percent / 100.0) * step_size)
            else:
                # 6-tile: Fixed 90° FOV, no overlap
                fov = 90
                overlap_percent = 0
            
            self.pipeline_config.update({
                'output_width': face_size,
                'output_height': face_size,
                'stage2_format': self.cubemap_format_combo.currentData(),
                'cubemap_params': {
                    'cubemap_type': cubemap_type,  # '6-face' or '8-tile'
                    'face_size': face_size,
                    'overlap_percent': overlap_percent,
                    'fov': fov,
                    'layout': 'separate'  # Always separate files for both modes
                }
            })
        
        # Stage 3 (if enabled)
        if self.stage3_enable.isChecked():
            # Build list of enabled person classes
            person_classes = []
            if self.persons_group.isChecked() and self.person_check.isChecked():
                person_classes.append(0)  # person
            
            # Build list of enabled personal object classes
            object_classes = []
            if self.objects_group.isChecked():
                if self.backpack_check.isChecked():
                    object_classes.append(24)
                if self.umbrella_check.isChecked():
                    object_classes.append(25)
                if self.handbag_check.isChecked():
                    object_classes.append(26)
                if self.tie_check.isChecked():
                    object_classes.append(27)
                if self.suitcase_check.isChecked():
                    object_classes.append(28)
                if self.cell_phone_check.isChecked():
                    object_classes.append(67)
            
            # Build list of enabled animal classes
            animal_classes = []
            if self.animals_group.isChecked():
                if self.bird_check.isChecked():
                    animal_classes.append(14)
                if self.cat_check.isChecked():
                    animal_classes.append(15)
                if self.dog_check.isChecked():
                    animal_classes.append(16)
                if self.horse_check.isChecked():
                    animal_classes.append(17)
                # Add remaining animals (sheep through giraffe)
                animal_classes.extend([18, 19, 20, 21, 22, 23])
            
            self.pipeline_config.update({
                'model_size': self.model_size_combo.currentData(),
                'confidence_threshold': self.confidence_spin.value(),
                'use_gpu': True,
                'masking_categories': {
                    'persons': len(person_classes) > 0,
                    'personal_objects': len(object_classes) > 0,
                    'animals': len(animal_classes) > 0
                },
                # Pass specific class IDs for fine-grained control
                'masking_classes': {
                    'persons': person_classes,
                    'personal_objects': object_classes,
                    'animals': animal_classes
                }
            })
        
        # Disable start button
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.pause_button.setEnabled(True)
        self._is_paused = False
        self.pause_button.setText("⏸ Pause")
        
        # Clear log
        self.log_text.clear()
        self.log_message("Starting pipeline...")
        
        # Start pipeline
        self.orchestrator.run_pipeline(
            config=self.pipeline_config,
            progress_callback=self.on_progress,
            stage_complete_callback=self.on_stage_complete,
            finished_callback=self.on_finished,
            error_callback=self.on_error
        )
    
    def stop_pipeline(self):
        """Stop the pipeline execution"""
        self.orchestrator.cancel()
        self.log_message("Pipeline stopped by user")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.pause_button.setEnabled(False)
    
    def toggle_pause(self):
        """Toggle pause/resume pipeline"""
        if hasattr(self, '_is_paused') and self._is_paused:
            # Resume
            self.orchestrator.resume()
            self.pause_button.setText("⏸ Pause")
            self.log_message("Pipeline resumed")
            self._is_paused = False
        else:
            # Pause
            self.orchestrator.pause()
            self.pause_button.setText("▶ Resume")
            self.log_message("Pipeline paused")
            self._is_paused = True
    
    def on_progress(self, current: int, total: int, message: str):
        """Handle progress updates"""
        if total > 0:
            progress = int((current / total) * 100)
            self.progress_bar.setValue(progress)
        
        self.status_label.setText(message)
        self.log_message(message)
    
    def on_stage_complete(self, stage_number: int, results: dict):
        """Handle stage completion and auto-advance to next stage"""
        if results.get('success'):
            self.log_message(f"✓ Stage {stage_number} complete")
            
            # Auto-advance ONLY if running in stage-only mode (not Full Pipeline)
            # Full Pipeline runs all stages sequentially without needing auto-advance
            if hasattr(self, '_auto_advance_enabled') and self._auto_advance_enabled:
                if stage_number == 1 and self.stage2_enable.isChecked():
                    self.log_message("→ Auto-advancing to Stage 2...")
                    QTimer.singleShot(500, self.run_stage_2_only)
                
                elif stage_number == 2 and self.stage3_enable.isChecked():
                    self.log_message("→ Auto-advancing to Stage 3...")
                    QTimer.singleShot(500, self.run_stage_3_only)
                
                elif stage_number == 3:
                    self.log_message("✓ All stages complete!")
                    self._auto_advance_enabled = False  # Reset flag
        
        else:
            self.log_message(f"✗ Stage {stage_number} failed: {results.get('error')}")
            self._auto_advance_enabled = False  # Reset flag on error
            # Stop auto-advance on error
    
    def on_finished(self, results: dict):
        """Handle pipeline completion"""
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.pause_button.setEnabled(False)
        self.progress_bar.setValue(100)
        
        if results.get('success'):
            self.log_message("✓ Pipeline complete!")
            QMessageBox.information(self, "Success", "Pipeline completed successfully!")
        else:
            self.log_message(f"✗ Pipeline failed: {results.get('error')}")
            QMessageBox.warning(self, "Pipeline Failed", f"Pipeline failed:\n{results.get('error')}")
    
    def on_error(self, error_message: str):
        """Handle pipeline errors"""
        self.log_message(f"✗ Error: {error_message}")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.pause_button.setEnabled(False)
    
    def log_message(self, message: str):
        """Add message to log panel"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
    
    def _generate_camera_positions(self) -> list:
        """Generate camera positions based on split count and transform type
        
        Returns camera positions for perspective mode.
        For cubemap mode, this is not used (cubemaps have fixed face positions).
        """
        # Get Stage 2 method
        stage2_method = self.stage2_method_combo.currentData()
        
        # Use appropriate widgets based on method
        if stage2_method == 'perspective':
            split_count = self.split_count_spin.value()
            fov = self.fov_spin.value()
            pitch_offset = self.pitch_offset_spin.value()
            roll_offset = self.roll_offset_spin.value()
        elif stage2_method == 'cubemap':
            # For cubemaps, camera positions are generated by the E2C transform
            # Return empty list - pipeline will handle cubemap-specific logic
            return []
        else:
            # Default fallback
            split_count = 8
            fov = 110
            pitch_offset = 0
            roll_offset = 0
        
        cameras = []
        for i in range(split_count):
            yaw = (360 / split_count) * i
            cameras.append({
                'yaw': yaw,
                'pitch': pitch_offset,  # Apply user-defined pitch offset
                'roll': roll_offset,    # Apply user-defined roll offset
                'fov': fov
            })
        
        return cameras
    
    def apply_dark_theme(self):
        """Apply dark theme stylesheet with improved 4K scaling"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: #e0e0e0;
                font-size: 11px;
            }
            QWidget {
                background-color: #2b2b2b;
                color: #e0e0e0;
                font-size: 11px;
            }
            QGroupBox {
                border: 1px solid #555555;
                border-radius: 6px;
                margin-top: 12px;
                padding: 12px;
                padding-top: 16px;
                font-weight: bold;
                font-size: 12px;
            }
            QGroupBox::title {
                color: #e0e0e0;
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                font-size: 12px;
            }
            QLabel {
                font-size: 11px;
                padding: 2px;
            }
            QCheckBox {
                font-size: 11px;
                spacing: 8px;
                padding: 4px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QPushButton {
                background-color: #3c3c3c;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 10px 20px;
                color: #e0e0e0;
                font-size: 12px;
                min-height: 24px;
            }
            QPushButton:hover {
                background-color: #4a9eff;
            }
            QPushButton:pressed {
                background-color: #3a7fcf;
            }
            QPushButton:disabled {
                background-color: #666666;
                color: #a0a0a0;
            }
            QPushButton#startButton {
                background-color: #28a745;
                border: 2px solid #1e7e34;
                font-weight: bold;
                font-size: 14px;
                color: white;
                min-height: 36px;
            }
            QPushButton#startButton:hover {
                background-color: #34ce57;
                border: 2px solid #28a745;
            }
            QPushButton#startButton:pressed {
                background-color: #1e7e34;
            }
            QPushButton#pauseButton {
                background-color: #ffc107;
                border: 2px solid #e0a800;
                font-weight: bold;
                font-size: 13px;
                color: #212529;
                min-height: 36px;
            }
            QPushButton#pauseButton:hover {
                background-color: #ffca2c;
                border: 2px solid #ffc107;
            }
            QPushButton#pauseButton:pressed {
                background-color: #e0a800;
            }
            QPushButton#stopButton {
                background-color: #dc3545;
                border: 2px solid #bd2130;
                font-weight: bold;
                font-size: 13px;
                color: white;
                min-height: 36px;
            }
            QPushButton#stopButton:hover {
                background-color: #e4606d;
                border: 2px solid #dc3545;
            }
            QPushButton#stopButton:pressed {
                background-color: #bd2130;
            }
            QWidget#controlPanel {
                background-color: #3c3c3c;
                border-bottom: 1px solid #555555;
            }
            QProgressBar {
                border: 1px solid #555555;
                border-radius: 4px;
                background-color: #2b2b2b;
                text-align: center;
                color: #e0e0e0;
                font-size: 12px;
                min-height: 24px;
            }
            QProgressBar::chunk {
                background-color: #4a9eff;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #3c3c3c;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 6px 8px;
                color: #e0e0e0;
                font-size: 11px;
                min-height: 24px;
            }
            QComboBox {
                min-width: 120px;
            }
            QComboBox::drop-down {
                width: 24px;
            }
            QSpinBox::up-button, QSpinBox::down-button,
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                width: 20px;
            }
            QTextEdit {
                background-color: #1e1e1e;
                border: 1px solid #555555;
                color: #e0e0e0;
                font-family: Consolas, Monaco, monospace;
                font-size: 10px;
            }
            QTabWidget::pane {
                border: 1px solid #555555;
                background-color: #2b2b2b;
            }
            QTabBar::tab {
                background-color: #3c3c3c;
                border: 1px solid #555555;
                padding: 10px 20px;
                color: #e0e0e0;
                font-size: 12px;
                min-width: 140px;
            }
            QTabBar::tab:selected {
                background-color: #4a9eff;
                font-weight: bold;
            }
            QScrollArea {
                border: none;
            }
            QScrollBar:vertical {
                width: 14px;
                background: #2b2b2b;
            }
            QScrollBar::handle:vertical {
                background: #555555;
                min-height: 30px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical:hover {
                background: #666666;
            }
        """)
