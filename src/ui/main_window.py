"""
360FrameTools - Main Window (Full-Screen UI/UX Rewrite)
Modern PyQt6 interface with sidebar navigation, full-screen layout,
and polished dark theme for professional photogrammetry workflow.
"""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QProgressBar, QTabWidget,
    QFileDialog, QMessageBox, QTextEdit, QGroupBox,
    QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
    QLineEdit, QSplitter, QScrollArea, QSizePolicy,
    QRadioButton, QButtonGroup, QFrame, QStackedWidget,
    QApplication, QToolButton, QGridLayout, QSpacerItem, QStyle
)
from PyQt6.QtCore import Qt, QTimer, QSize, QPropertyAnimation, QEasingCurve, pyqtSignal
from PyQt6.QtGui import QFont, QAction, QIcon, QPainter, QColor, QPen, QPixmap, QShortcut, QKeySequence, QPalette

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
from src.ui.styles import build_theme_stylesheet
from src.ui.widgets import (
    StageHeader as SharedStageHeader,
    CardSection as SharedCardSection,
    StageSummaryStrip,
    StageActionFooter,
    FormRow,
    MediaProcessingPanel,
)
from src.ui.preview_panels import EquirectPreviewWidget

logger = logging.getLogger(__name__)


class SidebarButton(QPushButton):
    """Custom sidebar navigation button with icon and text"""
    
    def __init__(self, text, icon_char="", parent=None):
        super().__init__(parent)
        self.setText(f"  {icon_char}  {text}")
        self.setCheckable(True)
        self.setFixedHeight(48)
        self.setCursor(Qt.CursorShape.PointingHandCursor)


class StageHeader(SharedStageHeader):
    """Backward-compatible alias to shared StageHeader scaffold."""


class CardWidget(SharedCardSection):
    """Backward-compatible alias to shared CardSection scaffold."""


class MainWindow(QMainWindow):
    """Main application window - Full Screen UI/UX"""
    
    def __init__(self):
        super().__init__()
        
        self.settings = get_settings()
        self.config_manager = get_config_manager()
        self.orchestrator = BatchOrchestrator()
        self.pipeline_config = {}
        self._is_paused = False
        self._auto_advance_enabled = False
        self._control_bar_compact = False
        self._user_log_visible = True
        self._effective_log_visible = True
        self._resolved_theme = "dark"
        
        self.init_ui()
        self.create_menu_bar()
        self.apply_theme()
        self._setup_shortcuts()
        
        # Trigger initial visibility  
        self.on_extraction_method_changed(0)
        self._update_overview_stage_summary()
        self.on_settings_changed()

        # Connect input file path to Stage 1 equirectangular preview
        if hasattr(self, 'stage1_eq_preview'):
            self.input_file_edit.textChanged.connect(self.stage1_eq_preview.set_video_path)
        
        # Show maximized by default
        self.showMaximized()
    
    def init_ui(self):
        """Initialize full-screen UI with sidebar + stacked content"""
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setMinimumSize(1200, 800)
        
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        
        # Main body: Sidebar + Content + Right Log
        body = QSplitter(Qt.Orientation.Horizontal)
        body.setHandleWidth(1)
        self.main_splitter = body
        
        # LEFT: Sidebar navigation
        self.sidebar = self._create_sidebar()
        body.addWidget(self.sidebar)
        
        # CENTER: header bar + stacked content
        content_wrapper = QWidget()
        content_wrapper.setObjectName("contentWrapper")
        content_layout = QVBoxLayout(content_wrapper)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        
        # Pipeline control bar (always visible)
        self.control_bar = self._create_control_bar()
        content_layout.addWidget(self.control_bar)
        
        # Stacked pages
        self.page_stack = QStackedWidget()
        self.page_stack.addWidget(self._create_overview_page())       # 0
        self.page_stack.addWidget(self._create_stage1_page())         # 1
        self.page_stack.addWidget(self._create_stage2_persp_page())   # 2
        self.page_stack.addWidget(self._create_stage2_cube_page())    # 3
        self.page_stack.addWidget(self._create_stage3_page())         # 4
        self.page_stack.addWidget(self._create_stage4_page())         # 5
        self.page_stack.addWidget(self._create_stage5_page())         # 6
        content_layout.addWidget(self.page_stack, stretch=1)

        body.addWidget(content_wrapper)

        # RIGHT: Log panel
        log_panel = self._create_log_panel()
        body.addWidget(log_panel)

        body.setStretchFactor(0, 0)  # Sidebar fixed
        body.setStretchFactor(1, 1)  # Content expands
        body.setStretchFactor(2, 0)  # Right log panel
        
        # Sidebar + content + right log panel
        body.setSizes([196, 1080, 280])
        QTimer.singleShot(0, self._update_responsive_layout)
        
        root_layout.addWidget(body)
    
    # ========================================================================
    # SIDEBAR
    # ========================================================================
    def _create_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(196)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # App branding
        brand = QLabel(f"  360toolkit")
        brand.setObjectName("sidebarBrand")
        brand.setFixedHeight(56)
        layout.addWidget(brand)
        
        version_label = QLabel(f"  v{APP_VERSION}")
        version_label.setObjectName("sidebarVersion")
        version_label.setFixedHeight(24)
        layout.addWidget(version_label)
        
        layout.addSpacing(12)
        
        # Navigation buttons
        self.nav_buttons = []
        nav_items = [
            ("Overview",              "\u2302"),   # ⌂
            ("Frame Extraction",     "1"),
            ("Perspective Split",    "2"),
            ("Cubemap Split",        "2"),
            ("AI Masking",           "3"),
            ("3D Reconstruction",    "4"),
            ("Training",             "5"),
        ]
        
        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)
        
        for i, (text, icon) in enumerate(nav_items):
            btn = SidebarButton(text, icon)
            self.nav_group.addButton(btn, i)
            layout.addWidget(btn)
            self.nav_buttons.append(btn)
        
        self.nav_buttons[0].setChecked(True)
        self.nav_group.idClicked.connect(self._on_nav_clicked)
        
        layout.addStretch()
        
        # Bottom: GPU status indicator
        self.gpu_status_label = QLabel("  GPU: Detecting...")
        self.gpu_status_label.setObjectName("sidebarGpuStatus")
        self.gpu_status_label.setProperty("status", "idle")
        self.gpu_status_label.setFixedHeight(32)
        layout.addWidget(self.gpu_status_label)
        QTimer.singleShot(500, self._detect_gpu)
        
        return sidebar
    
    def _on_nav_clicked(self, idx):
        self.page_stack.setCurrentIndex(idx)
    
    def _detect_gpu(self):
        """Detect GPU and update status.
        
        This is the first place torch gets imported in the frozen app.
        The runtime hook only sets DLL paths and env vars - torch import
        is deferred here to avoid C-extension double-initialization errors.
        """
        import logging
        import sys as _sys
        gpu_logger = logging.getLogger(__name__)
        try:
            import torch
            
            gpu_logger.info(f"[GPU Detect] PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                # Configure PyTorch performance (normally done in runtime hook)
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
                
                name = torch.cuda.get_device_name(0)
                gpu_logger.info(f"[GPU Detect] GPU: {name}")
                # Try tensor test for real compatibility
                try:
                    t = torch.zeros(1, device='cuda') + 1
                    del t
                    gpu_logger.info(f"[GPU Detect] CUDA kernel test PASSED - full GPU acceleration enabled")
                    self.gpu_status_label.setText(f"  GPU: {name}")
                    self.gpu_status_label.setProperty("status", "ok")
                    self._refresh_widget_style(self.gpu_status_label)
                except Exception as e:
                    gpu_logger.warning(f"[GPU Detect] CUDA kernel test FAILED: {e}")
                    self.gpu_status_label.setText(f"  GPU: {name} (CPU mode)")
                    self.gpu_status_label.setProperty("status", "warn")
                    self._refresh_widget_style(self.gpu_status_label)
            else:
                gpu_logger.info("[GPU Detect] CUDA not available - CPU only mode")
                self.gpu_status_label.setText("  GPU: CPU only")
                self.gpu_status_label.setProperty("status", "warn")
                self._refresh_widget_style(self.gpu_status_label)
        except Exception as e:
            import traceback
            gpu_logger.warning(f"[GPU Detect] PyTorch not available: {e}")
            gpu_logger.warning(f"[GPU Detect] Full traceback:\n{traceback.format_exc()}")
            self.gpu_status_label.setText("  GPU: PyTorch N/A")
            self.gpu_status_label.setProperty("status", "error")
            self._refresh_widget_style(self.gpu_status_label)
    
    # ========================================================================
    # CONTROL BAR (Pipeline actions + progress)
    # ========================================================================
    def _create_control_bar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("controlBar")
        bar.setFixedHeight(64)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(14, 8, 14, 8)
        layout.setSpacing(8)
        style = QApplication.style()
        
        # I/O compact
        self.input_label = QLabel("Input:")
        layout.addWidget(self.input_label)
        self.input_file_edit = QLineEdit()
        self.input_file_edit.setPlaceholderText("Select .INSV / .mp4 / equirect images folder...")
        self.input_file_edit.setMinimumWidth(130)
        layout.addWidget(self.input_file_edit, stretch=1)
        
        self.input_browse_btn = QPushButton("")
        self.input_browse_btn.setObjectName("iconButton")
        self.input_browse_btn.setToolTip("Browse input path")
        self.input_browse_btn.setFixedSize(34, 30)
        self.input_browse_btn.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        self.input_browse_btn.setIconSize(QSize(16, 16))
        self.input_browse_btn.clicked.connect(self.browse_input_file)
        layout.addWidget(self.input_browse_btn)
        
        # Separator
        self.control_sep1 = QFrame()
        self.control_sep1.setObjectName("controlSeparator")
        self.control_sep1.setFixedWidth(1)
        layout.addWidget(self.control_sep1)
        
        self.output_label = QLabel("Output:")
        layout.addWidget(self.output_label)
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("Output directory...")
        self.output_dir_edit.setMinimumWidth(120)
        layout.addWidget(self.output_dir_edit, stretch=1)
        
        self.output_browse_btn = QPushButton("")
        self.output_browse_btn.setObjectName("iconButton")
        self.output_browse_btn.setToolTip("Browse output directory")
        self.output_browse_btn.setFixedSize(34, 30)
        self.output_browse_btn.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        self.output_browse_btn.setIconSize(QSize(16, 16))
        self.output_browse_btn.clicked.connect(self.browse_output_dir)
        layout.addWidget(self.output_browse_btn)
        
        # Separator
        self.control_sep2 = QFrame()
        self.control_sep2.setObjectName("controlSeparator")
        self.control_sep2.setFixedWidth(1)
        layout.addWidget(self.control_sep2)
        
        # Action buttons
        self.start_button = QPushButton("Start")
        self.start_button.setObjectName("startButton")
        self.start_button.setFixedSize(90, 30)
        self.start_button.clicked.connect(self.start_pipeline)
        layout.addWidget(self.start_button)
        
        self.pause_button = QPushButton("Pause")
        self.pause_button.setObjectName("pauseButton")
        self.pause_button.setFixedSize(78, 30)
        self.pause_button.setEnabled(False)
        self.pause_button.clicked.connect(self.toggle_pause)
        layout.addWidget(self.pause_button)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("stopButton")
        self.stop_button.setFixedSize(78, 30)
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_pipeline)
        layout.addWidget(self.stop_button)

        self.log_toggle_button = QPushButton("")
        self.log_toggle_button.setObjectName("iconButton")
        self.log_toggle_button.setCheckable(True)
        self.log_toggle_button.setChecked(True)
        self.log_toggle_button.setFixedSize(34, 30)
        self.log_toggle_button.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView))
        self.log_toggle_button.setIconSize(QSize(16, 16))
        self.log_toggle_button.setToolTip("Show or hide the right log panel")
        self.log_toggle_button.toggled.connect(self._on_log_toggle_clicked)
        layout.addWidget(self.log_toggle_button)

        self.theme_toggle_button = QPushButton("◐")
        self.theme_toggle_button.setObjectName("iconButton")
        self.theme_toggle_button.setCheckable(True)
        self.theme_toggle_button.setFixedSize(34, 30)
        self.theme_toggle_button.setToolTip("Toggle dark/light theme")
        self.theme_toggle_button.clicked.connect(self._toggle_theme)
        layout.addWidget(self.theme_toggle_button)
        
        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(20)
        self.progress_bar.setMinimumWidth(100)
        self.progress_bar.setMaximumWidth(140)
        layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("controlStatus")
        self.status_label.setFixedWidth(58)
        layout.addWidget(self.status_label)
        
        return bar
    
    # ========================================================================
    # PAGE 0: OVERVIEW (Pipeline dashboard)
    # ========================================================================
    def _create_overview_page(self) -> QScrollArea:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(16)
        
        layout.addWidget(StageHeader(
            "Pipeline Overview",
            "Configure which stages to run and launch individual stages or the full pipeline."
        ))

        layout.addWidget(StageSummaryStrip(
            "Overview",
            "Enable the stages you need, configure details inside each stage page, then run the full pipeline from the header or footer."
        ))

        summary_card = CardWidget("Pipeline Readiness")
        self.overview_stage_status_label = QLabel("Enabled steps: 0/5")
        self.overview_stage_status_label.setObjectName("stageSummaryText")
        summary_card.addWidget(self.overview_stage_status_label)
        layout.addWidget(summary_card)
        
        # Stage toggles in a grid
        grid = QGridLayout()
        grid.setSpacing(16)
        
        # Frame Extraction
        card1 = CardWidget("Frame Extraction")
        self.stage1_enable = QCheckBox("Enable extraction from .INSV/.mp4")
        self.stage1_enable.setChecked(True)
        self.stage1_enable.toggled.connect(self._update_overview_stage_summary)
        card1.addWidget(self.stage1_enable)
        self.run_stage1_btn = QPushButton("Configure Extraction")
        self.run_stage1_btn.setFixedHeight(36)
        self.run_stage1_btn.clicked.connect(lambda: self._open_stage_page(1))
        self.run_stage1_btn.setObjectName("stageSecondaryButton")
        card1.addWidget(self.run_stage1_btn)
        grid.addWidget(card1, 0, 0)
        
        # Split Views
        card2 = CardWidget("Split Views")
        stage2_header = QHBoxLayout()
        self.stage2_enable = QCheckBox("Enable perspective/cubemap splitting")
        self.stage2_enable.setChecked(True)
        self.stage2_enable.toggled.connect(self._update_overview_stage_summary)
        stage2_header.addWidget(self.stage2_enable)
        self.stage2_method_combo = QComboBox()
        self.stage2_method_combo.addItem("Perspective (E2P)", "perspective")
        self.stage2_method_combo.addItem("Cubemap (E2C)", "cubemap")
        self.stage2_method_combo.setFixedWidth(180)
        self.stage2_method_combo.currentIndexChanged.connect(self.on_stage2_method_changed)
        stage2_header.addWidget(self.stage2_method_combo)
        card2.addLayout(stage2_header)
        self.skip_transform_check = QCheckBox("Skip Transform (Direct Mask)")
        self.skip_transform_check.setToolTip("Skip splitting, mask equirect images directly")
        self.skip_transform_check.toggled.connect(self.on_skip_transform_toggled)
        card2.addWidget(self.skip_transform_check)
        self.run_stage2_btn = QPushButton("Configure Split")
        self.run_stage2_btn.setFixedHeight(36)
        self.run_stage2_btn.clicked.connect(lambda: self._open_stage_page(2))
        self.run_stage2_btn.setObjectName("stageSecondaryButton")
        card2.addWidget(self.run_stage2_btn)
        grid.addWidget(card2, 0, 1)
        
        # AI Masking
        card3 = CardWidget("AI Masking")
        self.stage3_enable = QCheckBox("Enable AI person/object masking")
        self.stage3_enable.setChecked(True)
        self.stage3_enable.toggled.connect(self._update_overview_stage_summary)
        card3.addWidget(self.stage3_enable)
        self.run_stage3_btn = QPushButton("Configure Masking")
        self.run_stage3_btn.setFixedHeight(36)
        self.run_stage3_btn.clicked.connect(lambda: self._open_stage_page(4))
        self.run_stage3_btn.setObjectName("stageSecondaryButton")
        card3.addWidget(self.run_stage3_btn)
        grid.addWidget(card3, 1, 0)
        
        # 3D Reconstruction
        card4 = CardWidget("3D Reconstruction")
        self.stage4_enable = QCheckBox("Enable COLMAP reconstruction")
        self.stage4_enable.setChecked(True)
        self.stage4_enable.toggled.connect(self._update_overview_stage_summary)
        card4.addWidget(self.stage4_enable)
        self.run_stage4_btn = QPushButton("Configure Reconstruction")
        self.run_stage4_btn.setFixedHeight(36)
        self.run_stage4_btn.clicked.connect(lambda: self._open_stage_page(5))
        self.run_stage4_btn.setObjectName("stageSecondaryButton")
        card4.addWidget(self.run_stage4_btn)
        grid.addWidget(card4, 1, 1)
        
        # Training
        card5 = CardWidget("Training")
        self.stage5_enable = QCheckBox("Enable Gaussian Splatting training")
        self.stage5_enable.setChecked(False)
        self.stage5_enable.toggled.connect(self._update_overview_stage_summary)
        card5.addWidget(self.stage5_enable)
        self.run_stage5_btn = QPushButton("Configure Training")
        self.run_stage5_btn.setFixedHeight(36)
        self.run_stage5_btn.clicked.connect(lambda: self._open_stage_page(6))
        self.run_stage5_btn.setObjectName("stageSecondaryButton")
        card5.addWidget(self.run_stage5_btn)
        grid.addWidget(card5, 0, 2)
        
        layout.addLayout(grid)

        overview_footer = StageActionFooter("Start Full Pipeline")
        overview_footer.primary_button.clicked.connect(self.start_pipeline)
        overview_footer.validate_button.clicked.connect(lambda: self._validate_stage_config(0))
        layout.addWidget(overview_footer)

        layout.addStretch()
        
        return self._scroll_wrap(page)
    
    # ========================================================================
    # PAGE 1: FRAME EXTRACTION
    # ========================================================================
    def _create_stage1_page(self) -> QWidget:
        """
        Frame Extraction page — horizontal splitter:
          Left:  live 360° equirectangular preview (EquirectPreviewWidget)
          Right: scrollable config panel (existing cards + Media Processing)
        """
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(4)

        # ── LEFT: Equirectangular preview ────────────────────────────────────
        self.stage1_eq_preview = EquirectPreviewWidget()
        self.stage1_eq_preview.setMinimumWidth(260)
        splitter.addWidget(self.stage1_eq_preview)

        # ── RIGHT: Scrollable config ─────────────────────────────────────────
        right_page = QWidget()
        layout = QVBoxLayout(right_page)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        layout.addWidget(StageHeader(
            "Frame Extraction",
            "Extract equirectangular frames from Insta360 .INSV or .mp4 video files."
        ))

        layout.addWidget(StageSummaryStrip(
            "Extraction",
            "Configure extraction range, method, and quality, then validate metadata before running."
        ))

        # File Analysis card
        card_analysis = CardWidget("File Analysis")
        analyze_row = QHBoxLayout()
        analyze_btn = QPushButton("Analyze Input File")
        analyze_btn.setFixedWidth(180)
        analyze_btn.clicked.connect(self.analyze_video_file)
        analyze_row.addWidget(analyze_btn)
        analyze_row.addStretch()
        card_analysis.addLayout(analyze_row)

        self.file_metadata_label = QLabel("No file analyzed yet. Click 'Analyze Input File' to see metadata.")
        self.file_metadata_label.setObjectName("metadataLabel")
        self.file_metadata_label.setProperty("state", "idle")
        self.file_metadata_label.setWordWrap(True)
        card_analysis.addWidget(self.file_metadata_label)
        layout.addWidget(card_analysis)

        # Time Range card
        card_time = CardWidget("Time Range")
        self.full_video_check = QCheckBox("Extract full video")
        self.full_video_check.setChecked(True)
        self.full_video_check.toggled.connect(self.toggle_time_range)
        card_time.addWidget(self.full_video_check)

        time_range_widget = QWidget()
        time_row = QHBoxLayout(time_range_widget)
        time_row.setContentsMargins(0, 0, 0, 0)
        time_row.setSpacing(12)
        time_row.addWidget(QLabel("Start"))
        self.start_time_spin = QDoubleSpinBox()
        self.start_time_spin.setRange(0, 999999)
        self.start_time_spin.setValue(0)
        self.start_time_spin.setEnabled(False)
        self.start_time_spin.setFixedWidth(100)
        time_row.addWidget(self.start_time_spin)
        time_row.addWidget(QLabel("End"))
        self.end_time_spin = QDoubleSpinBox()
        self.end_time_spin.setRange(0, 999999)
        self.end_time_spin.setValue(0)
        self.end_time_spin.setEnabled(False)
        self.end_time_spin.setFixedWidth(100)
        time_row.addWidget(self.end_time_spin)
        time_row.addStretch()
        card_time.addWidget(FormRow("Range (s):", time_range_widget, "Start and end in seconds"))
        layout.addWidget(card_time)

        # Extraction Settings card
        card_extract = CardWidget("Extraction Settings")

        # FPS
        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setRange(0.1, 30.0)
        self.fps_spin.setValue(DEFAULT_FPS)
        self.fps_spin.setSingleStep(0.1)
        self.fps_spin.setFixedWidth(100)
        card_extract.addWidget(FormRow("Frame Rate (FPS):", self.fps_spin, "Range: 0.1 to 30.0"))

        # Method
        self.extraction_method_combo = QComboBox()
        for key, value in EXTRACTION_METHODS.items():
            self.extraction_method_combo.addItem(value, key)
        self.extraction_method_combo.setMinimumWidth(300)
        self.extraction_method_combo.currentIndexChanged.connect(self.on_extraction_method_changed)
        card_extract.addWidget(FormRow("Extraction Method:", self.extraction_method_combo))

        # SDK Quality (hidden by default)
        self.sdk_quality_widget = QWidget()
        sdk_q_layout = QHBoxLayout(self.sdk_quality_widget)
        sdk_q_layout.setContentsMargins(0, 0, 0, 0)
        sdk_q_layout.addWidget(QLabel("SDK Quality:"))
        self.sdk_quality_combo = QComboBox()
        from src.config.defaults import SDK_QUALITY_OPTIONS
        for key, label in SDK_QUALITY_OPTIONS.items():
            self.sdk_quality_combo.addItem(label, key)
        default_qi = self.sdk_quality_combo.findData(DEFAULT_SDK_QUALITY)
        if default_qi >= 0:
            self.sdk_quality_combo.setCurrentIndex(default_qi)
        self.sdk_quality_combo.setMinimumWidth(300)
        sdk_q_layout.addWidget(self.sdk_quality_combo)
        sdk_q_layout.addStretch()
        self.sdk_quality_widget.setVisible(False)
        card_extract.addWidget(self.sdk_quality_widget)

        # SDK Resolution (hidden by default)
        self.sdk_res_widget = QWidget()
        sdk_r_layout = QHBoxLayout(self.sdk_res_widget)
        sdk_r_layout.setContentsMargins(0, 0, 0, 0)
        sdk_r_layout.addWidget(QLabel("SDK Resolution:"))
        self.sdk_resolution_combo = QComboBox()
        self.sdk_resolution_combo.addItem("Original", "original")
        self.sdk_resolution_combo.addItem("8K (7680x3840)", "8k")
        self.sdk_resolution_combo.addItem("6K (6144x3072)", "6k")
        self.sdk_resolution_combo.addItem("4K (3840x1920)", "4k")
        self.sdk_resolution_combo.addItem("2K (1920x960)", "2k")
        self.sdk_resolution_combo.setCurrentIndex(1)
        self.sdk_resolution_combo.setMinimumWidth(200)
        sdk_r_layout.addWidget(self.sdk_resolution_combo)
        sdk_r_layout.addStretch()
        self.sdk_res_widget.setVisible(False)
        card_extract.addWidget(self.sdk_res_widget)

        # Output Format
        self.output_format_combo = QComboBox()
        self.output_format_combo.addItem("PNG (Lossless)", "png")
        self.output_format_combo.addItem("JPEG (Compressed)", "jpeg")
        self.output_format_combo.setFixedWidth(200)
        card_extract.addWidget(FormRow("Output Format:", self.output_format_combo))

        layout.addWidget(card_extract)

        # ── Media Processing card (SDK only) ─────────────────────────────────
        self.stage1_media_card = CardWidget("Media Processing")
        self.stage1_media_panel = MediaProcessingPanel()
        self.stage1_media_card.addWidget(self.stage1_media_panel)
        self.stage1_media_card.setVisible(False)   # shown only when SDK is selected
        layout.addWidget(self.stage1_media_card)

        # Footer
        stage1_footer = StageActionFooter("Run Extraction")
        stage1_footer.primary_button.clicked.connect(self.run_stage_1_only)
        stage1_footer.validate_button.clicked.connect(lambda: self._validate_stage_config(1))
        layout.addWidget(stage1_footer)

        layout.addStretch()

        # Wrap right side in a QScrollArea
        right_scroll = QScrollArea()
        right_scroll.setWidget(right_page)
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QFrame.Shape.NoFrame)
        right_scroll.setMinimumWidth(360)
        splitter.addWidget(right_scroll)

        # Set initial sizes: preview gets ~60 %, config gets ~40 %
        splitter.setSizes([600, 400])
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        # ── Connect signals ───────────────────────────────────────────────────
        # Preview updates when view orientation sliders change
        self.stage1_media_panel.view_changed.connect(
            lambda y, p, r: self.stage1_eq_preview.set_view(y, p, r)
        )
        # Preview updates when any color/option slider changes
        self.stage1_media_panel.values_changed.connect(
            self.stage1_eq_preview.set_color_opts
        )

        return splitter
    
    # ========================================================================
    # PAGE 2: PERSPECTIVE SPLIT
    # ========================================================================
    def _create_stage2_persp_page(self) -> QScrollArea:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(16)
        
        layout.addWidget(StageHeader(
            "Perspective Split (E2P)",
            "Convert equirectangular panoramas to multiple perspective camera views."
        ))

        layout.addWidget(StageSummaryStrip(
            "Perspective",
            "Configure perspective output size and camera groups for split-view workflows."
        ))
        
        # Output Resolution card
        card_output = CardWidget("Output Settings")

        dims_widget = QWidget()
        dims_layout = QHBoxLayout(dims_widget)
        dims_layout.setContentsMargins(0, 0, 0, 0)
        dims_layout.setSpacing(12)

        dims_layout.addWidget(QLabel("W"))
        self.stage2_width_spin = QSpinBox()
        self.stage2_width_spin.setRange(640, 7680)
        self.stage2_width_spin.setValue(1920)
        self.stage2_width_spin.setSingleStep(128)
        self.stage2_width_spin.setFixedWidth(100)
        dims_layout.addWidget(self.stage2_width_spin)

        dims_layout.addWidget(QLabel("H"))
        self.stage2_height_spin = QSpinBox()
        self.stage2_height_spin.setRange(480, 3840)
        self.stage2_height_spin.setValue(1920)
        self.stage2_height_spin.setSingleStep(128)
        self.stage2_height_spin.setFixedWidth(100)
        dims_layout.addWidget(self.stage2_height_spin)
        dims_layout.addStretch()

        self.stage2_format_combo = QComboBox()
        self.stage2_format_combo.addItem("PNG", "png")
        self.stage2_format_combo.addItem("JPEG", "jpeg")
        self.stage2_format_combo.addItem("TIFF", "tiff")
        self.stage2_format_combo.setFixedWidth(120)

        card_output.addWidget(FormRow("Output Size:", dims_widget, "Width × Height in pixels"))
        card_output.addWidget(FormRow("Format:", self.stage2_format_combo))
        self.stage2_output_group = card_output
        layout.addWidget(card_output)
        
        # Camera Groups card
        card_cameras = CardWidget("Camera Groups (Dome Configuration)")
        
        info_label = QLabel(
            "Each group creates a ring of cameras at a specific pitch angle. "
            "Pitch 0 = horizon, -30 = looking down, +30 = looking up."
        )
        info_label.setWordWrap(True)
        info_label.setProperty("role", "muted")
        card_cameras.addWidget(info_label)
        
        # Camera groups container
        self.camera_groups_container = QWidget()
        self.camera_groups_container_layout = QVBoxLayout(self.camera_groups_container)
        self.camera_groups_container_layout.setContentsMargins(0, 0, 0, 0)
        self.camera_groups_container_layout.setSpacing(8)
        self.camera_group_widgets = []
        
        self._add_camera_group(camera_count=8, pitch=0, fov=110, name="Horizon")
        self._add_camera_group(camera_count=8, pitch=-30, fov=110, name="Look Down")
        self._add_camera_group(camera_count=8, pitch=30, fov=110, name="Look Up")
        
        card_cameras.addWidget(self.camera_groups_container)
        
        add_btn = QPushButton("+ Add Camera Group")
        add_btn.setFixedWidth(180)
        add_btn.clicked.connect(lambda: self._add_camera_group())
        card_cameras.addWidget(add_btn)
        
        self.stage2_perspective_params_group = card_cameras
        layout.addWidget(card_cameras)

        stage2_footer = StageActionFooter("Run Perspective Split")
        stage2_footer.primary_button.clicked.connect(self.run_stage_2_only)
        stage2_footer.validate_button.clicked.connect(lambda: self._validate_stage_config(2))
        layout.addWidget(stage2_footer)

        layout.addStretch()
        return self._scroll_wrap(page)
    
    # ========================================================================
    # PAGE 3: CUBEMAP SPLIT
    # ========================================================================
    def _create_stage2_cube_page(self) -> QScrollArea:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(16)
        
        layout.addWidget(StageHeader(
            "Cubemap Split (E2C)",
            "Convert equirectangular panoramas to cubemap tiles for VR or photogrammetry."
        ))

        layout.addWidget(StageSummaryStrip(
            "Cubemap",
            "Configure cubemap tiling mode and tile dimensions; use 8-tile mode for photogrammetry output layouts."
        ))
        
        # Tile Size card
        card_tiles = CardWidget("Tile Dimensions")
        tile_dims_widget = QWidget()
        tw_row = QHBoxLayout(tile_dims_widget)
        tw_row.setContentsMargins(0, 0, 0, 0)
        tw_row.setSpacing(12)
        tw_row.addWidget(QLabel("W"))
        self.cubemap_tile_width_spin = QSpinBox()
        self.cubemap_tile_width_spin.setRange(512, 8192)
        self.cubemap_tile_width_spin.setValue(1920)
        self.cubemap_tile_width_spin.setSingleStep(128)
        self.cubemap_tile_width_spin.setFixedWidth(100)
        tw_row.addWidget(self.cubemap_tile_width_spin)
        tw_row.addWidget(QLabel("H"))
        self.cubemap_tile_height_spin = QSpinBox()
        self.cubemap_tile_height_spin.setRange(512, 8192)
        self.cubemap_tile_height_spin.setValue(1920)
        self.cubemap_tile_height_spin.setSingleStep(128)
        self.cubemap_tile_height_spin.setFixedWidth(100)
        tw_row.addWidget(self.cubemap_tile_height_spin)
        tw_row.addStretch()
        card_tiles.addWidget(FormRow("Tile Size (px):", tile_dims_widget, "Width × Height per cubemap tile"))
        
        self.cubemap_format_combo = QComboBox()
        self.cubemap_format_combo.addItem("PNG", "png")
        self.cubemap_format_combo.addItem("JPEG", "jpeg")
        self.cubemap_format_combo.addItem("TIFF", "tiff")
        self.cubemap_format_combo.setFixedWidth(120)
        card_tiles.addWidget(FormRow("Tile Format:", self.cubemap_format_combo))
        layout.addWidget(card_tiles)
        
        # Cubemap type card
        card_type = CardWidget("Cubemap Type")
        
        self.cubemap_type_combo = QComboBox()
        self.cubemap_type_combo.addItem("6-Tile Standard (90 FOV, Separate Files)", "6-face")
        self.cubemap_type_combo.addItem("8-Tile Grid (Photogrammetry)", "8-tile")
        self.cubemap_type_combo.setCurrentIndex(1)
        self.cubemap_type_combo.setMinimumWidth(360)
        self.cubemap_type_combo.currentIndexChanged.connect(self.on_cubemap_type_changed)
        card_type.addWidget(FormRow("Type:", self.cubemap_type_combo))
        
        # 8-tile info
        self.tile_8_controls_widget = QWidget()
        t8_layout = QVBoxLayout(self.tile_8_controls_widget)
        t8_layout.setContentsMargins(0, 8, 0, 0)
        t8_info = QLabel(
            "8-Tile: Each tile covers 90 horizontally in a 4x2 grid. "
            "Overlap is controlled by setting tile width wider than input_width/4."
        )
        t8_info.setWordWrap(True)
        t8_info.setProperty("role", "muted")
        t8_layout.addWidget(t8_info)
        self.tile_8_controls_widget.setVisible(True)
        card_type.addWidget(self.tile_8_controls_widget)
        
        layout.addWidget(card_type)

        stage2_cube_footer = StageActionFooter("Run Cubemap Split")
        stage2_cube_footer.primary_button.clicked.connect(self.run_stage_2_only)
        stage2_cube_footer.validate_button.clicked.connect(lambda: self._validate_stage_config(3))
        layout.addWidget(stage2_cube_footer)

        layout.addStretch()
        return self._scroll_wrap(page)
    
    # ========================================================================
    # PAGE 4: AI MASKING
    # ========================================================================
    def _create_stage3_page(self) -> QScrollArea:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(16)
        
        layout.addWidget(StageHeader(
            "AI Masking",
            "Detect and mask persons, objects, and animals using YOLO + SAM for photogrammetry."
        ))

        layout.addWidget(StageSummaryStrip(
            "Masking",
            "Choose masking target, categories, engine, and confidence, then validate before running."
        ))
        
        # Masking Target card
        card_target = CardWidget("Masking Target")
        
        target_info = QLabel("Choose where to apply masks based on your workflow:")
        target_info.setProperty("role", "secondary")
        card_target.addWidget(target_info)
        
        self.mask_split_radio = QRadioButton("Apply to Split Views (RealityScan Workflow)")
        self.mask_split_radio.setChecked(True)
        self.mask_split_radio.setToolTip("Extract -> Split -> Mask")
        card_target.addWidget(self.mask_split_radio)
        
        self.mask_equirect_radio = QRadioButton("Apply to Equirectangular (Rig SfM Workflow)")
        self.mask_equirect_radio.setToolTip("Extract -> Mask equirect -> Reconstruct")
        card_target.addWidget(self.mask_equirect_radio)
        
        tip = QLabel("Tip: Use 'Equirectangular' if enabling 3D Reconstruction for Gaussian Splatting.")
        tip.setObjectName("stageTip")
        tip.setWordWrap(True)
        card_target.addWidget(tip)
        layout.addWidget(card_target)
        
        # Detection categories card (2-column layout)
        cats_layout_h = QHBoxLayout()
        cats_layout_h.setSpacing(16)
        
        # Persons
        card_persons = CardWidget("Persons")
        self.persons_group = QGroupBox()
        self.persons_group.setObjectName("flatGroup")
        self.persons_group.setCheckable(True)
        self.persons_group.setChecked(True)
        self.persons_group.setFlat(True)
        pg_layout = QVBoxLayout(self.persons_group)
        pg_layout.setContentsMargins(0, 0, 0, 0)
        self.person_check = QCheckBox("Person (class 0)")
        self.person_check.setChecked(True)
        pg_layout.addWidget(self.person_check)
        card_persons.addWidget(self.persons_group)
        cats_layout_h.addWidget(card_persons)
        
        # Objects
        card_objects = CardWidget("Personal Objects")
        self.objects_group = QGroupBox()
        self.objects_group.setObjectName("flatGroup")
        self.objects_group.setCheckable(True)
        self.objects_group.setChecked(True)
        self.objects_group.setFlat(True)
        og_layout = QVBoxLayout(self.objects_group)
        og_layout.setContentsMargins(0, 0, 0, 0)
        self.backpack_check = QCheckBox("Backpack (24)")
        self.backpack_check.setChecked(True)
        og_layout.addWidget(self.backpack_check)
        self.umbrella_check = QCheckBox("Umbrella (25)")
        self.umbrella_check.setChecked(True)
        og_layout.addWidget(self.umbrella_check)
        self.handbag_check = QCheckBox("Handbag (26)")
        self.handbag_check.setChecked(True)
        og_layout.addWidget(self.handbag_check)
        self.tie_check = QCheckBox("Tie (27)")
        self.tie_check.setChecked(True)
        og_layout.addWidget(self.tie_check)
        self.suitcase_check = QCheckBox("Suitcase (28)")
        self.suitcase_check.setChecked(True)
        og_layout.addWidget(self.suitcase_check)
        self.cell_phone_check = QCheckBox("Cell Phone (67)")
        self.cell_phone_check.setChecked(True)
        og_layout.addWidget(self.cell_phone_check)
        card_objects.addWidget(self.objects_group)
        cats_layout_h.addWidget(card_objects)
        
        # Animals
        card_animals = CardWidget("Animals")
        self.animals_group = QGroupBox()
        self.animals_group.setObjectName("flatGroup")
        self.animals_group.setCheckable(True)
        self.animals_group.setChecked(False)
        self.animals_group.setFlat(True)
        ag_layout = QVBoxLayout(self.animals_group)
        ag_layout.setContentsMargins(0, 0, 0, 0)
        self.bird_check = QCheckBox("Bird (14)")
        self.bird_check.setChecked(True)
        ag_layout.addWidget(self.bird_check)
        self.cat_check = QCheckBox("Cat (15)")
        self.cat_check.setChecked(True)
        ag_layout.addWidget(self.cat_check)
        self.dog_check = QCheckBox("Dog (16)")
        self.dog_check.setChecked(True)
        ag_layout.addWidget(self.dog_check)
        self.horse_check = QCheckBox("Horse (17)")
        self.horse_check.setChecked(True)
        ag_layout.addWidget(self.horse_check)
        other_lbl = QLabel("+ sheep, cow, elephant, bear, zebra, giraffe")
        other_lbl.setProperty("role", "mutedSmall")
        ag_layout.addWidget(other_lbl)
        card_animals.addWidget(self.animals_group)
        cats_layout_h.addWidget(card_animals)
        
        layout.addLayout(cats_layout_h)
        
        # Model Settings card
        card_model = CardWidget("Model Settings")

        self.masking_engine_combo = QComboBox()
        self.masking_engine_combo.addItem("YOLO (ONNX) - Fast", "yolo_onnx")
        self.masking_engine_combo.addItem("YOLO (PyTorch) - Full", "yolo_pytorch")
        self.masking_engine_combo.addItem("SAM ViT-B - Best Quality", "sam_vitb")
        self.masking_engine_combo.addItem("YOLO+SAM Hybrid - Best", "hybrid")
        self.masking_engine_combo.setCurrentIndex(0)
        self.masking_engine_combo.setMinimumWidth(280)
        self.masking_engine_combo.currentIndexChanged.connect(self.on_masking_engine_changed)
        card_model.addWidget(FormRow("Masking Engine:", self.masking_engine_combo))
        
        self.engine_description_label = QLabel("NMS-free inference | 3-4x faster | CUDA accelerated")
        self.engine_description_label.setProperty("role", "mutedSmall")
        card_model.addWidget(self.engine_description_label)
        
        # Model size
        self.model_size_container = QWidget()
        ms_layout = QHBoxLayout(self.model_size_container)
        ms_layout.setContentsMargins(0, 0, 0, 0)
        self.model_size_combo = QComboBox()
        self.model_size_combo.addItem("Nano (10MB) - Fastest", "nano")
        self.model_size_combo.addItem("Small (40MB) - Balanced", "small")
        self.model_size_combo.addItem("Medium (90MB) - Best", "medium")
        self.model_size_combo.setCurrentIndex(2)
        self.model_size_combo.setFixedWidth(240)
        ms_layout.addWidget(FormRow("Model Size:", self.model_size_combo))
        card_model.addWidget(self.model_size_container)
        
        conf_row = QHBoxLayout()
        conf_row.addWidget(QLabel("Confidence:"))
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.0, 1.0)
        self.confidence_spin.setValue(0.6)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setFixedWidth(80)
        conf_row.addWidget(self.confidence_spin)
        conf_hint = QLabel("(0.5-0.7 recommended)")
        conf_hint.setProperty("role", "muted")
        conf_row.addWidget(conf_hint)
        conf_row.addStretch()
        card_model.addLayout(conf_row)
        layout.addWidget(card_model)
        
        # GPU card
        card_gpu = CardWidget("GPU Acceleration")
        self.use_gpu_check = QCheckBox("Enable GPU Acceleration (CUDA)")
        self.use_gpu_check.setChecked(True)
        card_gpu.addWidget(self.use_gpu_check)
        gpu_hint = QLabel("3-4x faster with compatible NVIDIA GPU. Auto-fallback to CPU if unavailable.")
        gpu_hint.setProperty("role", "accent")
        card_gpu.addWidget(gpu_hint)
        layout.addWidget(card_gpu)

        stage3_footer = StageActionFooter("Run Masking")
        stage3_footer.primary_button.clicked.connect(self.run_stage_3_only)
        stage3_footer.validate_button.clicked.connect(lambda: self._validate_stage_config(4))
        layout.addWidget(stage3_footer)
        
        layout.addStretch()
        return self._scroll_wrap(page)
    
    # ========================================================================
    # PAGE 5: 3D RECONSTRUCTION
    # ========================================================================
    def _create_stage4_page(self) -> QScrollArea:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(16)
        
        layout.addWidget(StageHeader(
            "3D Reconstruction",
            "Reconstruct camera poses and 3D structure. Required for Gaussian Splatting and photogrammetry."
        ))

        layout.addWidget(StageSummaryStrip(
            "Reconstruction",
            "Select reconstruction workflow and quality preset, then resolve dependency warnings before running."
        ))
        
        # Reconstruction Workflow card
        card_method = CardWidget("Reconstruction Workflow")
        
        self.alignment_mode_group = QButtonGroup(self)
        
        # Perspective Reconstruction (was Mode B — the default/recommended)
        self.mode_rig_sfm_radio = QRadioButton("Perspective Reconstruction (Recommended)")
        self.mode_rig_sfm_radio.setChecked(True)
        self.mode_rig_sfm_radio.setToolTip(
            "Split equirect → perspectives, then COLMAP GPU SfM\n"
            "Works for RealityScan, Metashape, COLMAP, Lichtfeld Studio, any 3DGS trainer"
        )
        self.alignment_mode_group.addButton(self.mode_rig_sfm_radio, 0)
        card_method.addWidget(self.mode_rig_sfm_radio)
        desc_b = QLabel("  COLMAP GPU on perspective images | Universal output")
        desc_b.setProperty("role", "indentedMuted")
        card_method.addWidget(desc_b)
        
        card_method.addSpacing(4)
        
        # Panorama SfM (was Mode A)
        self.mode_sphere_sfm_radio = QRadioButton("Panorama SfM (Equirectangular)")
        self.mode_sphere_sfm_radio.setToolTip("Direct spherical matching on equirectangular images | SPHERE camera model")
        self.alignment_mode_group.addButton(self.mode_sphere_sfm_radio, 1)
        card_method.addWidget(self.mode_sphere_sfm_radio)
        desc_a = QLabel("  SphereSfM on equirect images (no perspective split required)")
        desc_a.setProperty("role", "indentedMuted")
        card_method.addWidget(desc_a)
        
        # SphereSfM status
        self.spheresfm_status_label = QLabel("  Checking SphereSfM...")
        self.spheresfm_status_label.setObjectName("sphereStatus")
        self.spheresfm_status_label.setProperty("status", "idle")
        card_method.addWidget(self.spheresfm_status_label)
        QTimer.singleShot(300, self._check_spheresfm_status)
        
        # Connect mode changes
        self.mode_sphere_sfm_radio.toggled.connect(self._on_alignment_mode_changed)
        self.mode_rig_sfm_radio.toggled.connect(self._on_alignment_mode_changed)
        
        # Hidden legacy checkbox for config compat
        self.use_rig_sfm_check = QCheckBox()
        self.use_rig_sfm_check.setVisible(False)
        self.use_rig_sfm_check.setChecked(True)
        
        # Legacy: hidden pose_transfer radio (kept for config compat, never shown)
        self.mode_pose_transfer_radio = QRadioButton()
        self.mode_pose_transfer_radio.setVisible(False)
        self.alignment_mode_group.addButton(self.mode_pose_transfer_radio, 2)
        
        layout.addWidget(card_method)
        
        # Legacy: hidden pose_transfer config (kept for attribute compat)
        self.pose_transfer_config_group = CardWidget("Virtual Camera Configuration")
        self.pose_transfer_config_group.setVisible(False)
        layout.addWidget(self.pose_transfer_config_group)
        
        # Quality card
        card_quality = CardWidget("Reconstruction Quality")
        self.colmap_quality_combo = QComboBox()
        self.colmap_quality_combo.addItems(["Draft (Fast)", "Medium (Balanced)", "High (Best Quality)"])
        self.colmap_quality_combo.setCurrentIndex(1)
        self.colmap_quality_combo.setFixedWidth(240)
        card_quality.addWidget(FormRow("Quality Preset:", self.colmap_quality_combo))
        layout.addWidget(card_quality)
        
        # Performance card
        card_perf = CardWidget("Performance")
        self.use_gpu_colmap_check = QCheckBox("Use GPU Acceleration (CUDA) for feature extraction")
        self.use_gpu_colmap_check.setChecked(True)
        card_perf.addWidget(self.use_gpu_colmap_check)

        self.mapping_backend_combo = QComboBox()
        self.mapping_backend_combo.addItem("COLMAP Incremental", "colmap")
        self.mapping_backend_combo.addItem("COLMAP Global Mapper (integrated GLOMAP)", "glomap")
        self.mapping_backend_combo.setCurrentIndex(1)
        card_perf.addWidget(FormRow("Mapping Backend:", self.mapping_backend_combo))

        self.enable_hloc_fallback_check = QCheckBox("Enable HLOC fallback (ALIKED + LightGlue)")
        self.enable_hloc_fallback_check.setChecked(True)
        card_perf.addWidget(self.enable_hloc_fallback_check)

        self.prefer_colmap_learned_check = QCheckBox("Prefer COLMAP learned extractor (ALIKED)")
        self.prefer_colmap_learned_check.setChecked(False)
        self.prefer_colmap_learned_check.setToolTip(
            "If enabled, try COLMAP ALIKED first. If disabled, large jobs may prefer HLOC learned fallback automatically."
        )
        card_perf.addWidget(self.prefer_colmap_learned_check)

        self.require_learned_pipeline_check = QCheckBox("All-or-fail: require learned pipeline")
        self.require_learned_pipeline_check.setChecked(False)
        self.require_learned_pipeline_check.setToolTip(
            "If enabled, reconstruction fails unless ALIKED+LightGlue (or HLOC fallback) is actually used."
        )
        card_perf.addWidget(self.require_learned_pipeline_check)

        self.recon_paths_label = QLabel("")
        self.recon_paths_label.setWordWrap(True)
        card_perf.addWidget(FormRow("Dependency Paths:", self.recon_paths_label))

        open_settings_btn = QPushButton("Manage in Settings")
        open_settings_btn.setFixedWidth(160)
        open_settings_btn.clicked.connect(self.open_settings)
        card_perf.addWidget(open_settings_btn)
        layout.addWidget(card_perf)
        
        # Export card
        card_export = CardWidget("Export Options")
        self.export_lichtfeld_check = QCheckBox("Export to LichtFeld Studio Format")
        self.export_lichtfeld_check.setChecked(True)
        self.export_lichtfeld_check.setToolTip("transforms.json + pointcloud.ply + images/ + masks/")
        card_export.addWidget(self.export_lichtfeld_check)

        self.export_realityscan_check = QCheckBox("Export to RealityScan Format")
        self.export_realityscan_check.setChecked(False)
        self.export_realityscan_check.setToolTip(
            "Without COLMAP: exports images/ (+ optional *_mask.png). "
            "With COLMAP: also exports sparse/ and optional database.db"
        )
        card_export.addWidget(self.export_realityscan_check)
        
        self.export_include_masks_check = QCheckBox("Include Masks in Export")
        self.export_include_masks_check.setChecked(True)
        card_export.addWidget(self.export_include_masks_check)

        self.export_sidecar_check = QCheckBox("Export XMP sidecar files (optional)")
        self.export_sidecar_check.setChecked(False)
        self.export_sidecar_check.setToolTip("Create .xmp files beside aligned images with camera pose metadata")
        card_export.addWidget(self.export_sidecar_check)
        layout.addWidget(card_export)
        
        # Output info
        self.output_info_label = QLabel(
            "Output: <output_dir>/reconstruction/sparse/0/"
        )
        self.output_info_label.setProperty("role", "muted")
        self.output_info_label.setWordWrap(True)
        layout.addWidget(self.output_info_label)

        stage4_footer = StageActionFooter("Run Reconstruction")
        stage4_footer.primary_button.clicked.connect(self.run_stage_4_only)
        stage4_footer.validate_button.clicked.connect(lambda: self._validate_stage_config(5))
        layout.addWidget(stage4_footer)
        
        layout.addStretch()
        return self._scroll_wrap(page)
    
    # ========================================================================
    # PAGE 6: TRAINING
    # ========================================================================
    def _create_stage5_page(self) -> QScrollArea:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(16)
        
        layout.addWidget(StageHeader(
            "Training",
            "Launch Gaussian Splatting training with LichtFeld Studio using the generated COLMAP model."
        ))

        layout.addWidget(StageSummaryStrip(
            "Training",
            "Configure optional training launch and executable path after reconstruction output is ready."
        ))
        
        card = CardWidget("Training Target")
        self.train_lichtfeld_check = QCheckBox("Launch Lichtfeld Studio Training")
        self.train_lichtfeld_check.setChecked(False)
        card.addWidget(self.train_lichtfeld_check)
        
        path_row = QHBoxLayout()
        self.lichtfeld_path_edit = QLineEdit()
        self.lichtfeld_path_edit.setPlaceholderText("Auto-detect or browse...")
        path_row.addWidget(self.lichtfeld_path_edit, stretch=1)
        lf_browse = QPushButton("Browse...")
        lf_browse.setFixedWidth(80)
        lf_browse.clicked.connect(self.browse_lichtfeld_path)
        path_row.addWidget(lf_browse)
        path_row_widget = QWidget()
        path_row_widget.setLayout(path_row)
        card.addWidget(FormRow("Lichtfeld Path:", path_row_widget))
        
        note = QLabel("Lichtfeld Studio will be launched with the generated COLMAP model for 3DGS training.")
        note.setProperty("role", "muted")
        note.setWordWrap(True)
        card.addWidget(note)
        
        layout.addWidget(card)

        stage5_footer = StageActionFooter("Run Training")
        stage5_footer.primary_button.clicked.connect(self.run_stage_5_only)
        stage5_footer.validate_button.clicked.connect(lambda: self._validate_stage_config(6))
        layout.addWidget(stage5_footer)

        layout.addStretch()
        return self._scroll_wrap(page)
    
    # ========================================================================
    # LOG PANEL
    # ========================================================================
    def _create_log_panel(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("rightLogPanel")
        panel.setMinimumWidth(240)
        panel.setMaximumWidth(340)
        self.log_panel = panel
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)
        
        header = QLabel("Processing Log")
        header.setObjectName("logPanelTitle")
        layout.addWidget(header)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        self.log_text.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.log_text)
        
        return panel

    def _on_log_toggle_clicked(self, checked: bool):
        self._user_log_visible = checked
        self._update_responsive_layout()

    def _toggle_theme(self):
        next_theme = "light" if self._resolved_theme == "dark" else "dark"
        self.settings.set_theme(next_theme)
        self.apply_theme()

    def _set_log_panel_visible(self, visible: bool):
        if visible == self._effective_log_visible:
            return

        self._effective_log_visible = visible
        if hasattr(self, "log_panel"):
            self.log_panel.setVisible(visible)

        if hasattr(self, "main_splitter"):
            if visible:
                self.main_splitter.setSizes([196, 1080, 280])
            else:
                self.main_splitter.setSizes([196, 1180, 0])

    def _update_responsive_layout(self):
        compact_controls = self.width() < 1700
        auto_hide_log = self.width() < 1460

        self._set_control_bar_compact(compact_controls)

        if auto_hide_log:
            if hasattr(self, "log_toggle_button"):
                self.log_toggle_button.setEnabled(False)
                self.log_toggle_button.blockSignals(True)
                self.log_toggle_button.setChecked(False)
                self.log_toggle_button.blockSignals(False)
                self.log_toggle_button.setToolTip("Log auto-hidden on smaller window widths")
            self._set_log_panel_visible(False)
            return

        if hasattr(self, "log_toggle_button"):
            self.log_toggle_button.setEnabled(True)
            self.log_toggle_button.blockSignals(True)
            self.log_toggle_button.setChecked(self._user_log_visible)
            self.log_toggle_button.blockSignals(False)
            self.log_toggle_button.setToolTip("Show or hide the right log panel")

        self._set_log_panel_visible(self._user_log_visible)

    def _set_control_bar_compact(self, compact: bool):
        if compact == self._control_bar_compact:
            return

        self._control_bar_compact = compact

        # Reduce secondary chrome first in compact mode
        for widget in (
            self.input_label,
            self.output_label,
            self.control_sep1,
            self.control_sep2,
            self.progress_bar,
            self.status_label,
        ):
            widget.setVisible(not compact)

        if compact:
            self.control_bar.setFixedHeight(56)
            self.input_file_edit.setMinimumWidth(90)
            self.output_dir_edit.setMinimumWidth(90)
            self.input_browse_btn.setFixedSize(30, 28)
            self.output_browse_btn.setFixedSize(30, 28)
            self.start_button.setFixedSize(80, 28)
            self.pause_button.setFixedSize(68, 28)
            self.stop_button.setFixedSize(68, 28)
            self.log_toggle_button.setFixedSize(30, 28)
            self.theme_toggle_button.setFixedSize(30, 28)
        else:
            self.control_bar.setFixedHeight(64)
            self.input_file_edit.setMinimumWidth(130)
            self.output_dir_edit.setMinimumWidth(120)
            self.input_browse_btn.setFixedSize(34, 30)
            self.output_browse_btn.setFixedSize(34, 30)
            self.start_button.setFixedSize(90, 30)
            self.pause_button.setFixedSize(78, 30)
            self.stop_button.setFixedSize(78, 30)
            self.log_toggle_button.setFixedSize(34, 30)
            self.theme_toggle_button.setFixedSize(34, 30)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_responsive_layout()
    
    # ========================================================================
    # HELPERS
    # ========================================================================
    def _open_stage_page(self, page_index: int):
        """Navigate to a specific stage page and sync sidebar selection."""
        self.page_stack.setCurrentIndex(page_index)
        if 0 <= page_index < len(self.nav_buttons):
            self.nav_buttons[page_index].setChecked(True)

    def _validate_stage_config(self, stage_index: int):
        """Lightweight inline validation checks for each stage."""
        input_path = self.input_file_edit.text().strip()
        output_path = self.output_dir_edit.text().strip()
        input_obj = Path(input_path) if input_path else None

        if stage_index in (0, 1, 2, 3, 4, 5, 6):
            if not input_path:
                self.log_message("[WARN] Validation: Input path is not set.")
                self._set_control_status("Input path missing", "warn")
                return
            if not output_path:
                self.log_message("[WARN] Validation: Output path is not set.")
                self._set_control_status("Output path missing", "warn")
                return

        if stage_index == 1 and not self.full_video_check.isChecked():
            if self.end_time_spin.value() <= self.start_time_spin.value():
                self.log_message("[WARN] Extraction validation: End time must be greater than start time.")
                self._set_control_status("Invalid extraction time range", "warn")
                return

        if stage_index == 1 and input_obj is not None and input_obj.exists() and input_obj.is_dir():
            self.log_message("[WARN] Extraction validation: Stage 1 expects a video file (.insv/.mp4), not a folder.")
            self._set_control_status("Stage 1 requires a video file", "warn")
            return

        if stage_index in (2, 3) and self.skip_transform_check.isChecked():
            self.log_message("[INFO] Split validation: Transform is skipped (direct masking mode enabled).")
            self._set_control_status("Split skipped (direct mask)", "info")
            return

        if stage_index == 4 and not (
            self.persons_group.isChecked() or self.objects_group.isChecked() or self.animals_group.isChecked()
        ):
            self.log_message("[WARN] Masking validation: No masking category group is enabled.")
            self._set_control_status("No masking categories enabled", "warn")
            return

        if stage_index == 6 and self.train_lichtfeld_check.isChecked() and not self.lichtfeld_path_edit.text().strip():
            self.log_message("[WARN] Training validation: Lichtfeld path is required when training is enabled.")
            self._set_control_status("Lichtfeld path required", "warn")
            return

        self.log_message("[OK] Validation passed for current stage configuration.")
        self._set_control_status("Validation passed", "ok")

    def _set_control_status(self, text: str, level: str = "info"):
        """Update top control status with semantic color states."""
        if not hasattr(self, 'status_label'):
            return
        self.status_label.setText((text or "")[:40])
        self.status_label.setProperty("level", level)
        self._refresh_widget_style(self.status_label)

    def _refresh_widget_style(self, widget: QWidget):
        """Force Qt to re-apply dynamic property based styles."""
        widget.style().unpolish(widget)
        widget.style().polish(widget)

    def _scroll_wrap(self, widget: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(widget)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        return scroll
    
    def _add_camera_group(self, camera_count=8, pitch=0, fov=110, name=None):
        """Add a camera group row"""
        w = QWidget()
        w.setObjectName("cameraGroupRow")
        row = QHBoxLayout(w)
        row.setContentsMargins(10, 8, 10, 8)
        row.setSpacing(10)
        
        if name is None:
            name = f"Group {len(self.camera_group_widgets) + 1}"
        lbl = QLabel(f"{name}:")
        lbl.setFixedWidth(90)
        lbl.setProperty("role", "secondary")
        row.addWidget(lbl)
        
        cams_lbl = QLabel("Cams")
        cams_lbl.setProperty("role", "muted")
        row.addWidget(cams_lbl)
        cc_spin = QSpinBox()
        cc_spin.setRange(1, 12)
        cc_spin.setValue(camera_count)
        cc_spin.setFixedWidth(68)
        row.addWidget(cc_spin)
        
        pitch_lbl = QLabel("Pitch")
        pitch_lbl.setProperty("role", "muted")
        row.addWidget(pitch_lbl)
        p_spin = QSpinBox()
        p_spin.setRange(-90, 90)
        p_spin.setValue(pitch)
        p_spin.setFixedWidth(68)
        row.addWidget(p_spin)
        
        fov_lbl = QLabel("FOV")
        fov_lbl.setProperty("role", "muted")
        row.addWidget(fov_lbl)
        f_spin = QSpinBox()
        f_spin.setRange(30, 150)
        f_spin.setValue(fov)
        f_spin.setFixedWidth(68)
        row.addWidget(f_spin)
        
        rm_btn = QPushButton("−")
        rm_btn.setObjectName("dangerMiniButton")
        rm_btn.setToolTip("Remove this camera group")
        rm_btn.setFixedSize(28, 28)
        rm_btn.clicked.connect(lambda: self._remove_camera_group(w))
        row.addWidget(rm_btn)
        row.addStretch()
        
        data = {
            'widget': w, 'name_label': lbl,
            'camera_count': cc_spin, 'pitch': p_spin, 'fov': f_spin
        }
        self.camera_group_widgets.append(data)
        self.camera_groups_container_layout.addWidget(w)
    
    def _remove_camera_group(self, widget):
        for i, d in enumerate(self.camera_group_widgets):
            if d['widget'] == widget:
                self.camera_group_widgets.pop(i)
                break
        self.camera_groups_container_layout.removeWidget(widget)
        widget.deleteLater()
    
    def _generate_camera_positions(self) -> list:
        stage2_method = self.stage2_method_combo.currentData()
        if stage2_method == 'perspective':
            cameras = []
            for gd in self.camera_group_widgets:
                cc = gd['camera_count'].value()
                p = gd['pitch'].value()
                f = gd['fov'].value()
                for i in range(cc):
                    yaw = (360 / cc) * i
                    cameras.append({'yaw': yaw, 'pitch': p, 'roll': 0, 'fov': f})
            return cameras
        elif stage2_method == 'cubemap':
            return []
        else:
            return [{'yaw': (360/8)*i, 'pitch': 0, 'roll': 0, 'fov': 110} for i in range(8)]
    
    # ========================================================================
    # MENU BAR
    # ========================================================================
    def create_menu_bar(self):
        menubar = self.menuBar()
        
        file_menu = menubar.addMenu("&File")
        open_action = QAction("&Open File...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.browse_input_file)
        file_menu.addAction(open_action)
        open_output_action = QAction("Open &Output Folder", self)
        open_output_action.setShortcut("Ctrl+Shift+O")
        open_output_action.triggered.connect(self.open_output_folder)
        file_menu.addAction(open_output_action)

        file_menu.addSeparator()
        export_simple_action = QAction("Export RealityScan (&Simple: Images + Masks)", self)
        export_simple_action.triggered.connect(self.export_realityscan_simple)
        file_menu.addAction(export_simple_action)

        export_meta_action = QAction("Export RealityScan (&With COLMAP Metadata)", self)
        export_meta_action.triggered.connect(self.export_realityscan_with_metadata)
        file_menu.addAction(export_meta_action)

        file_menu.addSeparator()
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        config_menu = menubar.addMenu("&Configuration")
        save_action = QAction("Save Configuration...", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_configuration)
        config_menu.addAction(save_action)
        load_action = QAction("Load Configuration...", self)
        load_action.setShortcut("Ctrl+L")
        load_action.triggered.connect(self.load_configuration)
        config_menu.addAction(load_action)
        config_menu.addSeparator()
        manage_action = QAction("Manage Configurations...", self)
        manage_action.triggered.connect(self.manage_configurations)
        config_menu.addAction(manage_action)
        
        settings_menu = menubar.addMenu("&Settings")
        pref_action = QAction("&Preferences...", self)
        pref_action.setShortcut("Ctrl+,")
        pref_action.triggered.connect(self.open_settings)
        settings_menu.addAction(pref_action)
        settings_menu.addSeparator()
        detect_action = QAction("Detect SDK/FFmpeg Paths", self)
        detect_action.triggered.connect(self.detect_paths)
        settings_menu.addAction(detect_action)
        
        help_menu = menubar.addMenu("&Help")
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def _setup_shortcuts(self):
        """Keyboard shortcuts for core pipeline controls."""
        QShortcut(QKeySequence("Ctrl+Return"), self, activated=self.start_pipeline)
        QShortcut(QKeySequence("Ctrl+P"), self, activated=self.toggle_pause)
        QShortcut(QKeySequence("Ctrl+Shift+P"), self, activated=self.stop_pipeline)
        QShortcut(QKeySequence("F5"), self, activated=self._refresh_current_page)

    def _refresh_current_page(self):
        """Soft refresh hook for current page state feedback."""
        self.log_message("[INFO] UI refreshed.")
        self._update_overview_stage_summary()

    def _update_overview_stage_summary(self):
        """Update overview readiness summary from stage toggles."""
        if not hasattr(self, 'overview_stage_status_label'):
            return
        checks = [
            getattr(self, 'stage1_enable', None),
            getattr(self, 'stage2_enable', None),
            getattr(self, 'stage3_enable', None),
            getattr(self, 'stage4_enable', None),
            getattr(self, 'stage5_enable', None),
        ]
        enabled_count = sum(1 for check in checks if check is not None and check.isChecked())
        self.overview_stage_status_label.setText(
            f"Enabled steps: {enabled_count}/5 | Ready: {'Yes' if enabled_count > 0 else 'No'}"
        )
    
    # ========================================================================
    # DIALOG HANDLERS
    # ========================================================================
    def open_settings(self):
        dialog = SettingsDialog(self)
        dialog.settings_changed.connect(self.on_settings_changed)
        dialog.exec()
    
    def detect_paths(self):
        sdk = self.settings.auto_detect_sdk()
        ffmpeg = self.settings.auto_detect_ffmpeg()
        spheresfm = self.settings.auto_detect_spheresfm()
        colmap = self.settings.auto_detect_colmap()
        msg = "Path Detection Results:\n\n"
        if sdk:
            self.settings.set_sdk_path(sdk, auto_detected=True)
            msg += f"[OK] SDK: {sdk}\n"
        else:
            msg += "[X] SDK not found\n"
        if ffmpeg:
            self.settings.set_ffmpeg_path(ffmpeg, auto_detected=True)
            msg += f"[OK] FFmpeg: {ffmpeg}\n"
        else:
            msg += "[X] FFmpeg not found\n"
        if spheresfm:
            self.settings.set_spheresfm_path(spheresfm, auto_detected=True)
            msg += f"[OK] SphereSfM: {spheresfm}\n"
        else:
            msg += "[X] SphereSfM not found\n"
        if colmap:
            self.settings.set_colmap_gpu_path(colmap, auto_detected=True)
            msg += f"[OK] COLMAP GPU: {colmap}\n"
        else:
            msg += "[X] COLMAP GPU not found\n"
        QMessageBox.information(self, "Path Detection", msg)
        self.on_settings_changed()
    
    def on_settings_changed(self):
        self.apply_theme()
        sdk_path = self.settings.get_sdk_path()
        ffmpeg_path = self.settings.get_ffmpeg_path()
        spheresfm_path = self.settings.get_spheresfm_path()
        colmap_path = self.settings.get_colmap_gpu_path()
        colmap_version = "Unknown"
        if colmap_path:
            try:
                colmap_info = self.settings.get_colmap_info(colmap_path)
                colmap_version = colmap_info.get('version', 'Unknown')
                logger.info(f"[Diagnostics] COLMAP executable: {colmap_info.get('path', colmap_path)}")
                logger.info(f"[Diagnostics] COLMAP version: {colmap_version}")
            except Exception as colmap_info_error:
                logger.warning(f"[Diagnostics] Failed to query COLMAP version: {colmap_info_error}")
        parts = []
        if sdk_path:
            parts.append(f"SDK: {sdk_path.name}")
        if ffmpeg_path:
            parts.append(f"FFmpeg: {ffmpeg_path.name}")
        if spheresfm_path:
            parts.append(f"SphereSfM: {spheresfm_path.name}")
        if colmap_path:
            parts.append(f"COLMAP GPU: {colmap_path.name}")
        self.statusBar().showMessage(" | ".join(parts) if parts else "Dependencies not configured")

        if hasattr(self, 'recon_paths_label'):
            spheresfm_text = str(spheresfm_path) if spheresfm_path else "Not configured"
            colmap_text = str(colmap_path) if colmap_path else "Not configured"
            self.recon_paths_label.setText(
                f"SphereSfM: {spheresfm_text}\n"
                f"COLMAP GPU: {colmap_text}\n"
                f"COLMAP Version: {colmap_version}"
            )
    
    def show_about(self):
        QMessageBox.about(self, f"About {APP_NAME}",
            f"{APP_NAME} v{APP_VERSION}\n\n"
            "Unified photogrammetry preprocessing pipeline.\n"
            "Extract -> Split -> Mask -> Reconstruct -> Train\n\n"
            "Copyright 2026 | MIT License"
        )
    
    def save_configuration(self):
        config = self.get_current_config()
        dialog = SaveConfigDialog(config, self)
        dialog.exec()
    
    def load_configuration(self):
        dialog = ConfigManagementDialog(self)
        dialog.config_loaded.connect(self.apply_loaded_config)
        dialog.exec()
    
    def manage_configurations(self):
        dialog = ConfigManagementDialog(self)
        dialog.config_loaded.connect(self.apply_loaded_config)
        dialog.exec()
    
    # ========================================================================
    # CONFIG GET/SET
    # ========================================================================
    def get_current_config(self) -> dict:
        stage2_method = self.stage2_method_combo.currentData()
        if stage2_method == 'cubemap':
            stage2_format = self.cubemap_format_combo.currentData()
        else:
            stage2_format = self.stage2_format_combo.currentData()

        return {
            'input_file': self.input_file_edit.text(),
            'output_dir': self.output_dir_edit.text(),
            'stage1_enabled': self.stage1_enable.isChecked(),
            'fps_interval': self.fps_spin.value(),
            'extraction_method': self.extraction_method_combo.currentData(),
            'sdk_quality': self.sdk_quality_combo.currentData(),
            'sdk_resolution': self.sdk_resolution_combo.currentData(),
            'output_format': self.output_format_combo.currentData(),
            'stage2_enabled': self.stage2_enable.isChecked(),
            'transform_type': self.stage2_method_combo.currentData(),
            'stage2_format': stage2_format,
            'stage2_format_perspective': self.stage2_format_combo.currentData(),
            'stage2_format_cubemap': self.cubemap_format_combo.currentData(),
            'output_width': self.stage2_width_spin.value(),
            'output_height': self.stage2_height_spin.value(),
            'cubemap_tile_width': self.cubemap_tile_width_spin.value(),
            'cubemap_tile_height': self.cubemap_tile_height_spin.value(),
            'cubemap_fov': 90,
            'skip_transform': self.skip_transform_check.isChecked(),
            'stage3_enabled': self.stage3_enable.isChecked(),
            'model_size': self.model_size_combo.currentData(),
            'confidence_threshold': self.confidence_spin.value(),
            'use_gpu': self.use_gpu_check.isChecked(),
            'masking_categories': {
                'persons': self.persons_group.isChecked(),
                'personal_objects': self.objects_group.isChecked(),
                'animals': self.animals_group.isChecked()
            },
            'mask_target': 'equirect' if self.mask_equirect_radio.isChecked() else 'split',
            'stage4_enabled': self.stage4_enable.isChecked(),
            'use_rig_sfm': self.use_rig_sfm_check.isChecked(),
            'alignment_mode': self._get_alignment_mode(),
            'use_gpu_colmap': self.use_gpu_colmap_check.isChecked(),
            'colmap_quality': self.colmap_quality_combo.currentIndex(),
            'use_lightglue_aliked': True,
            'prefer_colmap_learned': self.prefer_colmap_learned_check.isChecked(),
            'enable_hloc_fallback': self.enable_hloc_fallback_check.isChecked(),
            'require_learned_pipeline': self.require_learned_pipeline_check.isChecked(),
            'reuse_colmap_database': True,
            'lichtfeld_fix_rotation': True,
            'spheresfm_path': str(self.settings.get_spheresfm_path()) if self.settings.get_spheresfm_path() else None,
            'colmap_path': str(self.settings.get_colmap_gpu_path()) if self.settings.get_colmap_gpu_path() else None,
            'sphere_alignment_path': str(self.settings.get_spheresfm_path()) if self.settings.get_spheresfm_path() else None,
            'mapping_backend': self.mapping_backend_combo.currentData(),
            'export_lichtfeld': self.export_lichtfeld_check.isChecked(),
            'export_realityscan': self.export_realityscan_check.isChecked(),
            'export_include_masks': self.export_include_masks_check.isChecked(),
            'export_sidecars': self.export_sidecar_check.isChecked(),
            'stage5_enabled': self.stage5_enable.isChecked(),
            'train_lighting': self.train_lichtfeld_check.isChecked(),
            'lichtfeld_path': self.lichtfeld_path_edit.text(),
            # SDK Media Processing options (colour sliders, stabilization toggles)
            'sdk_options': self.stage1_media_panel.get_sdk_options() if hasattr(self, 'stage1_media_panel') else {},
        }
    
    def _get_alignment_mode(self) -> str:
        if self.mode_rig_sfm_radio.isChecked():
            return 'perspective_reconstruction'
        elif self.mode_sphere_sfm_radio.isChecked():
            return 'panorama_sfm'
        return 'perspective_reconstruction'
    
    def apply_loaded_config(self, config: dict):
        try:
            if 'input_file' in config:
                self.input_file_edit.setText(config['input_file'])
            if 'output_dir' in config:
                self.output_dir_edit.setText(config['output_dir'])
            if 'stage1_enabled' in config:
                self.stage1_enable.setChecked(config['stage1_enabled'])
            if 'fps_interval' in config:
                self.fps_spin.setValue(config['fps_interval'])
            if 'extraction_method' in config:
                methods = list(EXTRACTION_METHODS.keys())
                if config['extraction_method'] in methods:
                    self.extraction_method_combo.setCurrentIndex(methods.index(config['extraction_method']))
            if 'sdk_quality' in config:
                idx = self.sdk_quality_combo.findData(config['sdk_quality'])
                if idx >= 0:
                    self.sdk_quality_combo.setCurrentIndex(idx)
            if 'sdk_resolution' in config:
                idx = self.sdk_resolution_combo.findData(config['sdk_resolution'])
                if idx >= 0:
                    self.sdk_resolution_combo.setCurrentIndex(idx)
            if 'sdk_options' in config and hasattr(self, 'stage1_media_panel'):
                self.stage1_media_panel.set_values(config.get('sdk_options') or {})
            if 'output_format' in config:
                idx = self.output_format_combo.findData(config['output_format'])
                if idx >= 0:
                    self.output_format_combo.setCurrentIndex(idx)
            if 'stage2_enabled' in config:
                self.stage2_enable.setChecked(config['stage2_enabled'])
            if 'transform_type' in config:
                idx = self.stage2_method_combo.findData(config['transform_type'])
                if idx >= 0:
                    self.stage2_method_combo.setCurrentIndex(idx)
            if 'output_width' in config:
                self.stage2_width_spin.setValue(config['output_width'])
            if 'output_height' in config:
                self.stage2_height_spin.setValue(config['output_height'])
            if 'stage2_format' in config:
                idx_p = self.stage2_format_combo.findData(config['stage2_format'])
                if idx_p >= 0:
                    self.stage2_format_combo.setCurrentIndex(idx_p)
                idx_c = self.cubemap_format_combo.findData(config['stage2_format'])
                if idx_c >= 0:
                    self.cubemap_format_combo.setCurrentIndex(idx_c)
            if 'stage2_format_perspective' in config:
                idx = self.stage2_format_combo.findData(config['stage2_format_perspective'])
                if idx >= 0:
                    self.stage2_format_combo.setCurrentIndex(idx)
            if 'stage2_format_cubemap' in config:
                idx = self.cubemap_format_combo.findData(config['stage2_format_cubemap'])
                if idx >= 0:
                    self.cubemap_format_combo.setCurrentIndex(idx)
            if 'skip_transform' in config:
                self.skip_transform_check.setChecked(config['skip_transform'])
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
            if 'mask_target' in config:
                if config['mask_target'] == 'equirect':
                    self.mask_equirect_radio.setChecked(True)
                else:
                    self.mask_split_radio.setChecked(True)
            if 'stage4_enabled' in config:
                self.stage4_enable.setChecked(config['stage4_enabled'])
            if 'alignment_mode' in config:
                m = config['alignment_mode']
                if m in ('sphere_sfm', 'panorama_sfm'):
                    self.mode_sphere_sfm_radio.setChecked(True)
                else:
                    # 'rig_sfm', 'pose_transfer', 'perspective_reconstruction' all → Perspective
                    self.mode_rig_sfm_radio.setChecked(True)
            if 'use_gpu_colmap' in config:
                self.use_gpu_colmap_check.setChecked(config['use_gpu_colmap'])
            if 'colmap_quality' in config:
                self.colmap_quality_combo.setCurrentIndex(config['colmap_quality'])
            if 'sphere_alignment_path' in config and config['sphere_alignment_path']:
                try:
                    self.settings.set_spheresfm_path(Path(config['sphere_alignment_path']), auto_detected=False)
                except Exception:
                    pass
            if 'spheresfm_path' in config and config['spheresfm_path']:
                try:
                    self.settings.set_spheresfm_path(Path(config['spheresfm_path']), auto_detected=False)
                except Exception:
                    pass
            if 'colmap_path' in config and config['colmap_path']:
                try:
                    self.settings.set_colmap_gpu_path(Path(config['colmap_path']), auto_detected=False)
                except Exception:
                    pass
            if 'glomap_path' in config and config['glomap_path']:
                try:
                    self.settings.set_colmap_gpu_path(Path(config['glomap_path']), auto_detected=False)
                except Exception:
                    pass
            if 'mapping_backend' in config:
                backend_idx = self.mapping_backend_combo.findData(config['mapping_backend'])
                if backend_idx >= 0:
                    self.mapping_backend_combo.setCurrentIndex(backend_idx)
            if 'enable_hloc_fallback' in config:
                self.enable_hloc_fallback_check.setChecked(config['enable_hloc_fallback'])
            if 'prefer_colmap_learned' in config:
                self.prefer_colmap_learned_check.setChecked(config['prefer_colmap_learned'])
            if 'require_learned_pipeline' in config:
                self.require_learned_pipeline_check.setChecked(config['require_learned_pipeline'])
            if 'export_lichtfeld' in config:
                self.export_lichtfeld_check.setChecked(config['export_lichtfeld'])
            if 'export_realityscan' in config:
                self.export_realityscan_check.setChecked(config['export_realityscan'])
            if 'export_include_masks' in config:
                self.export_include_masks_check.setChecked(config['export_include_masks'])
            if 'export_sidecars' in config:
                self.export_sidecar_check.setChecked(config['export_sidecars'])
            if 'stage5_enabled' in config:
                self.stage5_enable.setChecked(config['stage5_enabled'])
            if 'train_lighting' in config:
                self.train_lichtfeld_check.setChecked(config['train_lighting'])
            if 'lichtfeld_path' in config:
                self.lichtfeld_path_edit.setText(config['lichtfeld_path'])

            self.on_settings_changed()
            
            QMessageBox.information(self, "Config Loaded", "Configuration applied successfully.")
        except Exception as e:
            logger.error(f"Failed to apply config: {e}")
            QMessageBox.critical(self, "Error", f"Failed to apply config: {e}")
    
    # ========================================================================
    # EVENT HANDLERS
    # ========================================================================
    def on_extraction_method_changed(self, index: int):
        method = self.extraction_method_combo.currentData()
        is_sdk = method in ['sdk', 'sdk_stitching']
        self.sdk_quality_widget.setVisible(is_sdk)
        self.sdk_res_widget.setVisible(is_sdk)
        if hasattr(self, 'stage1_media_card'):
            self.stage1_media_card.setVisible(is_sdk)
    
    def on_skip_transform_toggled(self, checked: bool):
        if hasattr(self, 'stage2_output_group'):
            self.stage2_output_group.setEnabled(not checked)
        if hasattr(self, 'stage2_perspective_params_group'):
            self.stage2_perspective_params_group.setEnabled(not checked)
        if hasattr(self, 'run_stage2_btn'):
            self.run_stage2_btn.setEnabled(not checked)
        self.log_message("[SKIP] Direct Mask Mode" if checked else "[INFO] Perspective split enabled")
    
    def on_stage2_method_changed(self, index: int):
        method = self.stage2_method_combo.currentData()
        if method == 'perspective':
            self.page_stack.setCurrentIndex(2)
            self.nav_buttons[2].setChecked(True)
        elif method == 'cubemap':
            self.page_stack.setCurrentIndex(3)
            self.nav_buttons[3].setChecked(True)
    
    def on_cubemap_type_changed(self, index: int):
        t = self.cubemap_type_combo.currentData()
        self.tile_8_controls_widget.setVisible(t == '8-tile')
    
    def on_masking_engine_changed(self, index: int):
        engine = self.masking_engine_combo.currentData()
        if engine == "sam_vitb":
            self.model_size_container.setVisible(False)
            self.engine_description_label.setText("Superior segmentation quality | Best edge precision")
            self.confidence_spin.setEnabled(False)
        elif engine == "hybrid":
            self.model_size_container.setVisible(False)
            self.engine_description_label.setText("YOLO detection + SAM segmentation | Pixel-perfect edges")
            self.confidence_spin.setEnabled(True)
        else:
            self.model_size_container.setVisible(True)
            self.confidence_spin.setEnabled(True)
            if engine == "yolo_onnx":
                self.engine_description_label.setText("NMS-free inference | 3-4x faster | CUDA accelerated")
            else:
                self.engine_description_label.setText("Full-featured YOLO | PyTorch backend")
    
    def _on_alignment_mode_changed(self, checked: bool = True):
        if self.mode_rig_sfm_radio.isChecked():
            info = "Perspective Reconstruction: COLMAP GPU on perspective images"
        elif self.mode_sphere_sfm_radio.isChecked():
            info = "Panorama SfM: SphereSfM on equirectangular images"
        else:
            info = "Perspective Reconstruction (default)"
        self.output_info_label.setText(f"Output: <output_dir>/reconstruction/sparse/0/\n{info}")
    
    def toggle_time_range(self, checked: bool):
        self.start_time_spin.setEnabled(not checked)
        self.end_time_spin.setEnabled(not checked)
    
    def _check_spheresfm_status(self):
        try:
            from src.premium.sphere_sfm_integration import verify_spheresfm_installation
            status = verify_spheresfm_installation()
            if status['installed']:
                self.spheresfm_status_label.setText(f"  SphereSfM available ({status.get('version', 'Unknown')})")
                self.spheresfm_status_label.setProperty("status", "ok")
                self._refresh_widget_style(self.spheresfm_status_label)
                self.mode_sphere_sfm_radio.setEnabled(True)
            else:
                self.spheresfm_status_label.setText(f"  SphereSfM not available: {status.get('error', 'Unknown')}")
                self.spheresfm_status_label.setProperty("status", "error")
                self._refresh_widget_style(self.spheresfm_status_label)
                self.mode_sphere_sfm_radio.setEnabled(False)
                self.mode_rig_sfm_radio.setChecked(True)
        except Exception as e:
            self.spheresfm_status_label.setText(f"  SphereSfM error: {e}")
            self.spheresfm_status_label.setProperty("status", "error")
            self._refresh_widget_style(self.spheresfm_status_label)
            self.mode_sphere_sfm_radio.setEnabled(False)
            self.mode_rig_sfm_radio.setChecked(True)
    
    # ========================================================================
    # FILE BROWSING
    # ========================================================================
    def browse_input_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select Input Video", "",
            "Video Files (*.insv *.mp4 *.mov);;All Files (*.*)"
        )
        if filename:
            self.input_file_edit.setText(filename)
            self.analyze_video_file()
    
    def browse_output_dir(self):
        dirname = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dirname:
            self.output_dir_edit.setText(dirname)

    def open_output_folder(self):
        output_dir = self.output_dir_edit.text().strip()
        if not output_dir:
            QMessageBox.warning(self, "Output Required", "Please configure the output directory first.")
            return

        output_path = Path(output_dir)
        if not output_path.exists():
            QMessageBox.warning(self, "Not Found", f"Output folder does not exist:\n{output_path}")
            return

        try:
            if os.name == 'nt':
                os.startfile(str(output_path))
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', str(output_path)])
            else:
                subprocess.Popen(['xdg-open', str(output_path)])
            self.log_message(f"Opened output folder: {output_path}")
        except Exception as e:
            QMessageBox.warning(self, "Open Folder Failed", f"Could not open output folder:\n{e}")

    def export_realityscan_simple(self):
        """Export RealityScan package as a simple flat folder (images + masks only)."""
        output_dir = self.output_dir_edit.text().strip()
        if not output_dir:
            QMessageBox.warning(self, "Output Required", "Please configure the output directory first.")
            return

        output_root = Path(output_dir)
        if not output_root.exists():
            QMessageBox.warning(self, "Not Found", f"Output folder does not exist:\n{output_root}")
            return

        from src.pipeline.batch_orchestrator import PipelineWorker

        worker = PipelineWorker({
            'output_dir': str(output_root),
            'export_include_masks': self.export_include_masks_check.isChecked(),
        })
        result = worker._execute_realityscan_export_only()

        if result.get('success'):
            export_path = Path(result.get('realityscan_export', output_root / 'realityscan_export'))
            self.log_message(f"[OK] RealityScan simple export complete: {export_path}")
            self._set_control_status("RealityScan export complete", "ok")
            QMessageBox.information(
                self,
                "RealityScan Export Complete",
                f"Simple export created successfully:\n{export_path}\n\n"
                "Contains images and masks in the same folder."
            )
        else:
            error = result.get('error', 'Unknown error')
            self.log_message(f"[FAIL] RealityScan simple export failed: {error}")
            self._set_control_status("RealityScan export failed", "error")
            QMessageBox.warning(self, "RealityScan Export Failed", f"Simple export failed:\n{error}")

    def export_realityscan_with_metadata(self):
        """Export RealityScan package with COLMAP metadata (sparse model), plus images/masks."""
        output_dir = self.output_dir_edit.text().strip()
        if not output_dir:
            QMessageBox.warning(self, "Output Required", "Please configure the output directory first.")
            return

        output_root = Path(output_dir)
        if not output_root.exists():
            QMessageBox.warning(self, "Not Found", f"Output folder does not exist:\n{output_root}")
            return

        sparse_root = output_root / 'reconstruction' / 'sparse'
        colmap_dir = None

        candidates = [sparse_root / '0', sparse_root / '1', sparse_root]
        for candidate in candidates:
            if candidate.exists() and ((candidate / 'images.txt').exists() or (candidate / 'images.bin').exists()):
                colmap_dir = candidate
                break

        if colmap_dir is None:
            QMessageBox.warning(
                self,
                "COLMAP Metadata Not Found",
                "No COLMAP sparse model was found in reconstruction output.\n"
                "Run reconstruction first, or use 'Export RealityScan (Simple)'."
            )
            return

        images_dir = output_root / 'reconstruction' / 'images'
        if not images_dir.exists():
            images_dir = output_root / 'perspective_views'
        if not images_dir.exists():
            QMessageBox.warning(self, "Images Not Found", "Could not find perspective images to export.")
            return

        masks_dir = output_root / 'masks'
        export_masks_dir = str(masks_dir) if (self.export_include_masks_check.isChecked() and masks_dir.exists()) else None

        from src.premium.pose_transfer_integration import export_for_realityscan

        export_dir = output_root / 'realityscan_export_with_metadata'
        database_path = sparse_root / 'database.db'
        ok = export_for_realityscan(
            colmap_dir=str(colmap_dir),
            images_dir=str(images_dir),
            masks_dir=export_masks_dir,
            output_dir=str(export_dir),
            database_path=str(database_path) if database_path.exists() else None,
            flat_folder=True,
        )

        if ok:
            self.log_message(f"[OK] RealityScan metadata export complete: {export_dir}")
            self._set_control_status("RealityScan metadata export complete", "ok")
            QMessageBox.information(
                self,
                "RealityScan Export Complete",
                f"Metadata export created successfully:\n{export_dir}\n\n"
                "Contains images+masks in one folder plus sparse metadata."
            )
        else:
            self.log_message("[FAIL] RealityScan metadata export failed")
            self._set_control_status("RealityScan metadata export failed", "error")
            QMessageBox.warning(self, "RealityScan Export Failed", "Metadata export failed.")
    
    def browse_lichtfeld_path(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Lichtfeld Studio", "C:\\Program Files",
            "Executables (*.exe);;All (*.*)"
        )
        if path:
            self.lichtfeld_path_edit.setText(path)

    def browse_colmap_path(self):
        QMessageBox.information(
            self,
            "Dependency Paths",
            "Configure COLMAP path in Settings > Paths & Detection."
        )
        self.open_settings()

    def browse_glomap_path(self):
        QMessageBox.information(
            self,
            "Dependency Paths",
            "Configure GloMAP path in Settings > Paths & Detection."
        )
        self.open_settings()
    
    def analyze_video_file(self):
        from src.extraction import FrameExtractor
        input_file = self.input_file_edit.text()
        if not input_file or not Path(input_file).exists():
            return
        if Path(input_file).is_dir():
            self.file_metadata_label.setText(
                "Folder input detected. Stage 1 analysis expects a video file (.insv/.mp4)."
            )
            self.file_metadata_label.setProperty("state", "warn")
            self._refresh_widget_style(self.file_metadata_label)
            self.log_message("Input is a folder. Skip video analysis and use Split/Mask stages.")
            return
        
        extractor = FrameExtractor()
        info = extractor.get_video_info(input_file)
        
        if info.get('success'):
            self.file_metadata_label.setText(
                f"<b>Type:</b> {info.get('file_type_desc', 'Unknown')} | "
                f"<b>Duration:</b> {info.get('duration_formatted', 'N/A')} | "
                f"<b>Resolution:</b> {info.get('resolution', 'N/A')} | "
                f"<b>FPS:</b> {info.get('fps', 0):.2f} | "
                f"<b>Frames:</b> {info.get('frame_count', 0):,} | "
                f"<b>Camera:</b> {info.get('camera_model', 'Unknown')} | "
                f"<b>Size:</b> {info.get('file_size_mb', 0):.1f} MB"
            )
            self.file_metadata_label.setProperty("state", "ok")
            self._refresh_widget_style(self.file_metadata_label)
            duration = info.get('duration', 0)
            self.end_time_spin.setMaximum(duration)
            self.end_time_spin.setValue(duration)
            
            w = info.get('width', 0)
            h = info.get('height', 0)
            if w > 0 and h > 0:
                tw = ((w * 3 // 4 + 64) // 128) * 128
                th = ((h * 3 // 4 + 64) // 128) * 128
                self.cubemap_tile_width_spin.setValue(tw)
                self.cubemap_tile_height_spin.setValue(th)
            self.log_message(f"Analyzed: {Path(input_file).name}")
        else:
            self.file_metadata_label.setText(f"Error: {info.get('error', 'Unknown')}")
            self.file_metadata_label.setProperty("state", "error")
            self._refresh_widget_style(self.file_metadata_label)
    
    # ========================================================================
    # PIPELINE EXECUTION
    # ========================================================================
    def start_pipeline(self):
        if not hasattr(self, '_auto_advance_enabled') or not self._auto_advance_enabled:
            self._auto_advance_enabled = False
        
        input_file = self.input_file_edit.text()
        output_dir = self.output_dir_edit.text()
        
        if not input_file:
            QMessageBox.warning(self, "Input Required", "Please select an input file.")
            self._set_control_status("Input required", "warn")
            return
        if not output_dir:
            QMessageBox.warning(self, "Output Required", "Please select an output directory.")
            self._set_control_status("Output required", "warn")
            return
        if not Path(input_file).exists():
            QMessageBox.warning(self, "Not Found", f"Input file not found:\n{input_file}")
            self._set_control_status("Input file not found", "error")
            return
        if self.stage1_enable.isChecked() and Path(input_file).is_dir():
            QMessageBox.warning(
                self,
                "Invalid Input for Extraction",
                "Stage 1 (Frame Extraction) expects a video file (.insv/.mp4), not a folder.\n"
                "Either select a video file or disable Stage 1 and run Split/Mask on existing images."
            )
            self._set_control_status("Stage 1 requires a video file", "warn")
            return
        
        stage2_method = self.stage2_method_combo.currentData()
        
        self.pipeline_config = {
            'input_file': input_file,
            'output_dir': output_dir,
            'enable_stage1': self.stage1_enable.isChecked(),
            'skip_transform': self.skip_transform_check.isChecked(),
            'enable_stage2': self.stage2_enable.isChecked() and not self.skip_transform_check.isChecked(),
            'enable_stage3': self.stage3_enable.isChecked(),
            'use_rig_sfm': self.stage4_enable.isChecked() and self.use_rig_sfm_check.isChecked(),
            'train_lighting': self.stage5_enable.isChecked() and self.train_lichtfeld_check.isChecked(),
            'lichtfeld_path': self.lichtfeld_path_edit.text() or None,
            'export_realityscan': self.export_realityscan_check.isChecked(),
            'export_include_masks': self.export_include_masks_check.isChecked(),
            'fps': self.fps_spin.value(),
            'extraction_method': self.extraction_method_combo.currentData(),
            'start_time': 0.0 if self.full_video_check.isChecked() else self.start_time_spin.value(),
            'end_time': None if self.full_video_check.isChecked() else self.end_time_spin.value(),
            'sdk_quality': self.sdk_quality_combo.currentData(),
            'sdk_resolution': self.sdk_resolution_combo.currentData(),
            'output_format': self.output_format_combo.currentData(),
            'transform_type': stage2_method,
            'camera_config': {'cameras': self._generate_camera_positions()},
            'sdk_options': self.stage1_media_panel.get_sdk_options() if hasattr(self, 'stage1_media_panel') else {},
        }
        
        # Split method-specific params
        if stage2_method == 'perspective':
            camera_groups = []
            for gd in self.camera_group_widgets:
                camera_groups.append({
                    'camera_count': gd['camera_count'].value(),
                    'pitch': gd['pitch'].value(),
                    'fov': gd['fov'].value()
                })
            self.pipeline_config.update({
                'output_width': self.stage2_width_spin.value(),
                'output_height': self.stage2_height_spin.value(),
                'stage2_format': self.stage2_format_combo.currentData(),
                'perspective_params': {'camera_groups': camera_groups}
            })
        elif stage2_method == 'cubemap':
            tw = self.cubemap_tile_width_spin.value()
            th = self.cubemap_tile_height_spin.value()
            ct = self.cubemap_type_combo.currentData()
            self.pipeline_config.update({
                'output_width': tw,
                'output_height': th,
                'stage2_format': self.cubemap_format_combo.currentData(),
                'cubemap_params': {
                    'cubemap_type': ct, 'tile_width': tw, 'tile_height': th,
                    'fov': 90, 'layout': 'separate'
                }
            })
        
        # Masking classes
        if self.stage3_enable.isChecked():
            person_classes = [0] if self.persons_group.isChecked() and self.person_check.isChecked() else []
            
            object_classes = []
            if self.objects_group.isChecked():
                for chk, cls_id in [
                    (self.backpack_check, 24), (self.umbrella_check, 25),
                    (self.handbag_check, 26), (self.tie_check, 27),
                    (self.suitcase_check, 28), (self.cell_phone_check, 67)
                ]:
                    if chk.isChecked():
                        object_classes.append(cls_id)
            
            animal_classes = []
            if self.animals_group.isChecked():
                for chk, cls_id in [
                    (self.bird_check, 14), (self.cat_check, 15),
                    (self.dog_check, 16), (self.horse_check, 17)
                ]:
                    if chk.isChecked():
                        animal_classes.append(cls_id)
                animal_classes.extend([18, 19, 20, 21, 22, 23])
            
            self.pipeline_config.update({
                'masking_engine': self.masking_engine_combo.currentData(),
                'model_size': self.model_size_combo.currentData(),
                'confidence_threshold': self.confidence_spin.value(),
                'use_gpu': True,
                'masking_categories': {
                    'persons': len(person_classes) > 0,
                    'personal_objects': len(object_classes) > 0,
                    'animals': len(animal_classes) > 0
                },
                'masking_classes': {
                    'persons': person_classes,
                    'personal_objects': object_classes,
                    'animals': animal_classes
                }
            })
        
        # Reconstruction config
        if self.stage4_enable.isChecked():
            self.pipeline_config.update({
                'alignment_mode': self._get_alignment_mode(),
                'use_gpu_colmap': self.use_gpu_colmap_check.isChecked(),
                'colmap_quality': self.colmap_quality_combo.currentIndex(),
                'use_lightglue_aliked': True,
                'prefer_colmap_learned': self.prefer_colmap_learned_check.isChecked(),
                'enable_hloc_fallback': self.enable_hloc_fallback_check.isChecked(),
                'require_learned_pipeline': self.require_learned_pipeline_check.isChecked(),
                'reuse_colmap_database': True,
                'lichtfeld_fix_rotation': True,
                'spheresfm_path': str(self.settings.get_spheresfm_path()) if self.settings.get_spheresfm_path() else None,
                'colmap_path': str(self.settings.get_colmap_gpu_path()) if self.settings.get_colmap_gpu_path() else None,
                'sphere_alignment_path': str(self.settings.get_spheresfm_path()) if self.settings.get_spheresfm_path() else None,
                'mapping_backend': self.mapping_backend_combo.currentData(),
                'export_lichtfeld': self.export_lichtfeld_check.isChecked(),
                'export_sidecars': self.export_sidecar_check.isChecked(),
            })
        
        # UI state
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.pause_button.setEnabled(True)
        self._is_paused = False
        self.pause_button.setText("  Pause")
        
        self.log_text.clear()
        self.log_message("Starting pipeline...")
        self._set_control_status("Starting pipeline", "running")
        
        # Inject any stage-specific overrides set before this call
        # (e.g. run_stage_3_only sets _pending_stage3_input before calling us)
        if hasattr(self, '_pending_stage3_input') and self._pending_stage3_input:
            self.pipeline_config['stage3_input_dir'] = self._pending_stage3_input
            del self._pending_stage3_input

        self.orchestrator.run_pipeline(
            config=self.pipeline_config,
            progress_callback=self.on_progress,
            stage_complete_callback=self.on_stage_complete,
            finished_callback=self.on_finished,
            error_callback=self.on_error
        )
    
    def stop_pipeline(self):
        self.orchestrator.cancel()
        self.log_message("Pipeline stopped by user")
        self._set_control_status("Stopped", "warn")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.pause_button.setEnabled(False)
    
    def toggle_pause(self):
        if self._is_paused:
            self.orchestrator.resume()
            self.pause_button.setText("  Pause")
            self.log_message("Pipeline resumed")
            self._set_control_status("Running", "running")
            self._is_paused = False
        else:
            self.orchestrator.pause()
            self.pause_button.setText("  Resume")
            self.log_message("Pipeline paused")
            self._set_control_status("Paused", "warn")
            self._is_paused = True
    
    # ========================================================================
    # STAGE-ONLY RUNNERS
    # ========================================================================
    def run_stage_1_only(self):
        self.log_message("Running extraction only")
        self._auto_advance_enabled = True
        s2, s3 = self.stage2_enable.isChecked(), self.stage3_enable.isChecked()
        self.stage2_enable.setChecked(False)
        self.stage3_enable.setChecked(False)
        self.start_pipeline()
        self.stage2_enable.setChecked(s2)
        self.stage3_enable.setChecked(s3)
    
    def run_stage_2_only(self):
        self.log_message("Running split only")
        output_dir = self.output_dir_edit.text()
        if not output_dir:
            QMessageBox.warning(self, "Missing", "Configure output directory first")
            return
        
        from src.pipeline.batch_orchestrator import PipelineWorker
        worker = PipelineWorker({})
        folder = worker.discover_stage_input_folder(stage=2, output_dir=output_dir)
        if not folder:
            folder_str = QFileDialog.getExistingDirectory(self, "Select Extraction Output", str(Path(output_dir)))
            if not folder_str:
                return
            folder = Path(folder_str)
        
        images = []
        for ext in ['*.png', '*.PNG', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG']:
            images.extend(folder.glob(ext))
        if not images:
            QMessageBox.warning(self, "No Images", f"No images in: {folder}")
            return
        
        self.log_message(f"Found {len(images)} equirectangular images")
        self._auto_advance_enabled = True
        s1, s3 = self.stage1_enable.isChecked(), self.stage3_enable.isChecked()
        self.stage1_enable.setChecked(False)
        self.stage3_enable.setChecked(False)
        self.pipeline_config['stage2_input_dir'] = str(folder)
        self.start_pipeline()
        self.stage1_enable.setChecked(s1)
        self.stage3_enable.setChecked(s3)
    
    def run_stage_3_only(self):
        self.log_message("Running masking only")
        output_dir = self.output_dir_edit.text()
        if not output_dir:
            QMessageBox.warning(self, "Missing", "Configure output directory first")
            return

        from src.pipeline.batch_orchestrator import PipelineWorker
        worker = PipelineWorker({})
        folder = worker.discover_stage_input_folder(stage=3, output_dir=output_dir)
        if not folder:
            folder_str = QFileDialog.getExistingDirectory(
                self, "Select images to mask", str(Path(output_dir))
            )
            if not folder_str:
                return
            folder = Path(folder_str)

        images = []
        for ext in ['*.png', '*.PNG', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG']:
            images.extend(folder.glob(ext))
        if not images:
            QMessageBox.warning(self, "No Images", f"No images found in:\n{folder}")
            return

        self.log_message(f"Found {len(images)} images to mask in: {folder.name}")
        # Store the resolved folder so start_pipeline() can inject it after
        # rebuilding pipeline_config (it overwrites the whole dict from scratch).
        self._pending_stage3_input = str(folder)
        self._auto_advance_enabled = True
        s1, s2 = self.stage1_enable.isChecked(), self.stage2_enable.isChecked()
        self.stage1_enable.setChecked(False)
        self.stage2_enable.setChecked(False)
        self.start_pipeline()
        self.stage1_enable.setChecked(s1)
        self.stage2_enable.setChecked(s2)
    
    def run_stage_4_only(self):
        self.log_message("Running reconstruction only")
        output_dir = self.output_dir_edit.text()
        if not output_dir:
            QMessageBox.warning(self, "Missing", "Configure output directory first")
            return
        
        input_dir = Path(output_dir) / 'extracted_frames'
        if not input_dir.exists():
            from src.pipeline.batch_orchestrator import PipelineWorker
            worker = PipelineWorker({})
            input_dir = worker.discover_stage_input_folder(stage=2, output_dir=output_dir)
        
        if not input_dir or not input_dir.exists():
            folder = QFileDialog.getExistingDirectory(self, "Select Equirect Frames", str(Path(output_dir)))
            if not folder:
                return
            input_dir = Path(folder)
        
        self.log_message(f"Using input: {input_dir}")
        self._auto_advance_enabled = True
        orig = self.stage4_enable.isChecked()
        self.stage4_enable.setChecked(True)
        self.start_pipeline()
        self.stage4_enable.setChecked(orig)
    
    def run_stage_5_only(self):
        self.log_message("Running training only")
        output_dir = self.output_dir_edit.text()
        if not output_dir:
            QMessageBox.warning(self, "Missing", "Configure output directory first")
            return
        orig = self.stage5_enable.isChecked()
        self.stage5_enable.setChecked(True)
        self.start_pipeline()
        self.stage5_enable.setChecked(orig)
    
    # ========================================================================
    # CALLBACKS
    # ========================================================================
    def on_progress(self, current: int, total: int, message: str):
        if total > 0:
            self.progress_bar.setValue(int((current / total) * 100))
        self._set_control_status(message, "running")
        self.log_message(message)
    
    def on_stage_complete(self, stage_number: int, results: dict):
        if results.get('success'):
            self.log_message(f"[OK] Stage {stage_number} complete")
            if self._auto_advance_enabled:
                next_map = {
                    1: (self.stage2_enable, self.run_stage_2_only),
                    2: (self.stage3_enable, self.run_stage_3_only),
                    3: (self.stage4_enable, self.run_stage_4_only),
                    4: (self.stage5_enable, self.run_stage_5_only),
                }
                if stage_number in next_map:
                    check, runner = next_map[stage_number]
                    if check.isChecked():
                        self.log_message(f"[OK] Auto-advancing to Stage {stage_number + 1}...")
                        QTimer.singleShot(500, runner)
                    else:
                        self.log_message("[OK] All requested stages complete!")
                        self._auto_advance_enabled = False
                else:
                    self.log_message("[OK] All requested stages complete!")
                    self._auto_advance_enabled = False
        else:
            self.log_message(f"[FAIL] Stage {stage_number}: {results.get('error')}")
            self._auto_advance_enabled = False
    
    def on_finished(self, results: dict):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.pause_button.setEnabled(False)
        self.progress_bar.setValue(100)
        if results.get('success'):
            self.log_message("Pipeline complete!")
            self._set_control_status("Completed", "ok")
            QMessageBox.information(self, "Success", "Pipeline completed successfully!")
        else:
            self.log_message(f"Pipeline failed: {results.get('error')}")
            self._set_control_status("Failed", "error")
            QMessageBox.warning(self, "Failed", f"Pipeline failed:\n{results.get('error')}")
    
    def on_error(self, error_message: str):
        self.log_message(f"Error: {error_message}")
        self._set_control_status("Error", "error")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.pause_button.setEnabled(False)
    
    def log_message(self, message: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{ts}] {message}")
    
    # ========================================================================
    # THEME
    # ========================================================================
    def apply_theme(self):
        mode = self.settings.get_theme()
        if mode == "system":
            app = QApplication.instance()
            is_dark = bool(app and app.styleHints().colorScheme() == Qt.ColorScheme.Dark)
            mode = "dark" if is_dark else "light"
        if mode not in ("dark", "light"):
            mode = "dark"
        self._resolved_theme = mode
        self.setStyleSheet(build_theme_stylesheet(mode))
        if hasattr(self, "theme_toggle_button"):
            self.theme_toggle_button.blockSignals(True)
            self.theme_toggle_button.setChecked(mode == "dark")
            self.theme_toggle_button.setText("☾" if mode == "dark" else "☀")
            self.theme_toggle_button.blockSignals(False)
