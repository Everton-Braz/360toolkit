"""
360FrameTools - Main Window (Full-Screen UI/UX Rewrite)
Modern PyQt6 interface with sidebar navigation, full-screen layout,
and polished dark theme for professional photogrammetry workflow.
"""

import sys
import os
import logging
import shlex
import importlib
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
from PyQt6.QtCore import Qt, QTimer, QSize, QPropertyAnimation, QEasingCurve, pyqtSignal, QEvent
from PyQt6.QtGui import QFont, QAction, QIcon, QPainter, QColor, QPen, QPixmap, QShortcut, QKeySequence, QPalette, QStandardItem

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
from src.utils.runtime_backends import has_bundled_onnx_runtime, has_usable_torch_runtime, is_usable_torch_module

logger = logging.getLogger(__name__)

DEFAULT_SPHERESFM_FEATURE_FLAGS = (
    "--ImageReader.single_camera 1 --SiftExtraction.max_num_orientations 2 "
    "--SiftExtraction.peak_threshold 0.00667 --SiftExtraction.edge_threshold 10.0"
)

DEFAULT_SPHERESFM_MATCHER_FLAGS = (
    "--SequentialMatching.quadratic_overlap 1 --SequentialMatching.loop_detection 0 "
    "--SiftMatching.max_ratio 0.8 --SiftMatching.max_distance 0.7 --SiftMatching.cross_check 1 "
    "--SiftMatching.max_error 4.0 --SiftMatching.confidence 0.999 "
    "--SiftMatching.max_num_trials 10000 --SiftMatching.min_inlier_ratio 0.25"
)

DEFAULT_SPHERESFM_MAPPER_FLAGS = (
    "--Mapper.ba_refine_focal_length 0 --Mapper.ba_refine_principal_point 0 "
    "--Mapper.ba_refine_extra_params 0 --Mapper.init_min_num_inliers 100 "
    "--Mapper.init_num_trials 200 --Mapper.init_max_error 4 --Mapper.init_max_forward_motion 0.95 "
    "--Mapper.init_min_tri_angle 16 --Mapper.abs_pose_min_num_inliers 50 "
    "--Mapper.abs_pose_max_error 8 --Mapper.abs_pose_min_inlier_ratio 0.25 --Mapper.max_reg_trials 3 "
    "--Mapper.tri_min_angle 1.5 --Mapper.tri_max_transitivity 1 --Mapper.tri_ignore_two_view_tracks 1 "
    "--Mapper.filter_max_reproj_error 4 --Mapper.filter_min_tri_angle 1.5 --Mapper.multiple_models 1"
)


def _normalize_image_format(value: str | None, default: str = "png") -> str:
    normalized = str(value or default).strip().lower()
    aliases = {
        'jpeg': 'jpg',
        'jpe': 'jpg',
        'tif': 'tiff',
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in {'png', 'jpg', 'tiff'} else default


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


class _CheckableComboBox(QComboBox):
    """QComboBox where every item has a checkbox.

    The popup stays open while the user toggles items.
    Use ``addCheckItem()`` to populate, ``checkedTexts()`` to read.
    """

    def __init__(self, placeholder: str = "Select…", parent=None):
        super().__init__(parent)
        # Editable line used as read-only display label
        self.setEditable(True)
        le = self.lineEdit()
        le.setReadOnly(True)
        le.setPlaceholderText(placeholder)
        le.installEventFilter(self)   # forward click → open/close popup
        self._placeholder = placeholder
        # Intercept mouse release on the list view to toggle without closing
        self.view().viewport().installEventFilter(self)

    # ── public ────────────────────────────────────────────────────────────

    def addCheckItem(self, text: str, checked: bool = True) -> QStandardItem:
        """Append one checkable item. Returns the QStandardItem."""
        item = QStandardItem(text)
        item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        self.model().appendRow(item)
        self._update_display()
        return item

    def checkedTexts(self) -> list:
        """Return list of text strings for all checked items."""
        return [
            self.model().item(i).text()
            for i in range(self.model().rowCount())
            if self.model().item(i).checkState() == Qt.CheckState.Checked
        ]

    def isAnyChecked(self) -> bool:
        return bool(self.checkedTexts())

    def setAllChecked(self, checked: bool):
        """Check or uncheck every item."""
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for i in range(self.model().rowCount()):
            self.model().item(i).setCheckState(state)
        self._update_display()

    # ── event handling ─────────────────────────────────────────────────────

    def eventFilter(self, obj, event):
        # Item clicked inside the dropdown → toggle + stay open
        if obj is self.view().viewport() and event.type() == QEvent.Type.MouseButtonRelease:
            idx = self.view().indexAt(event.pos())
            if idx.isValid():
                item = self.model().item(idx.row())
                if item:
                    new = (Qt.CheckState.Unchecked
                           if item.checkState() == Qt.CheckState.Checked
                           else Qt.CheckState.Checked)
                    item.setCheckState(new)
                    self._update_display()
            return True   # <- suppress the default close-on-click behaviour
        # Click on the text field → toggle popup visibility
        if obj is self.lineEdit() and event.type() == QEvent.Type.MouseButtonPress:
            if self.view().isVisible():
                self.hidePopup()
            else:
                self.showPopup()
            return True
        return super().eventFilter(obj, event)

    # ── internal ───────────────────────────────────────────────────────────

    def _update_display(self):
        checked = self.checkedTexts()
        total = self.model().rowCount()
        if not checked:
            txt = "None"
        elif len(checked) == total:
            txt = "All"
        else:
            txt = ", ".join(checked)
        le = self.lineEdit()
        if le:
            le.setText(txt)


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
        import importlib.util
        import logging
        gpu_logger = logging.getLogger(__name__)
        try:
            if has_bundled_onnx_runtime():
                self.gpu_status_label.setText("  GPU: ONNX packaged, checked on use")
                self.gpu_status_label.setProperty("status", "warn")
                self._refresh_widget_style(self.gpu_status_label)
                return

            if importlib.util.find_spec('onnxruntime') is not None:
                try:
                    import onnxruntime as ort
                    providers = ort.get_available_providers()
                    gpu_logger.info("[GPU Detect] ONNX Runtime providers: %s", providers)
                    if 'CUDAExecutionProvider' in providers:
                        self.gpu_status_label.setText("  GPU: CUDA ready (ONNX/COLMAP)")
                        self.gpu_status_label.setProperty("status", "ok")
                        self._refresh_widget_style(self.gpu_status_label)
                        return
                except Exception as onnx_error:
                    gpu_logger.warning("[GPU Detect] ONNX Runtime unavailable: %s", onnx_error)

            if has_usable_torch_runtime():
                try:
                    import torch
                    if not is_usable_torch_module(torch):
                        raise ImportError("incomplete torch runtime")
                    gpu_logger.info("[GPU Detect] PyTorch %s, CUDA available: %s", torch.__version__, torch.cuda.is_available())
                    if torch.cuda.is_available():
                        name = torch.cuda.get_device_name(0)
                        self.gpu_status_label.setText(f"  GPU: {name}")
                        self.gpu_status_label.setProperty("status", "ok")
                        self._refresh_widget_style(self.gpu_status_label)
                        return
                except Exception as torch_error:
                    gpu_logger.warning("[GPU Detect] PyTorch probe unavailable: %s", torch_error)

            if Path(os.environ.get('WINDIR', 'C:/Windows')).joinpath('System32', 'nvcuda.dll').exists():
                self.gpu_status_label.setText("  GPU: NVIDIA driver detected")
                self.gpu_status_label.setProperty("status", "warn")
                self._refresh_widget_style(self.gpu_status_label)
                return

            gpu_logger.info("[GPU Detect] No CUDA-capable runtime backend available")
            self.gpu_status_label.setText("  GPU: CPU only")
            self.gpu_status_label.setProperty("status", "warn")
            self._refresh_widget_style(self.gpu_status_label)
        except Exception as e:
            gpu_logger.warning(f"[GPU Detect] Runtime backend probe failed: {e}")
            self.gpu_status_label.setText("  GPU: Backend probe failed")
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
        self.output_format_combo.addItem("JPEG (Compressed)", "jpg")
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
        # Auto-reset preview rotation when FlowState is enabled
        # (FlowState uses gyroscope to level the output — no manual rotation needed)
        self.stage1_media_panel.values_changed.connect(
            self._on_media_flowstate_changed
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
        self.stage2_format_combo.addItem("JPEG", "jpg")
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
        self.cubemap_format_combo.addItem("JPEG", "jpg")
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
            "Detect and mask persons, objects, and animals for photogrammetry."
        ))

        layout.addWidget(StageSummaryStrip(
            "Masking",
            "Choose detection categories, engine, and confidence, then validate before running."
        ))

        card_source = CardWidget("Input Source")
        self.mask_input_source_combo = QComboBox()
        self.mask_input_source_combo.addItem("Auto", "auto")
        self.mask_input_source_combo.addItem("Perspective Views", "perspective")
        self.mask_input_source_combo.addItem("Equirect / Extracted Frames", "equirect")
        card_source.addWidget(FormRow("Images:", self.mask_input_source_combo))

        auto_note = QLabel(
            "Auto uses perspective views when available and falls back to extracted frames. "
            "Masks are stored separately as masks_perspective or masks_equirect."
        )
        auto_note.setProperty("role", "secondary")
        auto_note.setWordWrap(True)
        card_source.addWidget(auto_note)
        layout.addWidget(card_source)

        # ── Detection Categories — compact dropdown card ────────────────────
        card_cats = CardWidget("Detection Categories")
        cats_grid = QGridLayout()
        cats_grid.setSpacing(10)
        cats_grid.setColumnMinimumWidth(0, 160)
        cats_grid.setColumnStretch(1, 1)

        # Row 0 — Persons
        self.persons_enable = QCheckBox("Persons")
        self.persons_enable.setChecked(True)
        self.persons_combo = _CheckableComboBox("Select…")
        self.persons_combo.addCheckItem("Person", True)
        cats_grid.addWidget(self.persons_enable, 0, 0)
        cats_grid.addWidget(self.persons_combo,  0, 1)

        # Row 1 — Personal Objects
        self.objects_enable = QCheckBox("Personal Objects")
        self.objects_enable.setChecked(True)
        self.objects_combo = _CheckableComboBox("Select…")
        for _item in ["Backpack", "Umbrella", "Handbag", "Tie", "Suitcase", "Cell Phone"]:
            self.objects_combo.addCheckItem(_item, True)
        cats_grid.addWidget(self.objects_enable, 1, 0)
        cats_grid.addWidget(self.objects_combo,  1, 1)

        # Row 2 — Animals
        self.animals_enable = QCheckBox("Animals")
        self.animals_enable.setChecked(False)
        self.animals_combo = _CheckableComboBox("Select…")
        for _item in ["Bird", "Cat", "Dog", "Horse", "Sheep", "Cow",
                      "Elephant", "Bear", "Zebra", "Giraffe"]:
            self.animals_combo.addCheckItem(_item, True)
        cats_grid.addWidget(self.animals_enable, 2, 0)
        cats_grid.addWidget(self.animals_combo,  2, 1)

        # Disable combos when enable checkbox is unchecked
        self.persons_enable.toggled.connect(self.persons_combo.setEnabled)
        self.objects_enable.toggled.connect(self.objects_combo.setEnabled)
        self.animals_enable.toggled.connect(self.animals_combo.setEnabled)
        self.animals_combo.setEnabled(False)   # animals off by default

        card_cats.addLayout(cats_grid)
        layout.addWidget(card_cats)

        # ── Engine & Quality — merged card ─────────────────────────────────
        card_model = CardWidget("Engine & Quality")

        self.masking_engine_combo = QComboBox()
        self.masking_engine_combo.addItem("YOLO (ONNX) - Fast", "yolo_onnx")
        self.masking_engine_combo.setCurrentIndex(0)
        self.masking_engine_combo.setMinimumWidth(280)
        self.masking_engine_combo.currentIndexChanged.connect(self.on_masking_engine_changed)
        card_model.addWidget(FormRow("Masking Engine:", self.masking_engine_combo))
        
        self.engine_description_label = QLabel("ONNX-first runtime | smallest packaged build | CUDA when available")
        self.engine_description_label.setProperty("role", "mutedSmall")
        card_model.addWidget(self.engine_description_label)
        self.masking_runtime_label = QLabel("Runtime: detecting ONNX / PyTorch backends...")
        self.masking_runtime_label.setProperty("role", "mutedSmall")
        card_model.addWidget(self.masking_runtime_label)
        
        # Model size
        self.model_size_container = QWidget()
        ms_layout = QHBoxLayout(self.model_size_container)
        ms_layout.setContentsMargins(0, 0, 0, 0)
        self.model_size_combo = QComboBox()
        self.model_size_combo.addItem("Nano (10MB) - Fastest", "nano")
        self.model_size_combo.addItem("Small (40MB) - Balanced", "small")
        self.model_size_combo.addItem("Medium (90MB) - Best", "medium")
        self.model_size_combo.setCurrentIndex(1)
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
        conf_hint = QLabel("(common range: 0.5-0.7)")
        conf_hint.setProperty("role", "muted")
        conf_row.addWidget(conf_hint)
        conf_row.addStretch()
        card_model.addLayout(conf_row)

        # GPU inline in same card
        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine); sep.setProperty("role", "divider")
        card_model.addWidget(sep)
        gpu_row = QHBoxLayout()
        self.use_gpu_check = QCheckBox("Enable GPU Acceleration (CUDA)")
        self.use_gpu_check.setChecked(True)
        self.use_gpu_check.toggled.connect(self._on_cuda_toggled)
        gpu_row.addWidget(self.use_gpu_check)
        self.gpu_hint_label = QLabel("3-4x faster with NVIDIA GPU. Auto-fallback to CPU.")
        self.gpu_hint_label.setProperty("role", "accent")
        gpu_row.addWidget(self.gpu_hint_label)
        gpu_row.addStretch()
        card_model.addLayout(gpu_row)
        layout.addWidget(card_model)

        stage3_footer = StageActionFooter("Run Masking")
        stage3_footer.primary_button.clicked.connect(self.run_stage_3_only)
        stage3_footer.validate_button.clicked.connect(lambda: self._validate_stage_config(4))
        layout.addWidget(stage3_footer)
        
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

        if stage_index in (2, 3) and not self.stage2_enable.isChecked():
            self.log_message("[INFO] Split validation: Splitting is disabled — masking targets extracted frames directly.")
            self._set_control_status("Split disabled, masking extracted frames", "info")
            return

        if stage_index == 4 and not (
            self.persons_enable.isChecked() or self.objects_enable.isChecked() or self.animals_enable.isChecked()
        ):
            self.log_message("[WARN] Masking validation: No masking category group is enabled.")
            self._set_control_status("No masking categories enabled", "warn")
            return

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
        self._detect_gpu()
        self._update_masking_runtime_status()
        self._update_reconstruction_backend_status()
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
            stage2_format = _normalize_image_format(self.cubemap_format_combo.currentData(), 'png')
        else:
            stage2_format = _normalize_image_format(self.stage2_format_combo.currentData(), 'png')

        return {
            'input_file': self.input_file_edit.text(),
            'output_dir': self.output_dir_edit.text(),
            'stage1_enabled': self.stage1_enable.isChecked(),
            'fps_interval': self.fps_spin.value(),
            'extraction_method': self.extraction_method_combo.currentData(),
            # Fisheye frame rotation: 0 | 90 | 180 | 270 (from preview rotate button)
            'frame_rotation': getattr(self.stage1_eq_preview, '_rotation', 0) if hasattr(self, 'stage1_eq_preview') else 0,
            'sdk_quality': self.sdk_quality_combo.currentData(),
            'sdk_resolution': self.sdk_resolution_combo.currentData(),
            'output_format': _normalize_image_format(self.output_format_combo.currentData(), 'jpg'),
            'stage2_enabled': self.stage2_enable.isChecked(),
            'transform_type': self.stage2_method_combo.currentData(),
            'stage2_format': stage2_format,
            'stage2_format_perspective': _normalize_image_format(self.stage2_format_combo.currentData(), 'png'),
            'stage2_format_cubemap': _normalize_image_format(self.cubemap_format_combo.currentData(), 'png'),
            'output_width': self.stage2_width_spin.value(),
            'output_height': self.stage2_height_spin.value(),
            'cubemap_tile_width': self.cubemap_tile_width_spin.value(),
            'cubemap_tile_height': self.cubemap_tile_height_spin.value(),
            'cubemap_fov': 90,
            'skip_transform': not self.stage2_enable.isChecked(),
            'stage3_enabled': self.stage3_enable.isChecked(),
            'model_size': self.model_size_combo.currentData(),
            'confidence_threshold': self.confidence_spin.value(),
            'use_gpu': self.use_gpu_check.isChecked(),
            'masking_categories': {
                'persons': self.persons_enable.isChecked(),
                'personal_objects': self.objects_enable.isChecked(),
                'animals': self.animals_enable.isChecked(),
            },
            'stage3_image_source': self._get_stage3_image_source(),
            'mask_target': self._legacy_mask_target_from_source(self._get_stage3_image_source()),
            'export_realityscan': self.export_realityscan_check.isChecked() if hasattr(self, 'export_realityscan_check') else False,
            'export_include_masks': self.export_include_masks_check.isChecked() if hasattr(self, 'export_include_masks_check') else True,
            'export_image_source': self._get_export_image_source(),
            'export_mask_source': self._get_export_mask_source(),
            'export_sidecars': self.export_sidecar_check.isChecked() if hasattr(self, 'export_sidecar_check') else False,
            # SDK Media Processing options (colour sliders, stabilization toggles)
            'sdk_options': self.stage1_media_panel.get_sdk_options() if hasattr(self, 'stage1_media_panel') else {},
        }
    
    def _get_stage3_image_source(self) -> str:
        if hasattr(self, 'mask_input_source_combo'):
            return self.mask_input_source_combo.currentData()
        return 'equirect' if not self.stage2_enable.isChecked() else 'perspective'

    def _get_export_image_source(self) -> str:
        if hasattr(self, 'export_image_source_combo'):
            return self.export_image_source_combo.currentData()
        return 'auto'

    def _get_export_mask_source(self) -> str:
        if hasattr(self, 'export_mask_source_combo'):
            return self.export_mask_source_combo.currentData()
        return 'auto'

    def _legacy_mask_target_from_source(self, source: str) -> str:
        if source == 'equirect':
            return 'equirect'
        return 'split'

    def _set_combo_data(self, combo: QComboBox, value: str):
        idx = combo.findData(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _directory_has_images(self, folder: Path) -> bool:
        if not folder or not folder.exists() or not folder.is_dir():
            return False
        for ext in ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp', '*.PNG', '*.JPG', '*.JPEG', '*.TIF', '*.TIFF', '*.BMP'):
            if any(folder.rglob(ext)):
                return True
        return False

    def _resolve_output_image_dir(
        self,
        output_root: Path,
        source: str,
        alignment_mode: str | None = None,
        reconstruction_dir: Path | None = None,
    ) -> Path | None:
        perspective_dir = output_root / 'perspective_views'
        equirect_dir = output_root / 'extracted_frames'

        if source == 'reconstruction':
            if reconstruction_dir and self._directory_has_images(reconstruction_dir):
                return reconstruction_dir
            source = 'auto'

        if source == 'perspective':
            return perspective_dir if self._directory_has_images(perspective_dir) else None
        if source == 'equirect':
            return equirect_dir if self._directory_has_images(equirect_dir) else None

        ordered = []
        if reconstruction_dir and self._directory_has_images(reconstruction_dir):
            ordered.append(reconstruction_dir)
        if alignment_mode == 'panorama_sfm':
            ordered.extend([equirect_dir, perspective_dir])
        else:
            ordered.extend([perspective_dir, equirect_dir])

        for folder in ordered:
            if self._directory_has_images(folder):
                return folder
        return None

    def _resolve_output_masks_dir(
        self,
        output_root: Path,
        source: str,
        image_source: str,
    ) -> Path | None:
        mask_dirs = {
            'perspective': output_root / 'masks_perspective',
            'equirect': output_root / 'masks_equirect',
            'custom': output_root / 'masks_custom',
        }
        legacy_dir = output_root / 'masks'

        def _existing(folder: Path | None) -> Path | None:
            if folder and folder.exists() and any(folder.rglob('*.png')):
                return folder
            return None

        if source in ('match_images', 'match_reconstruction'):
            source = image_source
        if source == 'none':
            return None
        if source in mask_dirs:
            return _existing(mask_dirs[source]) or _existing(legacy_dir)

        ordered = []
        if image_source in mask_dirs:
            ordered.append(mask_dirs[image_source])
        ordered.extend([legacy_dir, mask_dirs['perspective'], mask_dirs['equirect'], mask_dirs['custom']])
        for folder in ordered:
            existing = _existing(folder)
            if existing:
                return existing
        return None
    
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
                idx = self.output_format_combo.findData(_normalize_image_format(config['output_format'], 'jpg'))
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
                normalized_stage2 = _normalize_image_format(config['stage2_format'], 'png')
                idx_p = self.stage2_format_combo.findData(normalized_stage2)
                if idx_p >= 0:
                    self.stage2_format_combo.setCurrentIndex(idx_p)
                idx_c = self.cubemap_format_combo.findData(normalized_stage2)
                if idx_c >= 0:
                    self.cubemap_format_combo.setCurrentIndex(idx_c)
            if 'stage2_format_perspective' in config:
                idx = self.stage2_format_combo.findData(_normalize_image_format(config['stage2_format_perspective'], 'png'))
                if idx >= 0:
                    self.stage2_format_combo.setCurrentIndex(idx)
            if 'stage2_format_cubemap' in config:
                idx = self.cubemap_format_combo.findData(_normalize_image_format(config['stage2_format_cubemap'], 'png'))
                if idx >= 0:
                    self.cubemap_format_combo.setCurrentIndex(idx)
            if 'stage3_enabled' in config:
                self.stage3_enable.setChecked(config['stage3_enabled'])
            if 'stage3_image_source' in config and hasattr(self, 'mask_input_source_combo'):
                self._set_combo_data(self.mask_input_source_combo, config['stage3_image_source'])
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
                self.persons_enable.setChecked(cats.get('persons', True))
                self.objects_enable.setChecked(cats.get('personal_objects', True))
                self.animals_enable.setChecked(cats.get('animals', True))
            if 'export_realityscan' in config and hasattr(self, 'export_realityscan_check'):
                self.export_realityscan_check.setChecked(config['export_realityscan'])
            if 'export_include_masks' in config and hasattr(self, 'export_include_masks_check'):
                self.export_include_masks_check.setChecked(config['export_include_masks'])
            if 'export_image_source' in config and hasattr(self, 'export_image_source_combo'):
                self._set_combo_data(self.export_image_source_combo, config['export_image_source'])
            if 'export_mask_source' in config and hasattr(self, 'export_mask_source_combo'):
                self._set_combo_data(self.export_mask_source_combo, config['export_mask_source'])
            if 'export_sidecars' in config and hasattr(self, 'export_sidecar_check'):
                self.export_sidecar_check.setChecked(config['export_sidecars'])

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
        # Update preview to reflect the newly selected extraction mode
        if hasattr(self, 'stage1_eq_preview'):
            self.stage1_eq_preview.set_extraction_method(method)
    
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
        if engine == "yolo_onnx":
            self.model_size_container.setVisible(True)
            self.confidence_spin.setEnabled(True)
            self.engine_description_label.setText("ONNX-first runtime | smallest packaged build | CUDA when available")
        else:
            self.model_size_container.setVisible(True)
            self.confidence_spin.setEnabled(True)
            self.engine_description_label.setText("Masking runtime selected from legacy config")
        self._update_masking_runtime_status()

    def _on_cuda_toggled(self, enabled: bool):
        """Enable or disable CUDA for masking operations at runtime.

        Sets CUDA_VISIBLE_DEVICES so PyTorch workers spawned *after* this
        call respect the choice.  Already-running workers are unaffected.
        """
        if enabled:
            # Restore GPU — remove any suppression we set
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
            # Try to detect what GPU is actually available
            try:
                import torch
                if torch.cuda.is_available():
                    name = torch.cuda.get_device_name(0)
                    self.gpu_hint_label.setText(f"GPU active: {name}")
                    self.gpu_hint_label.setProperty("role", "accent")
                else:
                    self.gpu_hint_label.setText("CUDA not available on this machine — will use CPU")
                    self.gpu_hint_label.setProperty("role", "muted")
                    self.use_gpu_check.setChecked(False)
                    return
            except Exception:
                self.gpu_hint_label.setText("3-4x faster with NVIDIA GPU. Auto-fallback to CPU.")
                self.gpu_hint_label.setProperty("role", "accent")
            self.log_message("[GPU] CUDA enabled — GPU acceleration active for masking.")
        else:
            # Hide all CUDA devices from PyTorch processes
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            self.gpu_hint_label.setText("CPU mode (CUDA disabled by user)")
            self.gpu_hint_label.setProperty("role", "muted")
            self.log_message("[GPU] CUDA disabled — masking will run on CPU.")
        self._refresh_widget_style(self.gpu_hint_label)
    
    def _on_recon_tool_changed(self, index: int):
        self.recon_params_stack.setCurrentIndex(index)
        if index == 0:
            self.recon_tool_desc.setText(
                "Perspective reconstruction on split images using pycolmap, with external COLMAP CLI/GLOMAP when available."
            )
            self.output_info_label.setText(
                "Output: <output_dir>/reconstruction/sparse/0/"
            )
        else:
            self.recon_tool_desc.setText(
                "Panorama SfM on equirectangular images using SphereSfM."
            )
            self.output_info_label.setText(
                "Output: <output_dir>/reconstruction/sparse/0/  (SphereSfM SPHERE model)"
            )
        self._update_reconstruction_backend_status()
        self._update_recon_stack_height()

    def _update_masking_runtime_status(self):
        if not hasattr(self, 'masking_runtime_label'):
            return

        details = []
        try:
            if has_bundled_onnx_runtime():
                details.append('ONNX Runtime: packaged (checked on use)')
            elif importlib.util.find_spec('onnxruntime') is not None:
                import onnxruntime as ort
                providers = ort.get_available_providers()
                details.append('ONNX Runtime: CUDA ready' if 'CUDAExecutionProvider' in providers else 'ONNX Runtime: CPU only')
            else:
                details.append('ONNX Runtime: unavailable')
        except Exception as exc:
            details.append(f'ONNX Runtime: error ({exc})')

        details.append('PyTorch: installed' if has_usable_torch_runtime() else 'PyTorch: not bundled')
        self.masking_runtime_label.setText(' | '.join(details))

    def _update_reconstruction_backend_status(self):
        if not hasattr(self, 'recon_backend_label'):
            return

        details = []
        details.append('pycolmap: available' if importlib.util.find_spec('pycolmap') is not None else 'pycolmap: unavailable')

        colmap_path = self.settings.get_colmap_gpu_path()
        if colmap_path and colmap_path.exists():
            details.append(f'COLMAP CLI: {colmap_path.name}')
        else:
            details.append('COLMAP CLI: auto-download/system lookup')

        spheresfm_path = self.settings.get_spheresfm_path()
        if spheresfm_path and spheresfm_path.exists():
            details.append(f'SphereSfM: {spheresfm_path.name}')
        else:
            details.append('SphereSfM: auto-download/system lookup')

        self.recon_backend_label.setText('Backend: ' + ' | '.join(details))

    def _update_recon_stack_height(self):
        if not hasattr(self, 'recon_params_stack'):
            return
        current = self.recon_params_stack.currentWidget()
        if current is None:
            return
        self.recon_params_stack.setFixedHeight(current.sizeHint().height())

    def _parse_cli_flag_values(self, flags: str) -> dict[str, str]:
        if not flags or not flags.strip():
            return {}

        try:
            tokens = shlex.split(flags, posix=False)
        except ValueError:
            tokens = flags.split()

        values: dict[str, str] = {}
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if token.startswith("--"):
                value = "1"
                if index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
                    value = tokens[index + 1]
                    index += 1
                values[token] = value
            index += 1
        return values

    def _flag_enabled(self, values: dict[str, str], flag: str, default: bool) -> bool:
        raw_value = values.get(flag)
        if raw_value is None:
            return default
        return str(raw_value).strip().lower() not in {"0", "false", "no", "off"}

    def _format_flag_number(self, value) -> str:
        if isinstance(value, float):
            return f"{value:g}"
        return str(value)

    def _build_spheresfm_feature_flags(self) -> str:
        if hasattr(self, 'sphere_feature_flags_edit') and self.sphere_feature_flags_edit.text().strip():
            return self.sphere_feature_flags_edit.text().strip()
        parts = [
            "--ImageReader.single_camera", "1" if self.sphere_single_camera_check.isChecked() else "0",
            "--SiftExtraction.max_num_orientations", str(self.sphere_max_orientations_spin.value()),
            "--SiftExtraction.peak_threshold", self._format_flag_number(self.sphere_peak_threshold_spin.value()),
            "--SiftExtraction.edge_threshold", self._format_flag_number(self.sphere_edge_threshold_spin.value()),
        ]
        return " ".join(parts)

    def _build_spheresfm_matcher_flags(self) -> str:
        if hasattr(self, 'sphere_matcher_flags_edit') and self.sphere_matcher_flags_edit.text().strip():
            return self.sphere_matcher_flags_edit.text().strip()
        parts = [
            "--SequentialMatching.quadratic_overlap", "1" if self.sphere_quadratic_overlap_check.isChecked() else "0",
            "--SequentialMatching.loop_detection", "1" if self.sphere_loop_detection_check.isChecked() else "0",
            "--SiftMatching.max_ratio", self._format_flag_number(self.sphere_max_ratio_spin.value()),
            "--SiftMatching.max_distance", self._format_flag_number(self.sphere_max_distance_spin.value()),
            "--SiftMatching.cross_check", "1" if self.sphere_cross_check_check.isChecked() else "0",
            "--SiftMatching.max_error", self._format_flag_number(self.sphere_matcher_max_error_spin.value()),
            "--SiftMatching.confidence", self._format_flag_number(self.sphere_matcher_confidence_spin.value()),
            "--SiftMatching.max_num_trials", str(self.sphere_max_trials_spin.value()),
            "--SiftMatching.min_inlier_ratio", self._format_flag_number(self.sphere_min_inlier_ratio_spin.value()),
        ]
        return " ".join(parts)

    def _build_spheresfm_mapper_flags(self) -> str:
        if hasattr(self, 'sphere_mapper_flags_edit') and self.sphere_mapper_flags_edit.text().strip():
            return self.sphere_mapper_flags_edit.text().strip()
        parts = [
            "--Mapper.ba_refine_focal_length", "1" if self.sphere_refine_focal_check.isChecked() else "0",
            "--Mapper.ba_refine_principal_point", "1" if self.sphere_refine_principal_point_check.isChecked() else "0",
            "--Mapper.ba_refine_extra_params", "1" if self.sphere_refine_extra_params_check.isChecked() else "0",
            "--Mapper.init_min_num_inliers", str(self.sphere_init_min_inliers_spin.value()),
            "--Mapper.init_num_trials", str(self.sphere_init_trials_spin.value()),
            "--Mapper.init_max_error", self._format_flag_number(self.sphere_init_max_error_spin.value()),
            "--Mapper.init_max_forward_motion", self._format_flag_number(self.sphere_init_forward_motion_spin.value()),
            "--Mapper.init_min_tri_angle", self._format_flag_number(self.sphere_init_min_tri_angle_spin.value()),
            "--Mapper.abs_pose_min_num_inliers", str(self.sphere_abs_pose_min_inliers_spin.value()),
            "--Mapper.abs_pose_max_error", self._format_flag_number(self.sphere_abs_pose_max_error_spin.value()),
            "--Mapper.abs_pose_min_inlier_ratio", self._format_flag_number(self.sphere_abs_pose_min_inlier_ratio_spin.value()),
            "--Mapper.max_reg_trials", str(self.sphere_max_reg_trials_spin.value()),
            "--Mapper.tri_min_angle", self._format_flag_number(self.sphere_tri_min_angle_spin.value()),
            "--Mapper.tri_max_transitivity", str(self.sphere_tri_max_transitivity_spin.value()),
            "--Mapper.tri_ignore_two_view_tracks", "1" if self.sphere_ignore_two_view_tracks_check.isChecked() else "0",
            "--Mapper.filter_max_reproj_error", self._format_flag_number(self.sphere_filter_max_reproj_error_spin.value()),
            "--Mapper.filter_min_tri_angle", self._format_flag_number(self.sphere_filter_min_tri_angle_spin.value()),
            "--Mapper.multiple_models", "1" if self.sphere_multiple_models_check.isChecked() else "0",
        ]
        return " ".join(parts)

    def _apply_spheresfm_flag_config(self, sfm_params: dict):
        if hasattr(self, 'sphere_feature_flags_edit'):
            self.sphere_feature_flags_edit.setText(str(sfm_params.get('feature_flags', DEFAULT_SPHERESFM_FEATURE_FLAGS)))
        if hasattr(self, 'sphere_matcher_flags_edit'):
            self.sphere_matcher_flags_edit.setText(str(sfm_params.get('matcher_flags', DEFAULT_SPHERESFM_MATCHER_FLAGS)))
        if hasattr(self, 'sphere_mapper_flags_edit'):
            self.sphere_mapper_flags_edit.setText(str(sfm_params.get('mapper_flags', DEFAULT_SPHERESFM_MAPPER_FLAGS)))

        feature_values = self._parse_cli_flag_values(str(sfm_params.get('feature_flags', DEFAULT_SPHERESFM_FEATURE_FLAGS)))
        matcher_values = self._parse_cli_flag_values(str(sfm_params.get('matcher_flags', DEFAULT_SPHERESFM_MATCHER_FLAGS)))
        mapper_values = self._parse_cli_flag_values(str(sfm_params.get('mapper_flags', DEFAULT_SPHERESFM_MAPPER_FLAGS)))

        self.sphere_single_camera_check.setChecked(
            self._flag_enabled(feature_values, '--ImageReader.single_camera', True)
        )
        self.sphere_max_orientations_spin.setValue(
            int(feature_values.get('--SiftExtraction.max_num_orientations', 2))
        )
        self.sphere_peak_threshold_spin.setValue(
            float(feature_values.get('--SiftExtraction.peak_threshold', 0.00667))
        )
        self.sphere_edge_threshold_spin.setValue(
            float(feature_values.get('--SiftExtraction.edge_threshold', 10.0))
        )

        self.sphere_quadratic_overlap_check.setChecked(
            self._flag_enabled(matcher_values, '--SequentialMatching.quadratic_overlap', True)
        )
        self.sphere_loop_detection_check.setChecked(
            self._flag_enabled(matcher_values, '--SequentialMatching.loop_detection', False)
        )
        self.sphere_cross_check_check.setChecked(
            self._flag_enabled(matcher_values, '--SiftMatching.cross_check', True)
        )
        self.sphere_max_ratio_spin.setValue(float(matcher_values.get('--SiftMatching.max_ratio', 0.8)))
        self.sphere_max_distance_spin.setValue(float(matcher_values.get('--SiftMatching.max_distance', 0.7)))
        self.sphere_matcher_max_error_spin.setValue(float(matcher_values.get('--SiftMatching.max_error', 4.0)))
        self.sphere_matcher_confidence_spin.setValue(float(matcher_values.get('--SiftMatching.confidence', 0.999)))
        self.sphere_max_trials_spin.setValue(int(matcher_values.get('--SiftMatching.max_num_trials', 10000)))
        self.sphere_min_inlier_ratio_spin.setValue(float(matcher_values.get('--SiftMatching.min_inlier_ratio', 0.25)))

        self.sphere_refine_focal_check.setChecked(
            self._flag_enabled(mapper_values, '--Mapper.ba_refine_focal_length', False)
        )
        self.sphere_refine_principal_point_check.setChecked(
            self._flag_enabled(mapper_values, '--Mapper.ba_refine_principal_point', False)
        )
        self.sphere_refine_extra_params_check.setChecked(
            self._flag_enabled(mapper_values, '--Mapper.ba_refine_extra_params', False)
        )
        self.sphere_ignore_two_view_tracks_check.setChecked(
            self._flag_enabled(mapper_values, '--Mapper.tri_ignore_two_view_tracks', True)
        )
        self.sphere_multiple_models_check.setChecked(
            self._flag_enabled(mapper_values, '--Mapper.multiple_models', True)
        )
        self.sphere_init_min_inliers_spin.setValue(int(mapper_values.get('--Mapper.init_min_num_inliers', 100)))
        self.sphere_init_trials_spin.setValue(int(mapper_values.get('--Mapper.init_num_trials', 200)))
        self.sphere_init_max_error_spin.setValue(float(mapper_values.get('--Mapper.init_max_error', 4.0)))
        self.sphere_init_forward_motion_spin.setValue(float(mapper_values.get('--Mapper.init_max_forward_motion', 0.95)))
        self.sphere_init_min_tri_angle_spin.setValue(float(mapper_values.get('--Mapper.init_min_tri_angle', 16.0)))
        self.sphere_abs_pose_min_inliers_spin.setValue(int(mapper_values.get('--Mapper.abs_pose_min_num_inliers', 50)))
        self.sphere_abs_pose_max_error_spin.setValue(float(mapper_values.get('--Mapper.abs_pose_max_error', 8.0)))
        self.sphere_abs_pose_min_inlier_ratio_spin.setValue(
            float(mapper_values.get('--Mapper.abs_pose_min_inlier_ratio', 0.25))
        )
        self.sphere_max_reg_trials_spin.setValue(int(mapper_values.get('--Mapper.max_reg_trials', 3)))
        self.sphere_tri_min_angle_spin.setValue(float(mapper_values.get('--Mapper.tri_min_angle', 1.5)))
        self.sphere_tri_max_transitivity_spin.setValue(int(mapper_values.get('--Mapper.tri_max_transitivity', 1)))
        self.sphere_filter_max_reproj_error_spin.setValue(
            float(mapper_values.get('--Mapper.filter_max_reproj_error', 4.0))
        )
        self.sphere_filter_min_tri_angle_spin.setValue(
            float(mapper_values.get('--Mapper.filter_min_tri_angle', 1.5))
        )

    def _on_alignment_mode_changed(self, checked: bool = True):
        """Legacy — kept so old signal connections don\'t raise AttributeError."""
        pass
    
    def toggle_time_range(self, checked: bool):
        self.start_time_spin.setEnabled(not checked)
        self.end_time_spin.setEnabled(not checked)
    
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
            'export_include_masks': self.export_include_masks_check.isChecked() if hasattr(self, 'export_include_masks_check') else True,
            'export_image_source': self._get_export_image_source(),
            'export_mask_source': self._get_export_mask_source(),
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
            camera_model = (info.get('camera_model') or '').strip()
            camera_segment = f" | <b>Camera:</b> {camera_model}" if camera_model else ""
            self.file_metadata_label.setText(
                f"<b>Type:</b> {info.get('file_type_desc', 'Unknown')} | "
                f"<b>Duration:</b> {info.get('duration_formatted', 'N/A')} | "
                f"<b>Resolution:</b> {info.get('resolution', 'N/A')} | "
                f"<b>FPS:</b> {info.get('fps', 0):.2f} | "
                f"<b>Frames:</b> {info.get('frame_count', 0):,} | "
                f"<b>Size:</b> {info.get('file_size_mb', 0):.1f} MB"
                f"{camera_segment}"
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
        
        def _clear_pending():
            for _a in ('_pending_stage2_input', '_pending_stage3_input', '_pending_stage4_input'):
                if hasattr(self, _a):
                    delattr(self, _a)

        if self.stage1_enable.isChecked():
            if not input_file:
                QMessageBox.warning(self, "Input Required", "Please select an input file.")
                self._set_control_status("Input required", "warn")
                _clear_pending()
                return
            if not Path(input_file).exists():
                QMessageBox.warning(self, "Not Found", f"Input file not found:\n{input_file}")
                self._set_control_status("Input file not found", "error")
                _clear_pending()
                return
            if Path(input_file).is_dir():
                QMessageBox.warning(
                    self,
                    "Invalid Input for Extraction",
                    "Stage 1 (Frame Extraction) expects a video file (.insv/.mp4), not a folder.\n"
                    "Either select a video file or disable Stage 1 and run Split/Mask on existing images."
                )
                self._set_control_status("Stage 1 requires a video file", "warn")
                _clear_pending()
                return
        if not output_dir:
            QMessageBox.warning(self, "Output Required", "Please select an output directory.")
            self._set_control_status("Output required", "warn")
            _clear_pending()
            return
        
        stage2_method = self.stage2_method_combo.currentData()
        
        self.pipeline_config = {
            'input_file': input_file,
            'output_dir': output_dir,
            'enable_stage1': self.stage1_enable.isChecked(),
            'skip_transform': not self.stage2_enable.isChecked(),
            'enable_stage2': self.stage2_enable.isChecked(),
            'enable_stage3': self.stage3_enable.isChecked(),
            'export_realityscan': self.export_realityscan_check.isChecked() if hasattr(self, 'export_realityscan_check') else False,
            'export_include_masks': self.export_include_masks_check.isChecked() if hasattr(self, 'export_include_masks_check') else True,
            'fps': self.fps_spin.value(),
            'extraction_method': self.extraction_method_combo.currentData(),
            'start_time': 0.0 if self.full_video_check.isChecked() else self.start_time_spin.value(),
            'end_time': None if self.full_video_check.isChecked() else self.end_time_spin.value(),
            'sdk_quality': self.sdk_quality_combo.currentData(),
            'sdk_resolution': self.sdk_resolution_combo.currentData(),
            'output_format': _normalize_image_format(self.output_format_combo.currentData(), 'jpg'),
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
                'stage2_format': _normalize_image_format(self.stage2_format_combo.currentData(), 'png'),
                'perspective_params': {'camera_groups': camera_groups}
            })
        elif stage2_method == 'cubemap':
            tw = self.cubemap_tile_width_spin.value()
            th = self.cubemap_tile_height_spin.value()
            ct = self.cubemap_type_combo.currentData()
            self.pipeline_config.update({
                'output_width': tw,
                'output_height': th,
                'stage2_format': _normalize_image_format(self.cubemap_format_combo.currentData(), 'png'),
                'cubemap_params': {
                    'cubemap_type': ct, 'tile_width': tw, 'tile_height': th,
                    'fov': 90, 'layout': 'separate'
                }
            })
        
        # Masking classes
        if self.stage3_enable.isChecked():
            _obj_map = {
                "Backpack": 24, "Umbrella": 25, "Handbag": 26,
                "Tie": 27, "Suitcase": 28, "Cell Phone": 67,
            }
            _animal_map = {
                "Bird": 14, "Cat": 15, "Dog": 16, "Horse": 17,
                "Sheep": 18, "Cow": 19, "Elephant": 20, "Bear": 21,
                "Zebra": 22, "Giraffe": 23,
            }

            person_classes = (
                [0] if self.persons_enable.isChecked() and 'Person' in self.persons_combo.checkedTexts()
                else []
            )

            object_classes = []
            if self.objects_enable.isChecked():
                for txt, cls_id in _obj_map.items():
                    if txt in self.objects_combo.checkedTexts():
                        object_classes.append(cls_id)

            animal_classes = []
            if self.animals_enable.isChecked():
                for txt, cls_id in _animal_map.items():
                    if txt in self.animals_combo.checkedTexts():
                        animal_classes.append(cls_id)

            self.pipeline_config.update({
                'masking_engine': self.masking_engine_combo.currentData(),
                'model_size': self.model_size_combo.currentData(),
                'confidence_threshold': self.confidence_spin.value(),
                'use_gpu': self.use_gpu_check.isChecked(),
                'masking_categories': {
                    'persons': len(person_classes) > 0,
                    'personal_objects': len(object_classes) > 0,
                    'animals': len(animal_classes) > 0,
                },
                'masking_classes': {
                    'persons': person_classes,
                    'personal_objects': object_classes,
                    'animals': animal_classes,
                },
                'stage3_image_source': self._get_stage3_image_source(),
                'mask_target': self._legacy_mask_target_from_source(self._get_stage3_image_source()),
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
        # (e.g. run_stage_2_only / run_stage_3_only set _pending_stage*_input before calling us)
        if hasattr(self, '_pending_stage2_input') and self._pending_stage2_input:
            self.pipeline_config['stage2_input_dir'] = self._pending_stage2_input
            del self._pending_stage2_input
        if hasattr(self, '_pending_stage3_input') and self._pending_stage3_input:
            self.pipeline_config['stage3_input_dir'] = self._pending_stage3_input
            del self._pending_stage3_input
        if hasattr(self, '_pending_stage4_input') and self._pending_stage4_input:
            self.pipeline_config['stage4_input_dir'] = self._pending_stage4_input
            del self._pending_stage4_input

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
    def _on_media_flowstate_changed(self, sdk_opts: dict):
        """Auto-reset the preview rotation to 0° when FlowState is newly enabled.
        FlowState applies gyroscope stabilisation which levels the horizon,
        so no manual frame rotation is needed afterwards.
        """
        flowstate_on = sdk_opts.get('enable_flowstate', False)
        was_on = getattr(self, '_prev_flowstate_on', None)
        self._prev_flowstate_on = flowstate_on

        # Only act on the OFF → ON transition
        if flowstate_on and was_on is False:
            if hasattr(self, 'stage1_eq_preview'):
                preview = self.stage1_eq_preview
                if preview._rotation != 0:
                    preview._rotation = 0
                    preview._rotate_btn.setText("\u21bb 0\u00b0")
                    preview._trigger_render()

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
        self._pending_stage2_input = str(folder)
        self.start_pipeline()
        self.stage1_enable.setChecked(s1)
        self.stage3_enable.setChecked(s3)
    
    def run_stage_3_only(self):
        self.log_message("Running masking only")
        output_dir = self.output_dir_edit.text()
        if not output_dir:
            QMessageBox.warning(self, "Missing", "Configure output directory first")
            return

        output_root = Path(output_dir)
        _img_exts = ('*.png', '*.PNG', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.tif', '*.TIF', '*.tiff', '*.TIFF')

        def _has_images(p: Path) -> bool:
            return p.is_dir() and any(f for ext in _img_exts for f in p.glob(ext))

        # If output_root itself is already an images folder, use it directly
        # (happens when user sets output dir directly to perspective_views/extracted_frames)
        folder = None
        if _has_images(output_root):
            folder = output_root
        else:
            preferred_source = self._get_stage3_image_source()
            folder = self._resolve_output_image_dir(output_root, preferred_source)
            if folder is None:
                from src.pipeline.batch_orchestrator import PipelineWorker
                worker = PipelineWorker({})
                folder = worker.discover_stage_input_folder(stage=3, output_dir=output_dir)

        if not folder:
            folder_str = QFileDialog.getExistingDirectory(
                self, "Select images to mask", str(output_root)
            )
            if not folder_str:
                return
            folder = Path(folder_str)

        images = [f for ext in _img_exts for f in folder.glob(ext)]
        if not images:
            QMessageBox.warning(self, "No Images", f"No images found in:\n{folder}")
            return

        self.log_message(f"Found {len(images)} images to mask in: {folder.name}")
        # Store the resolved folder so start_pipeline() can inject it after
        # rebuilding pipeline_config (it overwrites the whole dict from scratch).
        self._pending_stage3_input = str(folder)
        # Stage 3 is the terminal stage — do NOT set _auto_advance_enabled to True.
        # If we were called via auto-advance from stage 2, the flag is already True
        # and will be reset to False by on_stage_complete when stage 3 finishes.
        # Setting it True here is what caused the infinite loop (skipped stage-2
        # signal → auto-advance → run_stage_3_only → repeat).
        s1, s2 = self.stage1_enable.isChecked(), self.stage2_enable.isChecked()
        self.stage1_enable.setChecked(False)
        self.stage2_enable.setChecked(False)
        self.start_pipeline()
        self.stage1_enable.setChecked(s1)
        self.stage2_enable.setChecked(s2)
    
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
            if self._auto_advance_enabled and not results.get('skipped'):
                next_map = {
                    1: (self.stage2_enable, self.run_stage_2_only),
                    2: (self.stage3_enable, self.run_stage_3_only),
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
