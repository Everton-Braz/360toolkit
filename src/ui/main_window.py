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
import tempfile
from pathlib import Path
from datetime import datetime

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QProgressBar, QTabWidget,
    QFileDialog, QMessageBox, QTextEdit, QGroupBox,
    QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
    QLineEdit, QSplitter, QScrollArea, QSizePolicy,
    QRadioButton, QButtonGroup, QFrame, QStackedWidget,
    QApplication, QToolButton, QGridLayout, QSpacerItem, QStyle,
    QSlider
)
from PyQt6.QtCore import Qt, QTimer, QSize, QPropertyAnimation, QEasingCurve, pyqtSignal, QEvent, QObject, QThread
from PyQt6.QtGui import QFont, QAction, QIcon, QPainter, QColor, QPen, QPixmap, QShortcut, QKeySequence, QPalette, QStandardItem

from src.pipeline.batch_orchestrator import BatchOrchestrator
from src.config.defaults import (
    APP_NAME, APP_VERSION,
    DEFAULT_FPS, DEFAULT_H_FOV, DEFAULT_SPLIT_COUNT,
    DEFAULT_STAGE2_LAYOUT_MODE, DEFAULT_STAGE2_NUMBERING_MODE,
    EXTRACTION_METHODS, TRANSFORM_TYPES, YOLOV8_MODELS,
    STAGE2_LAYOUT_MODES, STAGE2_NUMBERING_MODES,
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
    SAM3PreviewWidget,
)
from src.ui.preview_panels import EquirectPreviewWidget
from src.utils.runtime_backends import has_bundled_onnx_runtime, has_usable_torch_runtime, is_usable_torch_module
from src.pipeline.stage2_naming import perspective_output_sort_key
from src.utils.dependency_provisioning import resolve_masking_model_path

logger = logging.getLogger(__name__)


class _QtLogEmitter(QObject):
    message = pyqtSignal(str)


class _QtLogHandler(logging.Handler):
    def __init__(self, emitter: _QtLogEmitter):
        super().__init__(level=logging.INFO)
        self._emitter = emitter
        self.setFormatter(logging.Formatter('%(message)s'))

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno < logging.INFO:
            return
        try:
            self._emitter.message.emit(self.format(record))
        except Exception:
            pass


class _InputAnalysisWorker(QThread):
    completed = pyqtSignal(str, dict)
    failed = pyqtSignal(str, str)

    def __init__(self, input_path: str, parent=None):
        super().__init__(parent)
        self._input_path = input_path

    def run(self) -> None:
        try:
            from src.extraction import FrameExtractor

            extractor = FrameExtractor()
            info = extractor.get_video_info(self._input_path)
            self.completed.emit(self._input_path, info)
        except Exception as exc:
            self.failed.emit(self._input_path, str(exc))

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
        self._last_auto_sdk_defaults_key = None
        self._qt_log_emitter = _QtLogEmitter()
        self._qt_log_handler = _QtLogHandler(self._qt_log_emitter)
        self._input_analysis_worker = None
        self._active_input_analysis = ""
        self._pending_input_analysis = None
        
        self.init_ui()
        self._attach_backend_log_handler()
        self.create_menu_bar()
        self.apply_theme()
        self._setup_shortcuts()
        self.on_extraction_method_changed(0)
        self._update_overview_stage_summary()
        self.on_settings_changed()
        self.input_file_edit.textChanged.connect(self._on_input_file_changed)
        self.output_dir_edit.textChanged.connect(self._on_stage3_preview_source_changed)
        self.showMaximized()

    def _attach_backend_log_handler(self):
        self._qt_log_emitter.message.connect(self._append_backend_log)
        root_logger = logging.getLogger()
        if root_logger.level > logging.INFO:
            root_logger.setLevel(logging.INFO)
        root_logger.addHandler(self._qt_log_handler)

    def _append_backend_log(self, message: str):
        if not message:
            return
        if not self._should_show_backend_log(message):
            return
        self.log_message(message)

    def _should_show_backend_log(self, message: str) -> bool:
        normalized = str(message).strip()
        if not normalized:
            return False

        # Explicit suppression list — checked before anything else
        never_show = (
            '=== ONNX Runtime DLL Diagnostics ===',
            '=== End Diagnostics ===',
            'ONNX Runtime DLL Diagnostics',
        )
        if any(marker in normalized for marker in never_show):
            return False

        low_signal_markers = (
            'Found model:',
            'Found SDK executable:',
            'Found bundled FFprobe',
            'FFmpeg found at:',
            'FFprobe found at:',
            'Current PATH dirs:',
            'DLLs in capi:',
            'PYDs in capi:',
            'MSVC runtimes in _internal:',
            'Adding ONNX DLL path:',
            'Adding NumPy DLL path:',
            'Adding Internal DLL path:',
            'Adding Exe DLL path:',
            'ctypes preloaded',
            'Attempting to import onnxruntime...',
            '[GPU Detect]',
            'SDK PATH:',
            'DLLs in SDK bin:',
            'Found nvcuda.dll at',
            '[SDK STDOUT]',
            '[SDK STDERR]',
            '[Diagnostics]',
        )
        if any(marker in normalized for marker in low_signal_markers):
            return False

        high_signal_markers = (
            'Starting pipeline',
            'Pipeline complete',
            'Pipeline failed',
            '=== ',
            '[OK] Stage',
            '[FAIL] Stage',
            'Error:',
            'Masking Initialization Failed',
            'Split (',
            'Done \u2014',
            'Running ',
            'Found ',
            'Analyzed:',
            'SDK preview failed',
            'Lens preview failed',
            'Using specified input:',
        )
        if any(marker in normalized for marker in high_signal_markers):
            return True

        return normalized.startswith('[OK]') or normalized.startswith('[WARN]') or normalized.startswith('[FAIL]')

    def closeEvent(self, event):
        try:
            logging.getLogger().removeHandler(self._qt_log_handler)
        except Exception:
            pass
        try:
            if self._input_analysis_worker and self._input_analysis_worker.isRunning():
                self._input_analysis_worker.terminate()
                self._input_analysis_worker.wait()
        except Exception:
            pass
        super().closeEvent(event)
    
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
        self.stage1_eq_preview.preview_timestamp_changed.connect(
            self._on_stage1_preview_timestamp_changed
        )
        self.stage1_eq_preview.preview_frame_available.connect(
            self._on_stage1_preview_frame_available
        )
        self.start_time_spin.valueChanged.connect(
            self._sync_preview_timestamp_from_stage1_range
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

        self.stage2_numbering_combo = QComboBox()
        for mode, label in STAGE2_NUMBERING_MODES.items():
            self.stage2_numbering_combo.addItem(label, mode)
        self.stage2_numbering_combo.setCurrentIndex(
            max(0, self.stage2_numbering_combo.findData(DEFAULT_STAGE2_NUMBERING_MODE))
        )
        self.stage2_numbering_combo.setMinimumWidth(220)

        self.stage2_layout_combo = QComboBox()
        for mode, label in STAGE2_LAYOUT_MODES.items():
            self.stage2_layout_combo.addItem(label, mode)
        self.stage2_layout_combo.setCurrentIndex(
            max(0, self.stage2_layout_combo.findData(DEFAULT_STAGE2_LAYOUT_MODE))
        )
        self.stage2_layout_combo.setMinimumWidth(220)

        stage2_input_widget = QWidget()
        stage2_input_layout = QHBoxLayout(stage2_input_widget)
        stage2_input_layout.setContentsMargins(0, 0, 0, 0)
        stage2_input_layout.setSpacing(8)
        self.stage2_input_dir_edit = QLineEdit()
        self.stage2_input_dir_edit.setPlaceholderText("Optional: folder with extracted/equirectangular images for split-only runs")
        self.stage2_input_dir_edit.textChanged.connect(self._on_stage2_input_dir_changed)
        stage2_input_layout.addWidget(self.stage2_input_dir_edit)
        self.stage2_input_browse_btn = QPushButton("Browse…")
        self.stage2_input_browse_btn.setFixedWidth(92)
        self.stage2_input_browse_btn.clicked.connect(
            lambda: self._browse_for_directory(self.stage2_input_dir_edit, "Select Stage 2 Input Folder")
        )
        stage2_input_layout.addWidget(self.stage2_input_browse_btn)

        card_output.addWidget(FormRow("Output Size:", dims_widget, "Width × Height in pixels"))
        card_output.addWidget(FormRow("Format:", self.stage2_format_combo))
        card_output.addWidget(FormRow("Frame Numbering:", self.stage2_numbering_combo, "Preserve extracted frame ids or renumber sequentially"))
        card_output.addWidget(FormRow("Folder Layout:", self.stage2_layout_combo, "Keep all views flat or group them into per-camera folders"))
        card_output.addWidget(FormRow("Input Folder:", stage2_input_widget, "Used when running Stage 2 without Stage 1; leave empty for auto-discovery"))
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

        self.stage2_numbering_combo_cubemap = QComboBox()
        for mode, label in STAGE2_NUMBERING_MODES.items():
            self.stage2_numbering_combo_cubemap.addItem(label, mode)
        self.stage2_numbering_combo_cubemap.setCurrentIndex(
            max(0, self.stage2_numbering_combo_cubemap.findData(DEFAULT_STAGE2_NUMBERING_MODE))
        )
        self.stage2_numbering_combo_cubemap.setMinimumWidth(220)
        self.stage2_numbering_combo_cubemap.currentIndexChanged.connect(
            lambda _index: self._mirror_combo_selection(self.stage2_numbering_combo_cubemap, self.stage2_numbering_combo)
        )
        self.stage2_numbering_combo.currentIndexChanged.connect(
            lambda _index: self._mirror_combo_selection(self.stage2_numbering_combo, self.stage2_numbering_combo_cubemap)
        )
        card_tiles.addWidget(FormRow("Frame Numbering:", self.stage2_numbering_combo_cubemap, "Use the same frame ids as extraction or renumber sequentially"))

        self.cubemap_layout_combo = QComboBox()
        self.cubemap_layout_combo.addItem("Flat Folder", "flat")
        self.cubemap_layout_combo.addItem("Separate By Tile", "by_camera")
        self.cubemap_layout_combo.setCurrentIndex(
            max(0, self.cubemap_layout_combo.findData(DEFAULT_STAGE2_LAYOUT_MODE))
        )
        self.cubemap_layout_combo.setMinimumWidth(220)
        card_tiles.addWidget(FormRow("Folder Layout:", self.cubemap_layout_combo, "Keep all cubemap tiles together or group them into per-tile folders"))

        cubemap_input_widget = QWidget()
        cubemap_input_layout = QHBoxLayout(cubemap_input_widget)
        cubemap_input_layout.setContentsMargins(0, 0, 0, 0)
        cubemap_input_layout.setSpacing(8)
        self.stage2_input_dir_edit_cubemap = QLineEdit()
        self.stage2_input_dir_edit_cubemap.setPlaceholderText("Optional: folder with extracted/equirectangular images for split-only runs")
        self.stage2_input_dir_edit_cubemap.textChanged.connect(self._on_stage2_input_dir_changed)
        cubemap_input_layout.addWidget(self.stage2_input_dir_edit_cubemap)
        self.stage2_input_browse_btn_cubemap = QPushButton("Browse…")
        self.stage2_input_browse_btn_cubemap.setFixedWidth(92)
        self.stage2_input_browse_btn_cubemap.clicked.connect(
            lambda: self._browse_for_directory(self.stage2_input_dir_edit_cubemap, "Select Stage 2 Input Folder")
        )
        cubemap_input_layout.addWidget(self.stage2_input_browse_btn_cubemap)
        card_tiles.addWidget(FormRow("Input Folder:", cubemap_input_widget, "Used when running Stage 2 without Stage 1; leave empty for auto-discovery"))
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
            "Choose the masking engine and source, then validate the Stage 3 settings before running."
        ))

        # ── Engine & Quality — merged card ─────────────────────────────────
        card_model = CardWidget("Engine & Quality")

        self.masking_engine_combo = QComboBox()
        self.masking_engine_combo.addItem("SAM 3", "sam3_cpp")
        self.masking_engine_combo.addItem("YOLO Segmentation", "yolo")
        self.masking_engine_combo.setCurrentIndex(0)
        self.masking_engine_combo.setMinimumWidth(280)
        self.masking_engine_combo.currentIndexChanged.connect(self.on_masking_engine_changed)
        card_model.addWidget(FormRow("Masking Engine:", self.masking_engine_combo))
        
        self.engine_description_label = QLabel("Primary engine: SAM 3. Secondary packaged option: YOLO segmentation.")
        self.engine_description_label.setProperty("role", "mutedSmall")
        card_model.addWidget(self.engine_description_label)
        self.masking_runtime_label = QLabel("Runtime: detecting ONNX / PyTorch backends...")
        self.masking_runtime_label.setProperty("role", "mutedSmall")
        card_model.addWidget(self.masking_runtime_label)

        self.mask_input_source_combo = QComboBox()
        self.mask_input_source_combo.addItem("Auto", "auto")
        self.mask_input_source_combo.addItem("Perspective Views", "perspective")
        self.mask_input_source_combo.addItem("Equirect / Extracted Frames", "equirect")
        self.mask_input_source_combo.currentIndexChanged.connect(self._on_stage3_preview_source_changed)
        card_model.addWidget(FormRow("Input Source:", self.mask_input_source_combo))

        stage3_input_widget = QWidget()
        stage3_input_layout = QHBoxLayout(stage3_input_widget)
        stage3_input_layout.setContentsMargins(0, 0, 0, 0)
        stage3_input_layout.setSpacing(8)
        self.stage3_input_dir_edit = QLineEdit()
        self.stage3_input_dir_edit.setPlaceholderText("Optional: folder with images for masking-only runs")
        self.stage3_input_dir_edit.textChanged.connect(self._on_stage3_input_dir_changed)
        stage3_input_layout.addWidget(self.stage3_input_dir_edit)
        self.stage3_input_browse_btn = QPushButton("Browse…")
        self.stage3_input_browse_btn.setFixedWidth(92)
        self.stage3_input_browse_btn.clicked.connect(
            lambda: self._browse_for_directory(self.stage3_input_dir_edit, "Select Stage 3 Input Folder")
        )
        stage3_input_layout.addWidget(self.stage3_input_browse_btn)
        card_model.addWidget(FormRow("Input Folder:", stage3_input_widget, "Used when running Stage 3 without Stage 1/2; leave empty for auto-discovery"))

        auto_note = QLabel(
            "Auto uses perspective views when available and falls back to extracted frames. "
            "Masks are stored separately as masks_perspective or masks_equirect."
        )
        auto_note.setProperty("role", "secondary")
        auto_note.setWordWrap(True)
        card_model.addWidget(auto_note)
        
        # Model size
        self.model_size_container = QWidget()
        ms_layout = QHBoxLayout(self.model_size_container)
        ms_layout.setContentsMargins(0, 0, 0, 0)
        self.model_size_combo = QComboBox()
        self.model_size_combo.addItem("Nano (10MB) - Fastest", "nano")
        self.model_size_combo.addItem("Small (40MB) - Balanced", "small")
        self.model_size_combo.addItem("Medium (90MB) - Best", "medium")
        self.model_size_combo.addItem("Large (140MB) - Higher accuracy", "large")
        self.model_size_combo.addItem("XLarge (220MB) - Maximum accuracy", "xlarge")
        self.model_size_combo.setCurrentIndex(1)
        self.model_size_combo.setFixedWidth(240)
        ms_layout.addWidget(FormRow("Model Size:", self.model_size_combo))
        card_model.addWidget(self.model_size_container)

        self.yolo_model_path_container = QWidget()
        yolo_model_layout = QHBoxLayout(self.yolo_model_path_container)
        yolo_model_layout.setContentsMargins(0, 0, 0, 0)
        yolo_model_layout.setSpacing(8)
        self.yolo_model_path_edit = QLineEdit()
        self.yolo_model_path_edit.setPlaceholderText("Optional: custom YOLO ONNX model (.onnx). Overrides size selection.")
        self.yolo_model_path_edit.textChanged.connect(self._update_masking_runtime_status)
        yolo_model_layout.addWidget(self.yolo_model_path_edit)
        self.yolo_model_browse_btn = QPushButton("Browse…")
        self.yolo_model_browse_btn.setFixedWidth(92)
        self.yolo_model_browse_btn.clicked.connect(
            lambda: self._browse_for_file(self.yolo_model_path_edit, "Select YOLO ONNX Model", "ONNX Models (*.onnx)")
        )
        yolo_model_layout.addWidget(self.yolo_model_browse_btn)
        card_model.addWidget(FormRow("Custom ONNX:", self.yolo_model_path_container, "Optional: choose a specific YOLO ONNX file instead of the bundled size map"))

        self.mask_output_mode_container = QWidget()
        mask_output_layout = QHBoxLayout(self.mask_output_mode_container)
        mask_output_layout.setContentsMargins(0, 0, 0, 0)
        self.sam3_output_mode_combo = QComboBox()
        self.sam3_output_mode_combo.addItem("Mask files only  (separate _mask.png per image)", "masks_only")
        self.sam3_output_mode_combo.addItem("Alpha cutout PNG only  (transparent area embedded — no mask files)", "alpha_only")
        self.sam3_output_mode_combo.addItem("Both  (alpha PNG + separate mask file)", "both")
        self.sam3_output_mode_combo.setCurrentIndex(0)
        self.sam3_output_mode_combo.currentIndexChanged.connect(self._on_sam3_config_changed)
        mask_output_layout.addWidget(self.sam3_output_mode_combo)
        mask_output_layout.addStretch()
        card_model.addWidget(FormRow("Output mode:", self.mask_output_mode_container, "Controls whether masking writes separate masks, alpha cutouts, or both. PNG split output can reuse alpha cutouts before Stage 2."))
        
        self.confidence_container = QWidget()
        conf_row = QHBoxLayout(self.confidence_container)
        conf_row.setContentsMargins(0, 0, 0, 0)
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
        card_model.addWidget(self.confidence_container)

        self.sam3_options_container = QWidget()
        sam3_layout = QVBoxLayout(self.sam3_options_container)
        sam3_layout.setContentsMargins(0, 0, 0, 0)
        sam3_layout.setSpacing(8)

        sam3_paths_hint = QLabel("Configure segment_persons.exe, the SAM3 model, and sam3_image.exe in Settings > Paths & Detection.")
        sam3_paths_hint.setWordWrap(True)
        sam3_paths_hint.setProperty("role", "secondary")
        sam3_layout.addWidget(sam3_paths_hint)

        sam3_feather_row = QWidget()
        sam3_feather_layout = QHBoxLayout(sam3_feather_row)
        sam3_feather_layout.setContentsMargins(0, 0, 0, 0)
        sam3_feather_layout.setSpacing(8)
        self.sam3_feather_spin = QSpinBox()
        self.sam3_feather_spin.setRange(0, 40)
        self.sam3_feather_spin.setValue(8)
        self.sam3_feather_spin.valueChanged.connect(self._on_sam3_config_changed)
        sam3_feather_layout.addWidget(self.sam3_feather_spin)
        sam3_feather_layout.addWidget(QLabel("Width of the local boundary refinement band."))
        sam3_feather_layout.addStretch()
        sam3_layout.addWidget(FormRow("Refinement Band:", sam3_feather_row))

        self.sam3_enable_refinement = True
        self.sam3_refine_sky_only = True
        self.sam3_seam_aware_refinement = True

        sam3_sharpen_widget = QWidget()
        sam3_sharpen_layout = QHBoxLayout(sam3_sharpen_widget)
        sam3_sharpen_layout.setContentsMargins(0, 0, 0, 0)
        sam3_sharpen_layout.setSpacing(8)
        self.sam3_edge_sharpen_spin = QDoubleSpinBox()
        self.sam3_edge_sharpen_spin.setRange(0.0, 2.0)
        self.sam3_edge_sharpen_spin.setSingleStep(0.05)
        self.sam3_edge_sharpen_spin.setDecimals(2)
        self.sam3_edge_sharpen_spin.setValue(0.75)
        self.sam3_edge_sharpen_spin.setFixedWidth(80)
        self.sam3_edge_sharpen_spin.valueChanged.connect(self._on_sam3_config_changed)
        sam3_sharpen_layout.addWidget(self.sam3_edge_sharpen_spin)
        sam3_sharpen_layout.addWidget(QLabel("Higher values push mask edges harder toward image boundaries."))
        sam3_sharpen_layout.addStretch()
        sam3_layout.addWidget(FormRow("Edge Sharpen:", sam3_sharpen_widget))

        # ── Prompt Categories ────────────────────────────────────────────────
        sam3_cats_sep = QFrame()
        sam3_cats_sep.setFrameShape(QFrame.Shape.HLine)
        sam3_cats_sep.setProperty("role", "divider")
        sam3_layout.addWidget(sam3_cats_sep)

        sam3_cats_label = QLabel("Detection Targets (SAM3 prompts):")
        sam3_cats_label.setProperty("role", "secondary")
        sam3_layout.addWidget(sam3_cats_label)

        sam3_prompts_row = QWidget()
        sam3_prompts_grid = QGridLayout(sam3_prompts_row)
        sam3_prompts_grid.setContentsMargins(0, 0, 0, 0)
        sam3_prompts_grid.setHorizontalSpacing(16)
        sam3_prompts_grid.setVerticalSpacing(4)

        _sam3_cats = [
            ('persons',  'Persons',  0, 0),
            ('bags',     'Bags',     0, 1),
            ('phones',   'Phones',   0, 2),
            ('hats',     'Hats',     1, 0),
            ('helmets',  'Helmets',  1, 1),
            ('sky',      'Sky',      1, 2),
        ]
        self.sam3_prompt_checks: dict = {}
        for key, label, row, col in _sam3_cats:
            cb = QCheckBox(label)
            cb.setChecked(True)
            cb.toggled.connect(self._on_sam3_config_changed)
            sam3_prompts_grid.addWidget(cb, row, col)
            self.sam3_prompt_checks[key] = cb
        sam3_prompts_grid.setColumnStretch(3, 1)
        sam3_layout.addWidget(sam3_prompts_row)

        self.sam3_custom_prompts_edit = QLineEdit()
        self.sam3_custom_prompts_edit.setPlaceholderText("Additional prompts, e.g.: car, tree, backpack")
        self.sam3_custom_prompts_edit.textChanged.connect(self._on_sam3_config_changed)
        sam3_layout.addWidget(FormRow("Custom prompts:", self.sam3_custom_prompts_edit))

        # ── Morph / Dilate-Erode slider ─────────────────────────────────────
        sam3_morph_widget = QWidget()
        sam3_morph_layout = QHBoxLayout(sam3_morph_widget)
        sam3_morph_layout.setContentsMargins(0, 0, 0, 0)
        sam3_morph_layout.setSpacing(8)
        self.sam3_morph_slider = QSlider(Qt.Orientation.Horizontal)
        self.sam3_morph_slider.setRange(-50, 50)
        self.sam3_morph_slider.setValue(0)
        self.sam3_morph_slider.setFixedWidth(200)
        self.sam3_morph_slider.setTickInterval(10)
        self.sam3_morph_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.sam3_morph_spin = QSpinBox()
        self.sam3_morph_spin.setRange(-50, 50)
        self.sam3_morph_spin.setValue(0)
        self.sam3_morph_spin.setFixedWidth(60)
        self.sam3_morph_spin.setToolTip("Negative = dilate (expand), positive = erode (shrink)")
        def _update_morph_label(v):
            self.sam3_morph_spin.blockSignals(True)
            self.sam3_morph_spin.setValue(v)
            self.sam3_morph_spin.blockSignals(False)
            self._on_sam3_config_changed()
        def _update_morph_slider(v):
            self.sam3_morph_slider.blockSignals(True)
            self.sam3_morph_slider.setValue(v)
            self.sam3_morph_slider.blockSignals(False)
            self._on_sam3_config_changed()
        self.sam3_morph_slider.valueChanged.connect(_update_morph_label)
        self.sam3_morph_spin.valueChanged.connect(_update_morph_slider)
        sam3_morph_layout.addWidget(QLabel("Dilate"))
        sam3_morph_layout.addWidget(self.sam3_morph_slider)
        sam3_morph_layout.addWidget(QLabel("Erode"))
        sam3_morph_layout.addWidget(self.sam3_morph_spin)
        sam3_morph_layout.addStretch()
        sam3_layout.addWidget(FormRow("Mask Morphology:", sam3_morph_widget))

        # ── Max input width ──────────────────────────────────────────────────
        sam3_maxw_widget = QWidget()
        sam3_maxw_layout = QHBoxLayout(sam3_maxw_widget)
        sam3_maxw_layout.setContentsMargins(0, 0, 0, 0)
        sam3_maxw_layout.setSpacing(8)
        self.sam3_maxw_combo = QComboBox()
        self.sam3_maxw_combo.addItem("Original (no downscale)", 0)
        self.sam3_maxw_combo.addItem("3840px  (recommended — 4× faster encode)", 3840)
        self.sam3_maxw_combo.addItem("1920px  (fast preview)", 1920)
        self.sam3_maxw_combo.addItem("1024px  (ultra-fast)", 1024)
        self.sam3_maxw_combo.setCurrentIndex(1)
        self.sam3_maxw_combo.currentIndexChanged.connect(self._on_sam3_config_changed)
        sam3_maxw_layout.addWidget(self.sam3_maxw_combo)
        sam3_maxw_layout.addStretch()
        sam3_layout.addWidget(FormRow("Max input width:", sam3_maxw_widget))

        card_model.addWidget(self.sam3_options_container)

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

        # ── Detection Categories — compact dropdown card ────────────────────
        self.masking_categories_card = CardWidget("YOLO Detection Categories")
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

        self.masking_categories_card.addLayout(cats_grid)
        layout.addWidget(self.masking_categories_card)

        self.sam3_preview_card = CardWidget("SAM3 Preview")
        self.sam3_preview_widget = SAM3PreviewWidget()
        self.sam3_preview_widget.set_config_provider(self._get_sam3_preview_config)
        self.sam3_preview_widget.set_auto_image_resolver(self._resolve_stage3_preview_candidate)
        self.sam3_preview_widget.message_emitted.connect(lambda message: self.log_message(f"[SAM3] {message}"))
        self.sam3_preview_card.addWidget(self.sam3_preview_widget)
        layout.addWidget(self.sam3_preview_card)

        stage3_footer = StageActionFooter("Run Masking")
        stage3_footer.primary_button.clicked.connect(self.run_stage_3_only)
        stage3_footer.validate_button.clicked.connect(lambda: self._validate_stage_config(4))
        layout.addWidget(stage3_footer)
        
        self.on_masking_engine_changed(self.masking_engine_combo.currentIndex())
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
        stage2_input_obj = self._get_stage2_input_path()
        needs_global_input = not (stage_index in (2, 3) and not self.stage1_enable.isChecked() and stage2_input_obj is not None)

        if stage_index in (0, 1, 2, 3, 4, 5, 6):
            if needs_global_input and not input_path:
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

        if stage_index in (2, 3) and stage2_input_obj is not None:
            if not stage2_input_obj.exists() or not stage2_input_obj.is_dir():
                self.log_message(f"[WARN] Split validation: Stage 2 input folder not found: {stage2_input_obj}")
                self._set_control_status("Stage 2 input folder not found", "warn")
                return
            if not self._directory_has_images(stage2_input_obj):
                self.log_message(f"[WARN] Split validation: No images found in Stage 2 input folder: {stage2_input_obj}")
                self._set_control_status("No images in Stage 2 input folder", "warn")
                return

        stage3_input_obj = self._get_stage3_input_path()
        if stage_index == 4 and stage3_input_obj is not None:
            if not stage3_input_obj.exists() or not stage3_input_obj.is_dir():
                self.log_message(f"[WARN] Masking validation: Stage 3 input folder not found: {stage3_input_obj}")
                self._set_control_status("Stage 3 input folder not found", "warn")
                return
            if not self._directory_has_images(stage3_input_obj):
                self.log_message(f"[WARN] Masking validation: No images found in Stage 3 input folder: {stage3_input_obj}")
                self._set_control_status("No images in Stage 3 input folder", "warn")
                return

        if stage_index == 4 and self._normalize_masking_engine(self.masking_engine_combo.currentData()) == 'yolo' and not (
            self.persons_enable.isChecked() or self.objects_enable.isChecked() or self.animals_enable.isChecked()
        ):
            self.log_message("[WARN] Masking validation: No masking category group is enabled.")
            self._set_control_status("No masking categories enabled", "warn")
            return

        if stage_index == 4 and self._normalize_masking_engine(self.masking_engine_combo.currentData()) == 'yolo':
            custom_model_text = self.yolo_model_path_edit.text().strip() if hasattr(self, 'yolo_model_path_edit') else ''
            if custom_model_text:
                custom_model = Path(custom_model_text).expanduser()
                if not custom_model.exists() or not custom_model.is_file():
                    self.log_message(f"[WARN] Masking validation: Custom YOLO model not found: {custom_model}")
                    self._set_control_status("Custom YOLO model not found", "warn")
                    return
                if custom_model.suffix.lower() != '.onnx':
                    self.log_message(f"[WARN] Masking validation: Custom YOLO model must be an ONNX file: {custom_model.name}")
                    self._set_control_status("Custom YOLO model must be ONNX", "warn")
                    return

        if stage_index == 4 and self.masking_engine_combo.currentData() == 'sam3_cpp':
            segmenter_text = self._get_sam3_segmenter_text()
            model_text = self._get_sam3_model_text()
            segmenter = Path(segmenter_text) if segmenter_text else None
            model = Path(model_text) if model_text else None
            if not segmenter or not segmenter.exists():
                self.log_message("[WARN] Masking validation: segment_persons.exe is not configured or missing.")
                self._set_control_status("SAM3.cpp segmenter missing", "warn")
                return
            if not model or not model.exists():
                self.log_message("[WARN] Masking validation: SAM3 model path is not configured or missing.")
                self._set_control_status("SAM3 model missing", "warn")
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
        cc_spin.setMinimumWidth(92)
        cc_spin.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        row.addWidget(cc_spin)
        
        pitch_lbl = QLabel("Pitch")
        pitch_lbl.setProperty("role", "muted")
        row.addWidget(pitch_lbl)
        p_spin = QSpinBox()
        p_spin.setRange(-90, 90)
        p_spin.setValue(pitch)
        p_spin.setMinimumWidth(92)
        p_spin.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        row.addWidget(p_spin)
        
        fov_lbl = QLabel("FOV")
        fov_lbl.setProperty("role", "muted")
        row.addWidget(fov_lbl)
        f_spin = QSpinBox()
        f_spin.setRange(30, 150)
        f_spin.setValue(fov)
        f_spin.setMinimumWidth(92)
        f_spin.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
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
            'stage2_numbering_mode': self.stage2_numbering_combo.currentData() if hasattr(self, 'stage2_numbering_combo') else DEFAULT_STAGE2_NUMBERING_MODE,
            'stage2_perspective_layout': self.stage2_layout_combo.currentData() if hasattr(self, 'stage2_layout_combo') else DEFAULT_STAGE2_LAYOUT_MODE,
            'stage2_cubemap_layout': self.cubemap_layout_combo.currentData() if hasattr(self, 'cubemap_layout_combo') else DEFAULT_STAGE2_LAYOUT_MODE,
            'output_width': self.stage2_width_spin.value(),
            'output_height': self.stage2_height_spin.value(),
            'cubemap_tile_width': self.cubemap_tile_width_spin.value(),
            'cubemap_tile_height': self.cubemap_tile_height_spin.value(),
            'cubemap_fov': 90,
            'stage2_input_dir': self._get_stage2_input_dir(),
            'skip_transform': not self.stage2_enable.isChecked(),
            'stage3_enabled': self.stage3_enable.isChecked(),
            'masking_engine': self._normalize_masking_engine(self.masking_engine_combo.currentData()),
            'mask_output_mode': self._get_mask_output_mode(),
            'yolo_model_path': self.yolo_model_path_edit.text().strip() if hasattr(self, 'yolo_model_path_edit') else '',
            'model_size': self.model_size_combo.currentData(),
            'confidence_threshold': self.confidence_spin.value(),
            'use_gpu': self.use_gpu_check.isChecked(),
            'sam3_segmenter_path': self._get_sam3_segmenter_text(),
            'sam3_model_path': self._get_sam3_model_text(),
            'sam3_image_exe_path': self._get_sam3_gui_text(),
            'sam3_feather_radius': self.sam3_feather_spin.value() if hasattr(self, 'sam3_feather_spin') else 8,
            'sam3_enable_refinement': getattr(self, 'sam3_enable_refinement', True),
            'sam3_refine_sky_only': getattr(self, 'sam3_refine_sky_only', True),
            'sam3_seam_aware_refinement': getattr(self, 'sam3_seam_aware_refinement', True),
            'sam3_edge_sharpen_strength': self.sam3_edge_sharpen_spin.value() if hasattr(self, 'sam3_edge_sharpen_spin') else 0.75,
            'sam3_prompts': {k: cb.isChecked() for k, cb in self.sam3_prompt_checks.items()} if hasattr(self, 'sam3_prompt_checks') else {},
            'sam3_custom_prompts': self.sam3_custom_prompts_edit.text().strip() if hasattr(self, 'sam3_custom_prompts_edit') else '',
            'sam3_morph_radius': self.sam3_morph_spin.value() if hasattr(self, 'sam3_morph_spin') else (self.sam3_morph_slider.value() if hasattr(self, 'sam3_morph_slider') else 0),
            'sam3_output_mode': self._get_mask_output_mode(),
            'sam3_alpha_export': self._get_mask_output_mode() in ('alpha_only', 'both'),
            'sam3_alpha_only': self._get_mask_output_mode() == 'alpha_only',
            'sam3_max_input_width': self.sam3_maxw_combo.currentData() if hasattr(self, 'sam3_maxw_combo') else 3840,
            'masking_categories': {
                'persons': self.persons_enable.isChecked(),
                'personal_objects': self.objects_enable.isChecked(),
                'animals': self.animals_enable.isChecked(),
            },
            'stage3_input_dir': self._get_stage3_input_dir(),
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

    def _get_stage3_input_dir(self) -> str:
        widget = getattr(self, 'stage3_input_dir_edit', None)
        if widget is None:
            return ''
        return widget.text().strip()

    def _set_stage3_input_dir(self, value: str):
        normalized = str(value or '').strip()
        widget = getattr(self, 'stage3_input_dir_edit', None)
        if widget is None or widget.text().strip() == normalized:
            return
        widget.blockSignals(True)
        widget.setText(normalized)
        widget.blockSignals(False)

    def _on_stage3_input_dir_changed(self, value: str):
        self._set_stage3_input_dir(value)
        self._on_stage3_preview_source_changed()

    def _get_stage3_input_path(self) -> Path | None:
        value = self._get_stage3_input_dir()
        return Path(value) if value else None

    def _get_stage2_input_dir(self) -> str:
        for attr in ('stage2_input_dir_edit', 'stage2_input_dir_edit_cubemap'):
            widget = getattr(self, attr, None)
            if widget is not None:
                value = widget.text().strip()
                if value:
                    return value
        return ''

    def _set_stage2_input_dir(self, value: str):
        normalized = str(value or '').strip()
        for attr in ('stage2_input_dir_edit', 'stage2_input_dir_edit_cubemap'):
            widget = getattr(self, attr, None)
            if widget is None:
                continue
            if widget.text().strip() == normalized:
                continue
            widget.blockSignals(True)
            widget.setText(normalized)
            widget.blockSignals(False)

    def _on_stage2_input_dir_changed(self, value: str):
        self._set_stage2_input_dir(value)
        self._on_stage3_preview_source_changed()

    def _get_stage2_input_path(self) -> Path | None:
        value = self._get_stage2_input_dir()
        return Path(value) if value else None

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

    def _browse_for_file(self, target: QLineEdit, title: str, filter_text: str):
        start_dir = str(Path(target.text()).parent) if target.text().strip() else str(Path.cwd())
        selected, _ = QFileDialog.getOpenFileName(self, title, start_dir, filter_text)
        if selected:
            target.setText(selected)

    def _browse_for_directory(self, target: QLineEdit, title: str):
        start_dir = target.text().strip() or str(Path.cwd())
        selected = QFileDialog.getExistingDirectory(self, title, start_dir)
        if selected:
            target.setText(selected)

    def _browse_sam3_segmenter(self):
        self.open_settings()

    def _browse_sam3_model(self):
        self.open_settings()

    def _browse_sam3_gui(self):
        self.open_settings()

    def _sync_preview_timestamp_from_stage1_range(self, *_args):
        if not hasattr(self, 'stage1_eq_preview'):
            return

        preview_time = 0.0 if self.full_video_check.isChecked() else self.start_time_spin.value()
        self.stage1_eq_preview.set_preview_timestamp(preview_time)

    def _on_stage1_preview_timestamp_changed(self, timestamp: float):
        if hasattr(self, 'sam3_preview_widget'):
            self.sam3_preview_widget.refresh_auto_source_image(
                force=True,
                mark_overlay_stale=True,
            )

    def _on_stage1_preview_frame_available(self):
        if hasattr(self, 'sam3_preview_widget'):
            self.sam3_preview_widget.refresh_state()
            self.sam3_preview_widget.refresh_auto_source_image(
                force=False,
                mark_overlay_stale=True,
            )

    def _on_sam3_config_changed(self, *_args):
        self._update_masking_runtime_status()
        if hasattr(self, 'sam3_preview_widget'):
            self.sam3_preview_widget.refresh_state()
            self.sam3_preview_widget.mark_stale()

    def _get_sam3_segmenter_text(self) -> str:
        segmenter = self.settings.get_sam3_segmenter_path()
        return str(segmenter) if segmenter else ''

    def _get_sam3_model_text(self) -> str:
        model = self.settings.get_sam3_model_path()
        return str(model) if model else ''

    def _get_sam3_gui_text(self) -> str:
        gui = self.settings.get_sam3_image_exe_path()
        return str(gui) if gui else ''

    def _set_sam3_paths_from_config(self, config: dict):
        if 'sam3_segmenter_path' in config:
            self.settings.set_sam3_segmenter_path(config['sam3_segmenter_path'])
        if 'sam3_model_path' in config:
            self.settings.set_sam3_model_path(config['sam3_model_path'])
        if 'sam3_image_exe_path' in config:
            self.settings.set_sam3_image_exe_path(config['sam3_image_exe_path'])

    def _mirror_combo_selection(self, source: QComboBox, target: QComboBox):
        if target is None or source is None:
            return
        value = source.currentData()
        index = target.findData(value)
        if index < 0 or index == target.currentIndex():
            return
        target.blockSignals(True)
        target.setCurrentIndex(index)
        target.blockSignals(False)

    def _normalize_masking_engine(self, engine: str | None) -> str:
        engine_value = str(engine or '').strip().lower()
        if engine_value in {'yolo', 'yolo_onnx', 'yolo_pytorch', 'hybrid'}:
            return 'yolo'
        if engine_value == 'sam3_cpp':
            return 'sam3_cpp'
        if engine_value == 'sam_vitb':
            return 'sam3_cpp'
        return 'sam3_cpp'

    def _get_mask_output_mode(self) -> str:
        if hasattr(self, 'sam3_output_mode_combo'):
            mode = str(self.sam3_output_mode_combo.currentData() or '').strip().lower()
            if mode in {'masks_only', 'alpha_only', 'both'}:
                return mode
        return 'masks_only'

    def _get_available_yolo_model_sizes(self) -> dict[str, Path]:
        model_candidates = {
            'nano': ('yolo26n-seg.onnx', 'yolov8n-seg.onnx'),
            'small': ('yolo26s-seg.onnx', 'yolov8s-seg.onnx'),
            'medium': ('yolo26m-seg.onnx', 'yolov8m-seg.onnx'),
            'large': ('yolov8l-seg.onnx',),
            'xlarge': ('yolov8x-seg.onnx',),
        }

        available: dict[str, Path] = {}
        for size, model_names in model_candidates.items():
            for model_name in model_names:
                candidate = resolve_masking_model_path(model_name)
                if candidate.exists():
                    available[size] = candidate
                    break
        return available

    def _refresh_yolo_model_size_options(self):
        if not hasattr(self, 'model_size_combo'):
            return

        available = self._get_available_yolo_model_sizes()
        torch_ready = has_usable_torch_runtime()
        model = self.model_size_combo.model()

        for index in range(self.model_size_combo.count()):
            size_key = self.model_size_combo.itemData(index)
            item = model.item(index) if hasattr(model, 'item') else None
            if item is not None:
                item.setEnabled(torch_ready or size_key in available)

        if not torch_ready and available and self.model_size_combo.currentData() not in available:
            preferred_size = 'small' if 'small' in available else next(iter(available))
            self._set_combo_data(self.model_size_combo, preferred_size)

        if torch_ready:
            tooltip = 'YOLO ONNX models are bundled when available. Additional sizes can use a local PyTorch runtime when installed.'
        elif available:
            pretty_sizes = ', '.join(size.title() for size in available)
            tooltip = f'Bundled YOLO ONNX sizes in this build: {pretty_sizes}.'
        else:
            tooltip = 'No bundled YOLO ONNX model was found for this build.'
        self.model_size_combo.setToolTip(tooltip)

    def _get_sam3_preview_config(self) -> dict:
        prompts = {}
        if hasattr(self, 'sam3_prompt_checks'):
            prompts = {k: cb.isChecked() for k, cb in self.sam3_prompt_checks.items()}
        return {
            'segment_persons_exe': self._get_sam3_segmenter_text(),
            'model_path': self._get_sam3_model_text(),
            'sam3_image_exe': self._get_sam3_gui_text(),
            'use_gpu': self.use_gpu_check.isChecked() if hasattr(self, 'use_gpu_check') else True,
            'feather_radius': self.sam3_feather_spin.value() if hasattr(self, 'sam3_feather_spin') else 8,
            'morph_radius': self.sam3_morph_spin.value() if hasattr(self, 'sam3_morph_spin') else (self.sam3_morph_slider.value() if hasattr(self, 'sam3_morph_slider') else 0),
            'enable_refinement': getattr(self, 'sam3_enable_refinement', True),
            'refine_sky_only': getattr(self, 'sam3_refine_sky_only', True),
            'seam_aware_refinement': getattr(self, 'sam3_seam_aware_refinement', True),
            'edge_sharpen_strength': self.sam3_edge_sharpen_spin.value() if hasattr(self, 'sam3_edge_sharpen_spin') else 0.75,
            'alpha_export': (self.sam3_output_mode_combo.currentData() in ('alpha_only', 'both')) if hasattr(self, 'sam3_output_mode_combo') else False,
            'max_input_width': self.sam3_maxw_combo.currentData() if hasattr(self, 'sam3_maxw_combo') else 3840,
            'sam3_prompts': prompts,
            'sam3_custom_prompts': self.sam3_custom_prompts_edit.text().strip() if hasattr(self, 'sam3_custom_prompts_edit') else '',
        }

    def _resolve_stage3_preview_candidate(self) -> Path | None:
        stage3_input_path = self._get_stage3_input_path()
        if stage3_input_path is not None:
            return self._find_first_image_in_directory(stage3_input_path)

        input_text = self.input_file_edit.text().strip() if hasattr(self, 'input_file_edit') else ''
        if input_text:
            input_path = Path(input_text)
            if input_path.is_file() and input_path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}:
                return input_path
            if input_path.is_dir():
                preview_path = self._find_first_image_in_directory(input_path)
                if preview_path is not None:
                    return preview_path

        output_text = self.output_dir_edit.text().strip() if hasattr(self, 'output_dir_edit') else ''
        if output_text:
            output_root = Path(output_text)
            image_dir = output_root if self._directory_has_images(output_root) else self._resolve_output_image_dir(output_root, self._get_stage3_image_source())
            if image_dir and image_dir.exists():
                preview_path = self._find_first_image_in_directory(image_dir)
                if preview_path is not None:
                    return preview_path

        stage2_input_path = self._get_stage2_input_path()
        if stage2_input_path is not None:
            preview_path = self._find_first_image_in_directory(stage2_input_path)
            if preview_path is not None:
                return preview_path

        if hasattr(self, 'stage1_eq_preview') and self.stage1_eq_preview is not None:
            try:
                if not self.stage1_eq_preview.has_preview_frame():
                    return None
                preview_path = self.stage1_eq_preview.export_current_preview_frame(
                    Path(tempfile.gettempdir()) / '360toolkit_sam3_preview' / 'stage1_preview_frame.png'
                )
                if preview_path and preview_path.exists():
                    return preview_path
            except Exception as exc:
                logger.debug("Failed to export Stage 1 preview frame for SAM3: %s", exc)

        return None

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

    def _find_first_image_in_directory(self, folder: Path | None) -> Path | None:
        if folder is None or not folder.exists() or not folder.is_dir():
            return None
        candidates = sorted(
            [p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}],
            key=perspective_output_sort_key,
        )
        return candidates[0] if candidates else None

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
            if 'stage2_numbering_mode' in config and hasattr(self, 'stage2_numbering_combo'):
                idx = self.stage2_numbering_combo.findData(config['stage2_numbering_mode'])
                if idx >= 0:
                    self.stage2_numbering_combo.setCurrentIndex(idx)
            if 'stage2_input_dir' in config:
                self._set_stage2_input_dir(config['stage2_input_dir'])
            if 'stage2_perspective_layout' in config and hasattr(self, 'stage2_layout_combo'):
                idx = self.stage2_layout_combo.findData(config['stage2_perspective_layout'])
                if idx >= 0:
                    self.stage2_layout_combo.setCurrentIndex(idx)
            if 'stage2_cubemap_layout' in config and hasattr(self, 'cubemap_layout_combo'):
                idx = self.cubemap_layout_combo.findData(config['stage2_cubemap_layout'])
                if idx >= 0:
                    self.cubemap_layout_combo.setCurrentIndex(idx)
            if 'stage3_enabled' in config:
                self.stage3_enable.setChecked(config['stage3_enabled'])
            if 'stage3_image_source' in config and hasattr(self, 'mask_input_source_combo'):
                self._set_combo_data(self.mask_input_source_combo, config['stage3_image_source'])
            if 'stage3_input_dir' in config:
                self._set_stage3_input_dir(config['stage3_input_dir'])
            if 'masking_engine' in config and hasattr(self, 'masking_engine_combo'):
                self._set_combo_data(self.masking_engine_combo, self._normalize_masking_engine(config['masking_engine']))
            if 'yolo_model_path' in config and hasattr(self, 'yolo_model_path_edit'):
                self.yolo_model_path_edit.setText(config.get('yolo_model_path') or '')
            if 'model_size' in config:
                models = list(YOLOV8_MODELS.keys())
                if config['model_size'] in models:
                    self.model_size_combo.setCurrentIndex(models.index(config['model_size']))
            if 'confidence_threshold' in config:
                self.confidence_spin.setValue(config['confidence_threshold'])
            if 'use_gpu' in config:
                self.use_gpu_check.setChecked(config['use_gpu'])
            self._set_sam3_paths_from_config(config)
            if 'sam3_feather_radius' in config and hasattr(self, 'sam3_feather_spin'):
                self.sam3_feather_spin.setValue(int(config['sam3_feather_radius']))
            if 'sam3_enable_refinement' in config:
                self.sam3_enable_refinement = bool(config['sam3_enable_refinement'])
            if 'sam3_refine_sky_only' in config:
                self.sam3_refine_sky_only = bool(config['sam3_refine_sky_only'])
            if 'sam3_seam_aware_refinement' in config:
                self.sam3_seam_aware_refinement = bool(config['sam3_seam_aware_refinement'])
            if 'sam3_edge_sharpen_strength' in config and hasattr(self, 'sam3_edge_sharpen_spin'):
                self.sam3_edge_sharpen_spin.setValue(float(config['sam3_edge_sharpen_strength']))
            if 'sam3_prompts' in config and hasattr(self, 'sam3_prompt_checks'):
                for k, cb in self.sam3_prompt_checks.items():
                    cb.setChecked(config['sam3_prompts'].get(k, True))
            if 'sam3_custom_prompts' in config and hasattr(self, 'sam3_custom_prompts_edit'):
                self.sam3_custom_prompts_edit.setText(config['sam3_custom_prompts'])
            if 'sam3_morph_radius' in config and hasattr(self, 'sam3_morph_slider'):
                v = int(config['sam3_morph_radius'])
                self.sam3_morph_slider.setValue(v)
                if hasattr(self, 'sam3_morph_spin'):
                    self.sam3_morph_spin.setValue(v)
            if hasattr(self, 'sam3_output_mode_combo') and ('mask_output_mode' in config or 'sam3_output_mode' in config or 'sam3_alpha_export' in config or 'sam3_alpha_only' in config):
                mode = config.get('mask_output_mode') or config.get('sam3_output_mode', '')
                if not mode:
                    # backward compat: derive from old bool flags
                    if config.get('sam3_alpha_only', False):
                        mode = 'alpha_only'
                    elif config.get('sam3_alpha_export', False):
                        mode = 'both'
                    else:
                        mode = 'masks_only'
                idx = self.sam3_output_mode_combo.findData(mode)
                if idx >= 0:
                    self.sam3_output_mode_combo.setCurrentIndex(idx)
            if 'sam3_max_input_width' in config and hasattr(self, 'sam3_maxw_combo'):
                idx = self.sam3_maxw_combo.findData(int(config['sam3_max_input_width']))
                if idx >= 0:
                    self.sam3_maxw_combo.setCurrentIndex(idx)
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

    def _on_input_file_changed(self, path: str):
        normalized_path = (path or '').strip()

        if hasattr(self, 'stage1_eq_preview'):
            self.stage1_eq_preview.set_video_path(normalized_path)

        self._on_stage3_preview_source_changed()

        suffix = Path(normalized_path).suffix.lower() if normalized_path else ''
        if suffix in {'.mp4', '.mov', '.avi', '.mkv'}:
            ffmpeg_idx = self.extraction_method_combo.findData('ffmpeg_stitched')
            if ffmpeg_idx >= 0 and self.extraction_method_combo.currentIndex() != ffmpeg_idx:
                self.extraction_method_combo.setCurrentIndex(ffmpeg_idx)

    def _on_stage3_preview_source_changed(self, *_args):
        if hasattr(self, 'sam3_preview_widget'):
            self.sam3_preview_widget.refresh_state()
            self.sam3_preview_widget.refresh_auto_source_image(
                force=False,
                mark_overlay_stale=True,
            )
    
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
        engine = self._normalize_masking_engine(self.masking_engine_combo.currentData())
        is_sam3 = engine == "sam3_cpp"
        self._refresh_yolo_model_size_options()
        if hasattr(self, 'masking_categories_card'):
            self.masking_categories_card.setVisible(not is_sam3)
        self.model_size_container.setVisible(not is_sam3)
        self.yolo_model_path_container.setVisible(not is_sam3)
        self.confidence_container.setVisible(not is_sam3)
        self.mask_output_mode_container.setVisible(True)
        self.sam3_options_container.setVisible(is_sam3)
        self.sam3_preview_card.setVisible(is_sam3)
        if is_sam3:
            self.engine_description_label.setText("Primary masking engine | extraction preview feeds SAM 3 automatically | bundled model and executable expected")
        else:
            self.confidence_spin.setEnabled(True)
            available = self._get_available_yolo_model_sizes()
            custom_model_text = self.yolo_model_path_edit.text().strip() if hasattr(self, 'yolo_model_path_edit') else ''
            if custom_model_text:
                self.engine_description_label.setText(f"Secondary masking engine | custom YOLO ONNX model | masks equirect first and reuses alpha before split")
            elif available:
                pretty_sizes = ', '.join(size.title() for size in available)
                self.engine_description_label.setText(f"Secondary masking engine | ONNX YOLO segmentation | bundled sizes: {pretty_sizes} | default PNG flow masks equirect before split")
            else:
                self.engine_description_label.setText("Secondary masking engine | ONNX YOLO segmentation | no bundled model detected in this build")
        if hasattr(self, 'sam3_preview_widget'):
            self.sam3_preview_widget.refresh_state()
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
        available_yolo_sizes = self._get_available_yolo_model_sizes()
        custom_model_text = self.yolo_model_path_edit.text().strip() if hasattr(self, 'yolo_model_path_edit') else ''
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

        if custom_model_text:
            custom_model = Path(custom_model_text).expanduser()
            if custom_model.exists() and custom_model.is_file():
                details.append(f'YOLO: custom model {custom_model.name}')
            else:
                details.append('YOLO: custom model missing')
        elif available_yolo_sizes:
            details.append('YOLO: bundled ' + ', '.join(size.title() for size in available_yolo_sizes) + ' ONNX model')
        else:
            details.append('YOLO: bundled model not found')

        details.append('PyTorch: installed' if has_usable_torch_runtime() else 'PyTorch: not bundled')
        segmenter_text = self._get_sam3_segmenter_text()
        model_text = self._get_sam3_model_text()
        segmenter = Path(segmenter_text) if segmenter_text else None
        model = Path(model_text) if model_text else None
        if segmenter and model and segmenter.exists() and model.exists():
            details.append('SAM 3: ready')
        else:
            details.append('SAM 3: external exe/model not configured')
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
        self._sync_preview_timestamp_from_stage1_range()
    
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

        self._auto_apply_insv_sdk_defaults(input_file)

        if self._input_analysis_worker and self._input_analysis_worker.isRunning():
            self._pending_input_analysis = input_file
            return

        self._start_input_analysis(input_file)

    def _start_input_analysis(self, input_file: str):
        self._active_input_analysis = input_file
        self._pending_input_analysis = None
        self.file_metadata_label.setText("Analyzing input file metadata…")
        self.file_metadata_label.setProperty("state", "warn")
        self._refresh_widget_style(self.file_metadata_label)

        worker = _InputAnalysisWorker(input_file, self)
        self._input_analysis_worker = worker
        worker.completed.connect(self._on_input_analysis_completed)
        worker.failed.connect(self._on_input_analysis_failed)
        worker.finished.connect(self._on_input_analysis_finished)
        worker.start()

    def _on_input_analysis_completed(self, input_file: str, info: dict):
        if input_file != self._active_input_analysis:
            return
        if self.input_file_edit.text().strip() != input_file:
            return

        if info.get('success'):
            camera_model = (info.get('camera_model') or '').strip()
            camera_model_source = (info.get('camera_model_source') or '').strip()
            camera_segment = f" | <b>Camera:</b> {camera_model}" if camera_model else ""
            stream_count = max(1, int(info.get('video_stream_count') or 1))
            stream_segment = f" | <b>Streams:</b> {stream_count}"
            self.file_metadata_label.setText(
                f"<b>Type:</b> {info.get('file_type_desc', 'Unknown')} | "
                f"<b>Duration:</b> {info.get('duration_formatted', 'N/A')} | "
                f"<b>Resolution:</b> {info.get('resolution', 'N/A')} | "
                f"<b>FPS:</b> {info.get('fps', 0):.2f} | "
                f"<b>Frames:</b> {info.get('frame_count', 0):,} | "
                f"<b>Size:</b> {info.get('file_size_mb', 0):.1f} MB"
                f"{stream_segment}"
                f"{camera_segment}"
            )
            self.file_metadata_label.setProperty("state", "ok")
            self._refresh_widget_style(self.file_metadata_label)
            duration = info.get('duration', 0)
            self.end_time_spin.setMaximum(duration)
            self.end_time_spin.setValue(duration)
            if hasattr(self, 'stage1_eq_preview'):
                self.stage1_eq_preview.set_preview_timestamp_maximum(duration)
                current_preview_time = self.start_time_spin.value() if not self.full_video_check.isChecked() else 0.0
                self.stage1_eq_preview.set_preview_timestamp(min(current_preview_time, duration))

            w = info.get('width', 0)
            h = info.get('height', 0)
            if w > 0 and h > 0:
                tw = ((w * 3 // 4 + 64) // 128) * 128
                th = ((h * 3 // 4 + 64) // 128) * 128
                self.cubemap_tile_width_spin.setValue(tw)
                self.cubemap_tile_height_spin.setValue(th)
            if camera_model:
                source_suffix = f" ({camera_model_source})" if camera_model_source and camera_model_source != 'unavailable' else ""
                self.log_message(f"Analyzed: {Path(input_file).name} | device={camera_model}{source_suffix}")
            else:
                self.log_message(f"Analyzed: {Path(input_file).name}")
                if Path(input_file).suffix.lower() == '.insv':
                    self.log_message("[WARN] Analyze: could not identify the source device from INSV metadata/trailer.")
        else:
            self.file_metadata_label.setText(f"Error: {info.get('error', 'Unknown')}")
            self.file_metadata_label.setProperty("state", "error")
            self._refresh_widget_style(self.file_metadata_label)

    def _on_input_analysis_failed(self, input_file: str, error: str):
        if input_file != self._active_input_analysis:
            return
        if self.input_file_edit.text().strip() != input_file:
            return
        self.file_metadata_label.setText(f"Error: {error}")
        self.file_metadata_label.setProperty("state", "error")
        self._refresh_widget_style(self.file_metadata_label)

    def _on_input_analysis_finished(self):
        self._input_analysis_worker = None
        if self._pending_input_analysis and self._pending_input_analysis != self._active_input_analysis:
            pending_input = self._pending_input_analysis
            self._pending_input_analysis = None
            QTimer.singleShot(0, lambda path=pending_input: self._start_input_analysis(path))

    def _auto_apply_insv_sdk_defaults(self, input_file: str):
        """Enable FlowState and Direction Lock defaults when a new INSV file is analyzed."""
        if not hasattr(self, 'stage1_media_panel'):
            return

        path = Path(input_file)
        if path.suffix.lower() != '.insv':
            return

        try:
            key = str(path.resolve())
        except OSError:
            key = str(path)

        if self._last_auto_sdk_defaults_key == key:
            return

        opts = self.stage1_media_panel.get_sdk_options()
        needs_change = not opts.get('enable_flowstate', False) or not opts.get('enable_direction_lock', False)
        self.stage1_media_panel.set_stabilization(True, True)
        self._last_auto_sdk_defaults_key = key

        if needs_change:
            self.log_message(f"[INFO] Analyze: enabled FlowState and Direction Lock defaults for INSV input: {path.name}")
    
    # ========================================================================
    # PIPELINE EXECUTION
    # ========================================================================
    def start_pipeline(self):
        if not hasattr(self, '_auto_advance_enabled') or not self._auto_advance_enabled:
            self._auto_advance_enabled = False
        
        input_file = self.input_file_edit.text()
        output_dir = self.output_dir_edit.text()
        stage2_input_dir = self._get_stage2_input_dir()
        stage3_input_dir = self._get_stage3_input_dir()
        
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
        elif self.stage2_enable.isChecked() and not stage2_input_dir:
            if not input_file:
                QMessageBox.warning(
                    self,
                    "Input Required",
                    "Select a Stage 2 input folder or enable Stage 1 with a video input."
                )
                self._set_control_status("Stage 2 input folder required", "warn")
                _clear_pending()
                return
            if not Path(input_file).exists():
                QMessageBox.warning(self, "Not Found", f"Input path not found:\n{input_file}")
                self._set_control_status("Input path not found", "error")
                _clear_pending()
                return
        if not output_dir:
            QMessageBox.warning(self, "Output Required", "Please select an output directory.")
            self._set_control_status("Output required", "warn")
            _clear_pending()
            return

        if stage2_input_dir:
            stage2_input_path = Path(stage2_input_dir)
            if not stage2_input_path.exists() or not stage2_input_path.is_dir():
                QMessageBox.warning(self, "Stage 2 Input Missing", f"Stage 2 input folder not found:\n{stage2_input_path}")
                self._set_control_status("Stage 2 input folder not found", "error")
                _clear_pending()
                return
            if not self._directory_has_images(stage2_input_path):
                QMessageBox.warning(self, "Stage 2 Input Empty", f"No images found in Stage 2 input folder:\n{stage2_input_path}")
                self._set_control_status("No images in Stage 2 input folder", "warn")
                _clear_pending()
                return

        if stage3_input_dir and self.stage3_enable.isChecked():
            stage3_input_path = Path(stage3_input_dir)
            if not stage3_input_path.exists() or not stage3_input_path.is_dir():
                QMessageBox.warning(self, "Stage 3 Input Missing", f"Stage 3 input folder not found:\n{stage3_input_path}")
                self._set_control_status("Stage 3 input folder not found", "error")
                _clear_pending()
                return
            if not self._directory_has_images(stage3_input_path):
                QMessageBox.warning(self, "Stage 3 Input Empty", f"No images found in Stage 3 input folder:\n{stage3_input_path}")
                self._set_control_status("No images in Stage 3 input folder", "warn")
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
            'stage2_input_dir': stage2_input_dir,
            'stage3_input_dir': stage3_input_dir,
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
            'stage2_numbering_mode': self.stage2_numbering_combo.currentData() if hasattr(self, 'stage2_numbering_combo') else DEFAULT_STAGE2_NUMBERING_MODE,
            'stage2_perspective_layout': self.stage2_layout_combo.currentData() if hasattr(self, 'stage2_layout_combo') else DEFAULT_STAGE2_LAYOUT_MODE,
            'stage2_cubemap_layout': self.cubemap_layout_combo.currentData() if hasattr(self, 'cubemap_layout_combo') else DEFAULT_STAGE2_LAYOUT_MODE,
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
                'masking_engine': self._normalize_masking_engine(self.masking_engine_combo.currentData()),
                'mask_output_mode': self._get_mask_output_mode(),
                'yolo_model_path': self.yolo_model_path_edit.text().strip() if hasattr(self, 'yolo_model_path_edit') else '',
                'model_size': self.model_size_combo.currentData(),
                'confidence_threshold': self.confidence_spin.value(),
                'use_gpu': self.use_gpu_check.isChecked(),
                'sam3_segmenter_path': self._get_sam3_segmenter_text(),
                'sam3_model_path': self._get_sam3_model_text(),
                'sam3_image_exe_path': self._get_sam3_gui_text(),
                'sam3_feather_radius': self.sam3_feather_spin.value() if hasattr(self, 'sam3_feather_spin') else 8,
                'sam3_enable_refinement': getattr(self, 'sam3_enable_refinement', True),
                'sam3_refine_sky_only': getattr(self, 'sam3_refine_sky_only', True),
                'sam3_seam_aware_refinement': getattr(self, 'sam3_seam_aware_refinement', True),
                'sam3_edge_sharpen_strength': self.sam3_edge_sharpen_spin.value() if hasattr(self, 'sam3_edge_sharpen_spin') else 0.75,
                'sam3_morph_radius': self.sam3_morph_spin.value() if hasattr(self, 'sam3_morph_spin') else (self.sam3_morph_slider.value() if hasattr(self, 'sam3_morph_slider') else 0),
                'sam3_output_mode': self._get_mask_output_mode(),
                'sam3_alpha_export': self._get_mask_output_mode() in ('alpha_only', 'both'),
                'sam3_alpha_only': self._get_mask_output_mode() == 'alpha_only',
                'sam3_max_input_width': self.sam3_maxw_combo.currentData() if hasattr(self, 'sam3_maxw_combo') else 3840,
                'sam3_score_threshold': self.sam3_score_spin.value() if hasattr(self, 'sam3_score_spin') else 0.5,
                'sam3_nms_threshold': self.sam3_nms_spin.value() if hasattr(self, 'sam3_nms_spin') else 0.1,
                'sam3_prompts': {k: cb.isChecked() for k, cb in self.sam3_prompt_checks.items()} if hasattr(self, 'sam3_prompt_checks') else {},
                'sam3_custom_prompts': self.sam3_custom_prompts_edit.text().strip() if hasattr(self, 'sam3_custom_prompts_edit') else '',
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

        folder = None
        stage2_input_dir = self._get_stage2_input_dir()
        if stage2_input_dir:
            folder = Path(stage2_input_dir)
        else:
            from src.pipeline.batch_orchestrator import PipelineWorker
            worker = PipelineWorker({})
            folder = worker.discover_stage_input_folder(stage=2, output_dir=output_dir)
            if not folder:
                folder_str = QFileDialog.getExistingDirectory(self, "Select Extraction Output", str(Path(output_dir)))
                if not folder_str:
                    return
                folder = Path(folder_str)
                self._set_stage2_input_dir(str(folder))

        if not folder.exists() or not folder.is_dir():
            QMessageBox.warning(self, "Stage 2 Input Missing", f"Stage 2 input folder not found:\n{folder}")
            return
        
        images = []
        for ext in ['*.png', '*.PNG', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.tif', '*.TIF', '*.tiff', '*.TIFF', '*.bmp', '*.BMP']:
            images.extend(folder.rglob(ext))
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
            return p.is_dir() and any(f for ext in _img_exts for f in p.rglob(ext))

        # If output_root itself is already an images folder, use it directly
        # (happens when user sets output dir directly to perspective_views/extracted_frames)
        folder = None
        stage3_input_dir = self._get_stage3_input_dir()
        if stage3_input_dir:
            folder = Path(stage3_input_dir)
        elif _has_images(output_root):
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
            self._set_stage3_input_dir(str(folder))

        images = [f for ext in _img_exts for f in folder.rglob(ext)]
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
            if stage_number == 2 and results.get('processing_backend'):
                self.log_message(f"[INFO] Stage 2 backend: {results.get('processing_backend')}")
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
