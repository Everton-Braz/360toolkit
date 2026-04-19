from __future__ import annotations

import logging
from pathlib import Path

from PyQt6.QtCore import Qt, QPoint, QThread, pyqtSignal
from PyQt6.QtGui import QCursor, QPixmap, QWheelEvent
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ...masking.sam3_external_masker import SAM3ExternalMasker

logger = logging.getLogger(__name__)


class _ZoomableLabel(QLabel):
    """QLabel with mouse-wheel zoom and click-drag panning signals."""
    zoom_requested = pyqtSignal(int)       # +1 zoom in, -1 zoom out
    pan_dragged    = pyqtSignal(int, int)  # delta_x, delta_y in screen pixels

    def __init__(self, text: str = '', parent=None):
        super().__init__(text, parent)
        self.setMouseTracking(True)
        self._drag_last: QPoint | None = None

    def wheelEvent(self, event: QWheelEvent) -> None:  # type: ignore[override]
        delta = event.angleDelta().y()
        if delta != 0:
            self.zoom_requested.emit(1 if delta > 0 else -1)
        event.accept()

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_last = event.position().toPoint()
            self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if self._drag_last is not None and (event.buttons() & Qt.MouseButton.LeftButton):
            pos   = event.position().toPoint()
            delta = pos - self._drag_last
            self._drag_last = pos
            self.pan_dragged.emit(delta.x(), delta.y())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_last = None
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        super().mouseReleaseEvent(event)


class _SAM3PreviewWorker(QThread):
    completed = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, config: dict, image_path: str):
        super().__init__()
        self._config = dict(config)
        self._image_path = image_path

    def run(self):
        try:
            masker = SAM3ExternalMasker(
                segment_persons_exe=self._config['segment_persons_exe'],
                model_path=self._config['model_path'],
                sam3_image_exe=self._config.get('sam3_image_exe') or None,
                use_gpu=self._config.get('use_gpu', True),
                feather_radius=self._config.get('feather_radius', 8),
                morph_radius=self._config.get('morph_radius', 0),
                alpha_export=self._config.get('alpha_export', False),
                max_input_width=self._config.get('max_input_width', 3840),
                enable_refinement=self._config.get('enable_refinement', True),
                refine_sky_only=self._config.get('refine_sky_only', True),
                seam_aware_refinement=self._config.get('seam_aware_refinement', True),
                edge_sharpen_strength=self._config.get('edge_sharpen_strength', 0.75),
            )
            masker.set_enabled_categories(self._config.get('sam3_prompts', self._config.get('categories', {})))
            masker.set_custom_prompts(self._config.get('sam3_custom_prompts', ''))
            result = masker.generate_preview_assets(Path(self._image_path))
            self.completed.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class SAM3PreviewWidget(QWidget):
    message_emitted = pyqtSignal(str)

    _ZOOM_MIN  = 25    # slider value = 25%
    _ZOOM_MAX  = 800   # slider value = 800%
    _ZOOM_STEP = 15    # wheel / button step in %

    def __init__(self, parent=None):
        super().__init__(parent)
        self._config_provider = None
        self._auto_image_resolver = None
        self._worker = None
        self._preview_has_run = False
        self._using_auto_image = False
        self._zoom_factor: float = 1.0
        self._pan_x: int = 0   # crop offset from center in zoomed px (shared by both panels)
        self._pan_y: int = 0
        self._orig_pixmaps: dict[int, QPixmap] = {}   # keyed by id(label)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        image_row = QHBoxLayout()
        self.preview_image_edit = QLineEdit()
        self.preview_image_edit.setPlaceholderText('Preview image path or leave empty for auto resolution')
        image_row.addWidget(self.preview_image_edit, 1)

        browse_button = QPushButton('Browse…')
        browse_button.clicked.connect(self._browse_image)
        image_row.addWidget(browse_button)

        self.auto_button = QPushButton('Use Auto Preview')
        self.auto_button.clicked.connect(self._use_auto_image)
        image_row.addWidget(self.auto_button)
        layout.addLayout(image_row)

        button_row = QHBoxLayout()
        self.preview_button = QPushButton('Run SAM3 Preview')
        self.preview_button.clicked.connect(self.run_preview)
        button_row.addWidget(self.preview_button)

        self.launch_gui_button = QPushButton('Launch Interactive GUI')
        self.launch_gui_button.clicked.connect(self.launch_interactive_gui)
        button_row.addWidget(self.launch_gui_button)

        button_row.addSpacing(16)

        # Zoom slider
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(self._ZOOM_MIN, self._ZOOM_MAX)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setSingleStep(self._ZOOM_STEP)
        self.zoom_slider.setPageStep(50)
        self.zoom_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.zoom_slider.setTickInterval(100)
        self.zoom_slider.setFixedWidth(200)
        self.zoom_slider.setToolTip('Zoom (scroll wheel on image also works)')
        self.zoom_slider.valueChanged.connect(self._on_slider_changed)
        button_row.addWidget(self.zoom_slider)

        self.zoom_label = QLabel('100%')
        self.zoom_label.setFixedWidth(46)
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        button_row.addWidget(self.zoom_label)

        zoom_reset_btn = QPushButton('Reset')
        zoom_reset_btn.setFixedWidth(48)
        zoom_reset_btn.setToolTip('Reset zoom and pan')
        zoom_reset_btn.clicked.connect(self._zoom_reset)
        button_row.addWidget(zoom_reset_btn)

        button_row.addStretch()
        layout.addLayout(button_row)

        hint = QLabel('Drag image to pan  |  Scroll wheel or slider to zoom')
        hint.setProperty('role', 'mutedSmall')
        layout.addWidget(hint)

        self.status_label = QLabel('SAM3 resolves a preview from the Stage 3 input folder, current images, output folders, or the Extraction preview when available.')
        self.status_label.setWordWrap(True)
        self.status_label.setProperty('role', 'mutedSmall')
        layout.addWidget(self.status_label)

        preview_row = QHBoxLayout()
        preview_row.setSpacing(12)
        self.original_label = self._create_image_panel('Source Image')
        self.overlay_label = self._create_image_panel('Mask Overlay')
        preview_row.addWidget(self.original_label, 1)
        preview_row.addWidget(self.overlay_label, 1)
        layout.addLayout(preview_row)

    def _create_image_panel(self, title: str) -> _ZoomableLabel:
        panel = _ZoomableLabel(title)
        panel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        panel.setMinimumSize(280, 200)
        panel.setFrameShape(QFrame.Shape.Box)
        panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        panel.setStyleSheet('background: #1f1f1f; border: 1px solid #3a3a3a;')
        panel.zoom_requested.connect(self._on_label_zoom)
        panel.pan_dragged.connect(self._on_pan_dragged)
        return panel

    def _on_label_zoom(self, direction: int) -> None:
        new_val = max(self._ZOOM_MIN, min(self._ZOOM_MAX,
                      self.zoom_slider.value() + direction * self._ZOOM_STEP))
        self.zoom_slider.setValue(new_val)

    def _on_slider_changed(self, value: int) -> None:
        self._zoom_factor = value / 100.0
        if self._zoom_factor <= 1.0:
            self._pan_x = 0
            self._pan_y = 0
        self.zoom_label.setText(f'{value}%')
        self._redraw_all()

    def _on_pan_dragged(self, dx: int, dy: int) -> None:
        if self._zoom_factor <= 1.0:
            return
        self._pan_x -= dx
        self._pan_y -= dy
        self._redraw_all()

    def _zoom_reset(self) -> None:
        self._pan_x = 0
        self._pan_y = 0
        self.zoom_slider.setValue(100)

    def _redraw_all(self) -> None:
        for label in (self.original_label, self.overlay_label):
            orig = self._orig_pixmaps.get(id(label))
            if orig and not orig.isNull():
                self._render_zoomed(label, orig)

    def _apply_zoom(self) -> None:
        self._redraw_all()

    def _render_zoomed(self, label: QLabel, orig: QPixmap) -> None:
        """Render orig into label at current zoom and pan."""
        w = label.width()
        h = label.height()
        if w < 1 or h < 1:
            return

        if self._zoom_factor <= 1.0:
            # Fit-to-label, no pan needed
            scaled = orig.scaled(
                w, h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        else:
            # 1. Compute the base fitted size (what the image looks like at 100% zoom)
            base = orig.scaled(
                w, h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            # 2. Scale that by zoom factor
            new_w = int(base.width()  * self._zoom_factor)
            new_h = int(base.height() * self._zoom_factor)
            zoomed = base.scaled(
                new_w, new_h,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            # 3. Compute crop top-left: start from center, add pan offset
            #    _pan_x > 0 = shift window right (shows content to the right of center)
            cx = (zoomed.width()  - w) // 2 + self._pan_x
            cy = (zoomed.height() - h) // 2 + self._pan_y
            # 4. Clamp so we never go out of bounds
            max_cx = max(0, zoomed.width()  - w)
            max_cy = max(0, zoomed.height() - h)
            cx = max(0, min(cx, max_cx))
            cy = max(0, min(cy, max_cy))
            # 5. Keep clamped pan in sync so further drag feels correct
            self._pan_x = cx - (zoomed.width()  - w) // 2
            self._pan_y = cy - (zoomed.height() - h) // 2

            crop_w = min(w, zoomed.width())
            crop_h = min(h, zoomed.height())
            scaled = zoomed.copy(cx, cy, crop_w, crop_h)

        label.setPixmap(scaled)
        label.setText('')

    # ── Config / state ────────────────────────────────────────────────────────

    def set_config_provider(self, provider) -> None:
        self._config_provider = provider
        self.refresh_state()

    def set_auto_image_resolver(self, resolver) -> None:
        self._auto_image_resolver = resolver
        self.refresh_state()

    def refresh_state(self) -> None:
        config = self._get_config()
        segmenter = Path(config.get('segment_persons_exe', '')) if config.get('segment_persons_exe') else None
        model = Path(config.get('model_path', '')) if config.get('model_path') else None
        gui = Path(config.get('sam3_image_exe', '')) if config.get('sam3_image_exe') else None
        configured = bool(segmenter and segmenter.exists() and model and model.exists())
        self.preview_button.setEnabled(configured)
        self.launch_gui_button.setEnabled(bool(configured and gui and gui.exists()))
        current_path = self.preview_image_edit.text().strip()
        if not current_path or not Path(current_path).exists():
            self.refresh_auto_source_image(force=not current_path, mark_overlay_stale=False)

    def refresh_auto_source_image(self, *, force: bool = False, mark_overlay_stale: bool = True) -> bool:
        if not callable(self._auto_image_resolver):
            return False

        current_path = self.preview_image_edit.text().strip()
        if not force and current_path and not self._using_auto_image:
            return False

        resolved = self._auto_image_resolver()
        if not resolved or not Path(resolved).exists():
            return False

        resolved_path = Path(resolved)
        self.preview_image_edit.setText(str(resolved_path))
        self._using_auto_image = True
        self._set_image(self.original_label, resolved_path)

        if mark_overlay_stale:
            self._clear_image(self.overlay_label, 'Mask Overlay')
            self._preview_has_run = False
            self._set_status(f'Source image refreshed: {resolved_path.name} — click Run SAM3 Preview to update.')

        return True

    def _get_config(self) -> dict:
        if callable(self._config_provider):
            return self._config_provider() or {}
        return {}

    def _browse_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            'Select Preview Image',
            '',
            'Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)',
        )
        if path:
            self.preview_image_edit.setText(path)
            self._using_auto_image = False
            self._set_image(self.original_label, Path(path))

    def _use_auto_image(self) -> None:
        if not callable(self._auto_image_resolver):
            self._set_status('No automatic preview image resolver is configured.', error=True)
            return
        refreshed = self.refresh_auto_source_image(force=True, mark_overlay_stale=False)
        if not refreshed:
            self._set_status('No preview image could be resolved from the current input/output settings.', error=True)
            return
        image_path = Path(self.preview_image_edit.text().strip())
        self._set_status(f'Using resolved preview image: {image_path.name}')

    def _set_status(self, text: str, error: bool = False) -> None:
        self.status_label.setText(text)
        self.status_label.setProperty('role', 'danger' if error else 'mutedSmall')
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)
        self.message_emitted.emit(text)

    def _set_image(self, label: QLabel, image_path: Path) -> None:
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            label.setText(f'Failed to load\n{image_path.name}')
            return
        self._orig_pixmaps[id(label)] = pixmap
        self._render_zoomed(label, pixmap)

    def _clear_image(self, label: QLabel, text: str) -> None:
        self._orig_pixmaps.pop(id(label), None)
        label.clear()
        label.setText(text)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        for label in (self.original_label, self.overlay_label):
            orig = self._orig_pixmaps.get(id(label))
            if orig and not orig.isNull():
                self._render_zoomed(label, orig)

    def run_preview(self) -> None:
        config = self._get_config()
        image_path = self.preview_image_edit.text().strip()
        if callable(self._auto_image_resolver) and (self._using_auto_image or not image_path or not Path(image_path).exists()):
            if self.refresh_auto_source_image(force=self._using_auto_image or not image_path, mark_overlay_stale=False):
                image_path = self.preview_image_edit.text().strip()
        if not image_path:
            self._set_status('Select a preview image first.', error=True)
            return
        if not Path(image_path).exists():
            self._set_status('Preview image path does not exist.', error=True)
            return
        if not config.get('segment_persons_exe') or not config.get('model_path'):
            self._set_status('SAM3.cpp executable and model must be configured first.', error=True)
            return

        self.preview_button.setEnabled(False)
        self._set_image(self.original_label, Path(image_path))
        self._set_status('Running SAM3.cpp preview...')

        self._worker = _SAM3PreviewWorker(config, image_path)
        self._worker.completed.connect(self._on_preview_completed)
        self._worker.failed.connect(self._on_preview_failed)
        self._worker.finished.connect(self._on_preview_finished)
        self._worker.start()

    def mark_stale(self) -> None:
        """Notify the user that settings changed and preview needs re-run."""
        if self._preview_has_run:
            self._set_status('Settings changed — click Run SAM3 Preview to update.')

    def _on_preview_completed(self, result: dict) -> None:
        overlay_path = Path(result.get('overlay_path', '')) if result.get('overlay_path') else None
        if overlay_path and overlay_path.exists():
            self._set_image(self.overlay_label, overlay_path)
        self._preview_has_run = True
        self._set_status(f"SAM3.cpp preview ready for {Path(result['image_path']).name}")

    def _on_preview_failed(self, error: str) -> None:
        self._set_status(error, error=True)

    def _on_preview_finished(self) -> None:
        self.preview_button.setEnabled(True)
        self.refresh_state()
        self._worker = None

    def launch_interactive_gui(self) -> None:
        config = self._get_config()
        image_path = self.preview_image_edit.text().strip()
        if not image_path or not Path(image_path).exists():
            QMessageBox.warning(self, 'Missing Image', 'Select a preview image first.')
            return
        if not config.get('sam3_image_exe'):
            QMessageBox.warning(self, 'Missing GUI', 'Configure sam3_image.exe first.')
            return

        try:
            masker = SAM3ExternalMasker(
                segment_persons_exe=config['segment_persons_exe'],
                model_path=config['model_path'],
                sam3_image_exe=config['sam3_image_exe'],
                use_gpu=config.get('use_gpu', True),
                feather_radius=config.get('feather_radius', 8),
                enable_refinement=config.get('enable_refinement', True),
                refine_sky_only=config.get('refine_sky_only', True),
                seam_aware_refinement=config.get('seam_aware_refinement', True),
                edge_sharpen_strength=config.get('edge_sharpen_strength', 0.75),
            )
            masker.launch_interactive_gui(Path(image_path))
            self._set_status(f'Launched SAM3 interactive GUI for {Path(image_path).name}')
        except Exception as exc:
            logger.error('Failed to launch SAM3 interactive GUI: %s', exc, exc_info=True)
            QMessageBox.critical(self, 'SAM3 Launch Failed', str(exc))