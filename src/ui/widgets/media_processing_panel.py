"""
MediaProcessingPanel — SDK Color & View Orientation controls.

Matches Insta360 Studio's "Media Processing" panel layout.
Contains: View Orientation (Yaw / Pitch / Roll), Stabilization
(FlowState + Direction Lock), and Color Optimization sliders.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QSpinBox, QPushButton, QCheckBox, QFrame, QToolButton,
    QGroupBox, QSizePolicy,
)
from PyQt6.QtCore import Qt, pyqtSignal


# ──────────────────────────────────────────────────────────────────────────────
# Primitive building blocks
# ──────────────────────────────────────────────────────────────────────────────

class _SliderRow(QWidget):
    """Labeled horizontal slider with a linked spinbox and a reset button."""

    value_changed = pyqtSignal(int)

    def __init__(self, label: str, lo: int, hi: int, default: int = 0, parent=None):
        super().__init__(parent)
        self._default = default

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 1, 0, 1)
        layout.setSpacing(8)

        lbl = QLabel(label)
        lbl.setFixedWidth(82)
        lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        lbl.setProperty("role", "secondary")
        layout.addWidget(lbl)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(lo, hi)
        self._slider.setValue(default)
        self._slider.setTickPosition(QSlider.TickPosition.NoTicks)
        layout.addWidget(self._slider, stretch=1)

        self._spin = QSpinBox()
        self._spin.setRange(lo, hi)
        self._spin.setValue(default)
        self._spin.setFixedWidth(52)
        self._spin.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        layout.addWidget(self._spin)

        reset_btn = QToolButton()
        reset_btn.setText("↺")
        reset_btn.setFixedSize(20, 20)
        reset_btn.setToolTip(f"Reset to {default}")
        reset_btn.clicked.connect(self.reset)
        layout.addWidget(reset_btn)

        # Bidirectional sync (block re-signals to avoid double-fire)
        self._slider.valueChanged.connect(self._on_slider)
        self._spin.valueChanged.connect(self._on_spin)

    def _on_slider(self, v: int):
        self._spin.blockSignals(True)
        self._spin.setValue(v)
        self._spin.blockSignals(False)
        self.value_changed.emit(v)

    def _on_spin(self, v: int):
        self._slider.blockSignals(True)
        self._slider.setValue(v)
        self._slider.blockSignals(False)
        self.value_changed.emit(v)

    def reset(self):
        self._slider.setValue(self._default)

    def value(self) -> int:
        return self._slider.value()

    def setValue(self, v: int):
        self._slider.blockSignals(True)
        self._spin.blockSignals(True)
        self._slider.setValue(v)
        self._spin.setValue(v)
        self._slider.blockSignals(False)
        self._spin.blockSignals(False)


class _ToggleRow(QWidget):
    """A full-width row with a label on the left and a checkbox on the right."""

    toggled = pyqtSignal(bool)

    def __init__(self, label: str, subtitle: str = "", parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(8)

        col = QVBoxLayout()
        col.setSpacing(0)
        lbl = QLabel(label)
        col.addWidget(lbl)
        if subtitle:
            sub = QLabel(subtitle)
            sub.setProperty("role", "muted")
            f = sub.font()
            f.setPointSize(max(f.pointSize() - 1, 7))
            sub.setFont(f)
            col.addWidget(sub)
        layout.addLayout(col, stretch=1)

        self._check = QCheckBox()
        layout.addWidget(self._check)
        self._check.toggled.connect(self.toggled)

    def isChecked(self) -> bool:
        return self._check.isChecked()

    def setChecked(self, v: bool):
        self._check.setChecked(v)


def _h_sep() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    line.setProperty("role", "separator")
    return line


# ──────────────────────────────────────────────────────────────────────────────
# Main panel
# ──────────────────────────────────────────────────────────────────────────────

class MediaProcessingPanel(QWidget):
    """
    Media Processing controls panel matching Insta360 Studio's interface.

    Sections
    --------
    1. View Orientation  — Yaw, Pitch, Roll (for preview overlay & metadata)
    2. Stabilization     — FlowState, Direction Lock
    3. Color Optimization — Color Plus + 11 adjustment sliders
    """

    values_changed = pyqtSignal(dict)          # Full sdk_options dict
    view_changed   = pyqtSignal(float, float, float)  # yaw, pitch, roll

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(8)

        # ── 1. View Orientation ──────────────────────────────────────────
        view_box = self._section("View Orientation")
        self._yaw = _SliderRow("Yaw",   -180, 180, -180)  # -180 = left edge (operator at center)
        view_box.layout().addWidget(self._yaw)
        reset_view_btn = QPushButton("Reset View")
        reset_view_btn.setFixedWidth(90)
        reset_view_btn.clicked.connect(self.reset_view)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(reset_view_btn)
        view_box.layout().addLayout(btn_row)
        outer.addWidget(view_box)

        # ── 2. Stabilization ─────────────────────────────────────────────
        stab_box = self._section("Stabilization")
        self._flowstate     = _ToggleRow("FlowState",      "Gyroscope-based video stabilization")
        self._directionlock = _ToggleRow("Direction Lock", "Locks horizon to front — requires FlowState")
        stab_box.layout().addWidget(self._flowstate)
        stab_box.layout().addWidget(self._directionlock)
        outer.addWidget(stab_box)

        # ── 3. Color Optimization ─────────────────────────────────────────
        color_box = self._section("Color Optimization")
        self._colorplus = _ToggleRow("Color Plus", "AI-based color enhancement")
        color_box.layout().addWidget(self._colorplus)
        color_box.layout().addWidget(_h_sep())

        self._exposure   = _SliderRow("Exposure",    -100, 100, 0)
        self._highlights = _SliderRow("Highlights",  -100, 100, 0)
        self._shadows    = _SliderRow("Shadow",      -100, 100, 0)
        self._contrast   = _SliderRow("Contrast",    -100, 100, 0)
        self._brightness = _SliderRow("Brightness",  -100, 100, 0)
        self._blackpoint = _SliderRow("Black Point", -100, 100, 0)
        self._saturation = _SliderRow("Saturation",  -100, 100, 0)
        self._vibrance   = _SliderRow("Vibrance",    -100, 100, 0)
        self._warmth     = _SliderRow("Temp",        -100, 100, 0)
        self._tint       = _SliderRow("Tint",        -100, 100, 0)
        self._definition = _SliderRow("Definition",    0,  100, 0)

        self._color_sliders = (
            self._exposure, self._highlights, self._shadows, self._contrast,
            self._brightness, self._blackpoint, self._saturation, self._vibrance,
            self._warmth, self._tint, self._definition,
        )
        for row in self._color_sliders:
            color_box.layout().addWidget(row)

        reset_color_btn = QPushButton("Reset All Colors")
        reset_color_btn.setFixedWidth(126)
        reset_color_btn.clicked.connect(self.reset_colors)
        btn_row2 = QHBoxLayout()
        btn_row2.addStretch()
        btn_row2.addWidget(reset_color_btn)
        color_box.layout().addLayout(btn_row2)
        outer.addWidget(color_box)

        outer.addStretch()

        # ── Wire all signals ──────────────────────────────────────────────
        for w in (self._yaw,
                  self._exposure, self._highlights, self._shadows, self._contrast,
                  self._brightness, self._blackpoint, self._saturation, self._vibrance,
                  self._warmth, self._tint, self._definition):
            w.value_changed.connect(self._emit_values)

        for w in (self._flowstate, self._directionlock, self._colorplus):
            w.toggled.connect(self._emit_values)

        self._yaw.value_changed.connect(self._emit_view)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _section(title: str) -> QGroupBox:
        box = QGroupBox(title)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(3)
        return box

    def _emit_values(self, *_args):
        self.values_changed.emit(self.get_sdk_options() or {})

    def _emit_view(self, *_args):
        self.view_changed.emit(
            float(self._yaw.value()),
            0.0,
            0.0,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset_view(self):
        self._yaw.reset()

    def reset_colors(self):
        for w in self._color_sliders:
            w.reset()

    def get_sdk_options(self) -> dict:
        """
        Return a complete dict of all user-controlled options for SDKExtractor.
        Always includes the three toggle booleans so explicit False values
        override quality-preset defaults (e.g. 'best' preset enables colorplus
        by default — the user may have turned it off).
        """
        opts: dict = {}

        # Always include toggle states — False must override preset defaults
        opts["enable_flowstate"]      = self._flowstate.isChecked()
        opts["enable_direction_lock"] = self._directionlock.isChecked()
        opts["enable_colorplus"]      = self._colorplus.isChecked()

        # Always include yaw so extracted frames are rolled to match preview
        opts["yaw"] = float(self._yaw.value())

        color_map = {
            "exposure":    self._exposure,
            "highlights":  self._highlights,
            "shadows":     self._shadows,
            "contrast":    self._contrast,
            "brightness":  self._brightness,
            "blackpoint":  self._blackpoint,
            "saturation":  self._saturation,
            "vibrance":    self._vibrance,
            "warmth":      self._warmth,
            "tint":        self._tint,
            "definition":  self._definition,
        }
        for key, widget in color_map.items():
            v = widget.value()
            if v != 0:
                opts[key] = v

        return opts

    def get_view_orientation(self) -> tuple:
        """Return (yaw, pitch, roll) as floats — only yaw is used."""
        return float(self._yaw.value()), 0.0, 0.0

    def set_values(self, sdk_options: dict):
        """Restore panel state from a saved sdk_options dict."""
        if not sdk_options:
            return
        self._flowstate.setChecked(sdk_options.get("enable_flowstate", False))
        self._directionlock.setChecked(sdk_options.get("enable_direction_lock", False))
        self._colorplus.setChecked(sdk_options.get("enable_colorplus", False))
        self._yaw.setValue(sdk_options.get("yaw", -180))
        map_ = {
            "exposure":    self._exposure,   "highlights": self._highlights,
            "shadows":     self._shadows,    "contrast":   self._contrast,
            "brightness":  self._brightness, "blackpoint": self._blackpoint,
            "saturation":  self._saturation, "vibrance":   self._vibrance,
            "warmth":      self._warmth,     "tint":       self._tint,
            "definition":  self._definition,
        }
        for key, widget in map_.items():
            widget.setValue(sdk_options.get(key, 0))

    def set_stabilization(self, flowstate: bool, direction_lock: bool):
        """Programmatically set the stabilization toggles."""
        self._flowstate.setChecked(flowstate)
        self._directionlock.setChecked(direction_lock)
