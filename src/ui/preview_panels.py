"""
Perspective Preview Panel Widget
Real-time preview for perspective split with circular compass.
"""

import cv2
import subprocess
import numpy as np
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QSlider, QSpinBox, QPushButton, QGroupBox, QComboBox,
    QSizePolicy, QFileDialog, QToolButton,
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize, QPoint, QRect, QThread, QTimer
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QBrush, QFont

from ..transforms import E2PTransform


class CircularCompassWidget(QWidget):
    """
    Interactive circular compass showing camera positions.
    Clickable slices represent camera directions.
    """
    
    camera_clicked = pyqtSignal(int)  # Camera index
    
    # Camera states
    STATE_EXPORT = 0    # Blue
    STATE_PREVIEW = 1   # Yellow
    STATE_DISABLED = 2  # Red
    STATE_MASK = 3      # Green
    
    STATE_COLORS = {
        STATE_EXPORT: QColor(74, 158, 255),    # Blue
        STATE_PREVIEW: QColor(255, 215, 0),    # Gold
        STATE_DISABLED: QColor(220, 20, 60),   # Crimson
        STATE_MASK: QColor(50, 205, 50)        # Green
    }
    
    def __init__(self, camera_count: int = 8, parent=None):
        super().__init__(parent)
        
        self.camera_count = camera_count
        self.camera_states = [self.STATE_EXPORT] * camera_count
        self.selected_camera = 0
        
        self.setFixedSize(400, 400)
        self.setToolTip("Click slices to select camera\nClick icons to cycle states")
    
    def set_camera_count(self, count: int):
        """Update number of cameras"""
        self.camera_count = count
        self.camera_states = [self.STATE_EXPORT] * count
        self.selected_camera = 0
        self.update()
    
    def set_camera_state(self, index: int, state: int):
        """Set state of specific camera"""
        if 0 <= index < self.camera_count:
            self.camera_states[index] = state
            self.update()
    
    def get_camera_state(self, index: int) -> int:
        """Get state of specific camera"""
        if 0 <= index < self.camera_count:
            return self.camera_states[index]
        return self.STATE_EXPORT
    
    def cycle_camera_state(self, index: int):
        """Cycle through camera states"""
        if 0 <= index < self.camera_count:
            current = self.camera_states[index]
            # Cycle: Export -> Preview -> Disabled -> Mask -> Export
            self.camera_states[index] = (current + 1) % 4
            self.update()
    
    def paintEvent(self, event):
        """Draw the circular compass"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        center_x = self.width() // 2
        center_y = self.height() // 2
        radius = min(center_x, center_y) - 40
        
        # Draw background circle
        painter.setBrush(QBrush(QColor(60, 60, 60)))
        painter.setPen(QPen(QColor(85, 85, 85), 2))
        painter.drawEllipse(QPoint(center_x, center_y), radius, radius)
        
        # Draw pizza slices
        angle_per_camera = 360 / self.camera_count
        
        for i in range(self.camera_count):
            start_angle = int(i * angle_per_camera - 90)  # Start from top
            
            # Get color based on state
            color = self.STATE_COLORS[self.camera_states[i]]
            
            # Highlight selected camera
            if i == self.selected_camera:
                painter.setBrush(QBrush(color.lighter(130)))
                painter.setPen(QPen(Qt.GlobalColor.white, 3))
            else:
                painter.setBrush(QBrush(color))
                painter.setPen(QPen(QColor(85, 85, 85), 1))
            
            # Draw slice
            painter.drawPie(
                QRect(center_x - radius, center_y - radius, radius * 2, radius * 2),
                start_angle * 16,  # Qt uses 1/16th degree units
                int(angle_per_camera * 16)
            )
            
            # Draw camera icon (white circle)
            angle_mid = (start_angle + angle_per_camera / 2) * np.pi / 180
            icon_x = center_x + int((radius * 0.7) * np.cos(angle_mid))
            icon_y = center_y + int((radius * 0.7) * np.sin(angle_mid))
            
            painter.setBrush(QBrush(Qt.GlobalColor.white))
            painter.setPen(QPen(Qt.GlobalColor.black, 2))
            painter.drawEllipse(QPoint(icon_x, icon_y), 12, 12)
            
            # Draw camera number
            painter.setPen(QPen(Qt.GlobalColor.black))
            painter.drawText(QRect(icon_x - 10, icon_y - 10, 20, 20), Qt.AlignmentFlag.AlignCenter, str(i + 1))
        
        # Draw center indicator
        painter.setBrush(QBrush(QColor(43, 43, 43)))
        painter.setPen(QPen(QColor(224, 224, 224), 2))
        painter.drawEllipse(QPoint(center_x, center_y), 30, 30)
        painter.setPen(QPen(QColor(224, 224, 224)))
        painter.drawText(QRect(center_x - 30, center_y - 30, 60, 60), Qt.AlignmentFlag.AlignCenter, "N")
        
        # Draw legend
        legend_y = self.height() - 25
        legend_x = 10
        for state, color in self.STATE_COLORS.items():
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(Qt.GlobalColor.black, 1))
            painter.drawEllipse(QPoint(legend_x, legend_y), 8, 8)
            legend_x += 20
    
    def mousePressEvent(self, event):
        """Handle mouse clicks on compass"""
        center_x = self.width() // 2
        center_y = self.height() // 2
        radius = min(center_x, center_y) - 40
        
        # Get click position relative to center
        dx = event.pos().x() - center_x
        dy = event.pos().y() - center_y
        distance = np.sqrt(dx**2 + dy**2)
        
        # Check if click is within compass
        if distance < radius and distance > 30:  # Outside center circle
            # Calculate angle
            angle = np.arctan2(dy, dx) * 180 / np.pi
            angle = (angle + 90) % 360  # Rotate to start from top
            
            # Determine which camera was clicked
            angle_per_camera = 360 / self.camera_count
            camera_index = int(angle / angle_per_camera)
            
            # Check if clicked on camera icon (for state cycling)
            angle_mid = (camera_index * angle_per_camera + angle_per_camera / 2) * np.pi / 180
            icon_x = center_x + int((radius * 0.7) * np.cos(angle_mid - np.pi / 2))
            icon_y = center_y + int((radius * 0.7) * np.sin(angle_mid - np.pi / 2))
            
            icon_dx = event.pos().x() - icon_x
            icon_dy = event.pos().y() - icon_y
            icon_distance = np.sqrt(icon_dx**2 + icon_dy**2)
            
            if icon_distance < 15:  # Clicked on icon
                self.cycle_camera_state(camera_index)
            else:  # Clicked on slice
                self.selected_camera = camera_index
                self.camera_clicked.emit(camera_index)
                self.update()


class PerspectivePreviewPanel(QWidget):
    """
    Preview panel for perspective mode (E2P transform).
    Shows main camera view with controls and circular compass.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.current_equirect_image = None
        self.current_camera_index = 0
        self.transform = E2PTransform()
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI"""
        layout = QHBoxLayout(self)
        
        # Left: Look-down button (optional)
        left_panel = QVBoxLayout()
        self.look_down_btn = QPushButton("⬇ Look Down\n[+ Add Ring]")
        self.look_down_btn.setFixedSize(120, 80)
        self.look_down_btn.setEnabled(False)  # TODO: Implement multi-ring
        left_panel.addWidget(self.look_down_btn)
        left_panel.addStretch()
        layout.addLayout(left_panel)
        
        # Center: Main preview and controls
        center_panel = QVBoxLayout()
        
        # Main camera preview
        self.preview_label = QLabel("No image loaded")
        self.preview_label.setFixedSize(640, 360)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet("background: #2b2b2b; border: 2px solid #555;")
        center_panel.addWidget(self.preview_label)
        
        # View controls
        controls_group = QGroupBox("View Controls")
        controls_layout = QVBoxLayout()
        
        # Yaw/Pan
        yaw_layout = QHBoxLayout()
        yaw_layout.addWidget(QLabel("Yaw/Pan:"))
        self.yaw_slider = QSlider(Qt.Orientation.Horizontal)
        self.yaw_slider.setRange(-180, 180)
        self.yaw_slider.setValue(0)
        self.yaw_slider.valueChanged.connect(self.update_preview)
        yaw_layout.addWidget(self.yaw_slider)
        self.yaw_spin = QSpinBox()
        self.yaw_spin.setRange(-180, 180)
        self.yaw_spin.setValue(0)
        self.yaw_spin.valueChanged.connect(self.yaw_slider.setValue)
        self.yaw_slider.valueChanged.connect(self.yaw_spin.setValue)
        yaw_layout.addWidget(self.yaw_spin)
        controls_layout.addLayout(yaw_layout)
        
        # Pitch/Tilt
        pitch_layout = QHBoxLayout()
        pitch_layout.addWidget(QLabel("Pitch/Tilt:"))
        self.pitch_slider = QSlider(Qt.Orientation.Horizontal)
        self.pitch_slider.setRange(-90, 90)
        self.pitch_slider.setValue(0)
        self.pitch_slider.valueChanged.connect(self.update_preview)
        pitch_layout.addWidget(self.pitch_slider)
        self.pitch_spin = QSpinBox()
        self.pitch_spin.setRange(-90, 90)
        self.pitch_spin.setValue(0)
        self.pitch_spin.valueChanged.connect(self.pitch_slider.setValue)
        self.pitch_slider.valueChanged.connect(self.pitch_spin.setValue)
        pitch_layout.addWidget(self.pitch_spin)
        controls_layout.addLayout(pitch_layout)
        
        # FOV
        fov_layout = QHBoxLayout()
        fov_layout.addWidget(QLabel("FOV:"))
        self.fov_slider = QSlider(Qt.Orientation.Horizontal)
        self.fov_slider.setRange(30, 150)
        self.fov_slider.setValue(110)
        self.fov_slider.valueChanged.connect(self.update_preview)
        fov_layout.addWidget(self.fov_slider)
        self.fov_spin = QSpinBox()
        self.fov_spin.setRange(30, 150)
        self.fov_spin.setValue(110)
        self.fov_spin.valueChanged.connect(self.fov_slider.setValue)
        self.fov_slider.valueChanged.connect(self.fov_spin.setValue)
        fov_layout.addWidget(self.fov_spin)
        controls_layout.addLayout(fov_layout)
        
        controls_group.setLayout(controls_layout)
        center_panel.addWidget(controls_group)
        
        # Circular compass
        self.compass = CircularCompassWidget(8)
        self.compass.camera_clicked.connect(self.on_camera_selected)
        center_panel.addWidget(self.compass)
        
        layout.addLayout(center_panel)
        
        # Right: Look-up button (optional)
        right_panel = QVBoxLayout()
        self.look_up_btn = QPushButton("⬆ Look Up\n[+ Add Ring]")
        self.look_up_btn.setFixedSize(120, 80)
        self.look_up_btn.setEnabled(False)  # TODO: Implement multi-ring
        right_panel.addWidget(self.look_up_btn)
        right_panel.addStretch()
        layout.addLayout(right_panel)
    
    def load_equirect_image(self, image_path: str):
        """Load an equirectangular image for preview"""
        self.current_equirect_image = cv2.imread(image_path)
        if self.current_equirect_image is not None:
            self.update_preview()
    
    def on_camera_selected(self, camera_index: int):
        """Handle camera selection from compass"""
        self.current_camera_index = camera_index
        
        # Calculate yaw for this camera
        camera_count = self.compass.camera_count
        yaw = int((camera_index * 360 / camera_count)) - 180
        
        self.yaw_slider.setValue(yaw)
        self.update_preview()
    
    def update_preview(self):
        """Update the preview image based on current settings"""
        if self.current_equirect_image is None:
            return
        
        yaw = self.yaw_slider.value()
        pitch = self.pitch_slider.value()
        fov = self.fov_slider.value()
        
        # Generate perspective view
        try:
            perspective_img = self.transform.equirect_to_pinhole(
                self.current_equirect_image,
                yaw=yaw,
                pitch=pitch,
                roll=0,
                h_fov=fov,
                output_width=640,
                output_height=360
            )
            
            # Convert to QPixmap
            height, width, channel = perspective_img.shape
            bytes_per_line = 3 * width
            q_img = QImage(perspective_img.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_img)
            
            self.preview_label.setPixmap(pixmap)
            
        except Exception as e:
            self.preview_label.setText(f"Error: {e}")


class CubemapPreviewPanel(QWidget):
    """
    Preview panel for cubemap mode (E2C transform).
    Shows equirectangular image with grid overlay and tile visualization.
    """
    
    # Tile colors for visualization
    TILE_COLORS = {
        'left': QColor(255, 100, 100, 128),    # Red
        'front': QColor(100, 255, 100, 128),   # Green
        'right': QColor(100, 100, 255, 128),   # Blue
        'back': QColor(255, 255, 100, 128),    # Yellow
        'top': QColor(100, 255, 255, 128),     # Cyan
        'bottom': QColor(255, 100, 255, 128),  # Magenta
        # 8-tile diagonal corners
        'top_left': QColor(200, 150, 100, 128),
        'top_right': QColor(150, 200, 100, 128)
    }
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.current_equirect_image = None
        self.cubemap_mode = '6-face'  # or '8-tile'
        self.tile_fov = 90
        self.overlap_percent = 5
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout(self)
        
        # Equirectangular preview with grid overlay
        self.preview_label = QLabel("No image loaded")
        self.preview_label.setFixedSize(1200, 600)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet("background: #2b2b2b; border: 2px solid #555;")
        layout.addWidget(self.preview_label)
        
        # Cubemap settings
        settings_group = QGroupBox("Cubemap Settings")
        settings_layout = QVBoxLayout()
        
        # Mode selection
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("6-Face Cubemap", "6-face")
        self.mode_combo.addItem("8-Tile Cubemap", "8-tile")
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        mode_layout.addWidget(self.mode_combo)
        mode_layout.addStretch()
        settings_layout.addLayout(mode_layout)
        
        # Tile FOV
        fov_layout = QHBoxLayout()
        fov_layout.addWidget(QLabel("Tile FOV (°):"))
        self.tile_fov_slider = QSlider(Qt.Orientation.Horizontal)
        self.tile_fov_slider.setRange(60, 120)
        self.tile_fov_slider.setValue(90)
        self.tile_fov_slider.valueChanged.connect(self.update_preview)
        fov_layout.addWidget(self.tile_fov_slider)
        self.tile_fov_spin = QSpinBox()
        self.tile_fov_spin.setRange(60, 120)
        self.tile_fov_spin.setValue(90)
        self.tile_fov_spin.valueChanged.connect(self.tile_fov_slider.setValue)
        self.tile_fov_slider.valueChanged.connect(self.tile_fov_spin.setValue)
        fov_layout.addWidget(self.tile_fov_spin)
        settings_layout.addLayout(fov_layout)
        
        # Overlap percentage
        overlap_layout = QHBoxLayout()
        overlap_layout.addWidget(QLabel("Overlap (%):"))
        self.overlap_slider = QSlider(Qt.Orientation.Horizontal)
        self.overlap_slider.setRange(0, 50)
        self.overlap_slider.setValue(5)
        self.overlap_slider.valueChanged.connect(self.update_preview)
        overlap_layout.addWidget(self.overlap_slider)
        self.overlap_spin = QSpinBox()
        self.overlap_spin.setRange(0, 50)
        self.overlap_spin.setValue(5)
        self.overlap_spin.valueChanged.connect(self.overlap_slider.setValue)
        self.overlap_slider.valueChanged.connect(self.overlap_spin.setValue)
        overlap_layout.addWidget(self.overlap_spin)
        settings_layout.addLayout(overlap_layout)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # Legend
        legend_layout = QHBoxLayout()
        legend_layout.addWidget(QLabel("<b>Tile Colors:</b>"))
        legend_layout.addWidget(QLabel("🟥 Left"))
        legend_layout.addWidget(QLabel("🟩 Front"))
        legend_layout.addWidget(QLabel("🟦 Right"))
        legend_layout.addWidget(QLabel("🟨 Back"))
        legend_layout.addWidget(QLabel("🟦 Top"))
        legend_layout.addWidget(QLabel("🟪 Bottom"))
        legend_layout.addStretch()
        layout.addLayout(legend_layout)
    
    def load_equirect_image(self, image_path: str):
        """Load an equirectangular image for preview"""
        self.current_equirect_image = cv2.imread(image_path)
        if self.current_equirect_image is not None:
            self.update_preview()
    
    def on_mode_changed(self, index: int):
        """Handle mode change"""
        self.cubemap_mode = self.mode_combo.currentData()
        self.update_preview()
    
    def update_preview(self):
        """Update the preview with grid overlay"""
        if self.current_equirect_image is None:
            return
        
        # Get current settings
        self.tile_fov = self.tile_fov_slider.value()
        self.overlap_percent = self.overlap_slider.value()
        
        try:
            # Resize equirect for display
            display_img = cv2.resize(self.current_equirect_image, (1200, 600))
            
            # Draw grid overlay based on cubemap mode
            if self.cubemap_mode == '6-face':
                self._draw_6face_grid(display_img)
            else:  # 8-tile
                self._draw_8tile_grid(display_img)
            
            # Convert to QPixmap
            height, width, channel = display_img.shape
            bytes_per_line = 3 * width
            q_img = QImage(display_img.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_img)
            
            self.preview_label.setPixmap(pixmap)
            
        except Exception as e:
            self.preview_label.setText(f"Error: {e}")
    
    def _draw_6face_grid(self, img: np.ndarray):
        """Draw 6-face cubemap grid overlay"""
        height, width = img.shape[:2]
        
        # Standard cubemap layout (horizontal strip)
        # [Left] [Front] [Right] [Back] [Top] [Bottom]
        tile_width = width // 6
        
        tile_positions = [
            ('left', 0, 0, tile_width, height),
            ('front', tile_width, 0, tile_width * 2, height),
            ('right', tile_width * 2, 0, tile_width * 3, height),
            ('back', tile_width * 3, 0, tile_width * 4, height),
            ('top', tile_width * 4, 0, tile_width * 5, height),
            ('bottom', tile_width * 5, 0, width, height)
        ]
        
        for name, x1, y1, x2, y2 in tile_positions:
            # Draw semi-transparent colored overlay
            overlay = img.copy()
            color = self.TILE_COLORS[name]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), 
                         (color.blue(), color.green(), color.red()), -1)
            cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
            
            # Draw border
            border_width = 3 if self.overlap_percent > 0 else 2
            cv2.rectangle(img, (x1, y1), (x2, y2), 
                         (color.blue(), color.green(), color.red()), border_width)
            
            # Draw label
            cv2.putText(img, name.upper(), (x1 + 10, y1 + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def _draw_8tile_grid(self, img: np.ndarray):
        """Draw 8-tile cubemap grid overlay (with diagonals)"""
        height, width = img.shape[:2]
        
        # 8-tile layout includes diagonal corner views
        # More complex grid pattern
        tile_width = width // 8
        
        tile_positions = [
            ('left', 0, 0, tile_width, height),
            ('top_left', tile_width, 0, tile_width * 2, height),
            ('front', tile_width * 2, 0, tile_width * 3, height),
            ('top_right', tile_width * 3, 0, tile_width * 4, height),
            ('right', tile_width * 4, 0, tile_width * 5, height),
            ('back', tile_width * 5, 0, tile_width * 6, height),
            ('top', tile_width * 6, 0, tile_width * 7, height),
            ('bottom', tile_width * 7, 0, width, height)
        ]
        
        for name, x1, y1, x2, y2 in tile_positions:
            # Draw semi-transparent colored overlay
            overlay = img.copy()
            color = self.TILE_COLORS.get(name, QColor(128, 128, 128, 128))
            cv2.rectangle(overlay, (x1, y1), (x2, y2),
                         (color.blue(), color.green(), color.red()), -1)
            cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
            
            # Draw border
            cv2.rectangle(img, (x1, y1), (x2, y2),
                         (color.blue(), color.green(), color.red()), 3)
            
            # Draw label
            cv2.putText(img, name.replace('_', ' ').upper(), (x1 + 5, y1 + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


# ──────────────────────────────────────────────────────────────────────────────
# Equirectangular preview panel for Frame Extraction stage
# ──────────────────────────────────────────────────────────────────────────────

def _apply_color_corrections(img: np.ndarray, opts: dict) -> np.ndarray:
    """
    Apply Insta360-style color corrections to a BGR uint8 numpy image.
    All parameters match the SDK range (-100..100 / 0..100 for definition).
    Operations execute in float32 for accuracy then clip back to uint8.
    """
    if not opts:
        return img

    f = img.astype(np.float32) / 255.0

    # ── Exposure: photographic stops [-100..100] ─────────────────────────────
    exp = opts.get('exposure', 0)
    if exp:
        f *= 2.0 ** (exp / 100.0)

    # ── Brightness: linear lift/cut ───────────────────────────────────────────
    br = opts.get('brightness', 0)
    if br:
        f += br / 400.0

    # ── Contrast: S-curve around mid-grey ─────────────────────────────────────
    con = opts.get('contrast', 0)
    if con:
        f = (f - 0.5) * (1.0 + con / 100.0) + 0.5

    # ── Highlights: compress/expand bright pixels ─────────────────────────────
    hl = opts.get('highlights', 0)
    if hl:
        mask = np.clip((f - 0.5) * 2.0, 0.0, 1.0)
        f += mask * (hl / 100.0) * 0.25

    # ── Shadows: lift/cut dark pixels ────────────────────────────────────────
    sh = opts.get('shadows', 0)
    if sh:
        mask = np.clip((0.5 - f) * 2.0, 0.0, 1.0)
        f += mask * (sh / 100.0) * 0.25

    # ── Black point ───────────────────────────────────────────────────────────
    bp = opts.get('blackpoint', 0)
    if bp:
        f = np.where(f < 0.5, f + bp / 500.0, f)

    f = np.clip(f, 0.0, 1.0)

    # ── Saturation & Vibrance (HSV) ───────────────────────────────────────────
    sat = opts.get('saturation', 0)
    vib = opts.get('vibrance', 0)
    if sat or vib:
        u8 = (f * 255.0).astype(np.uint8)
        hsv = cv2.cvtColor(u8, cv2.COLOR_BGR2HSV).astype(np.float32)
        if sat:
            hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + sat / 100.0), 0.0, 255.0)
        if vib:
            sat_norm = hsv[..., 1] / 255.0
            factor = 1.0 + (vib / 100.0) * (1.0 - sat_norm) * 0.8
            hsv[..., 1] = np.clip(hsv[..., 1] * factor, 0.0, 255.0)
        f = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0

    # ── Warmth: colour temperature shift ─────────────────────────────────────
    # +100 = warm/orange, -100 = cool/blue
    wm = opts.get('warmth', 0)
    if wm:
        delta = wm / 500.0
        f[..., 0] = f[..., 0] - delta  # blue channel  (BGR index 0)
        f[..., 2] = f[..., 2] + delta  # red channel   (BGR index 2)

    # ── Tint: +100 = green, -100 = magenta ───────────────────────────────────
    ti = opts.get('tint', 0)
    if ti:
        delta = ti / 500.0
        f[..., 1] = f[..., 1] + delta          # green boost
        f[..., 0] = f[..., 0] - delta * 0.4    # slight blue suppress for magenta
        f[..., 2] = f[..., 2] - delta * 0.4    # slight red  suppress for magenta

    result = np.clip(f * 255.0, 0, 255).astype(np.uint8)

    # ── Definition (sharpness): 0..100 ────────────────────────────────────────
    defi = opts.get('definition', 0)
    if defi > 0:
        blurred = cv2.GaussianBlur(result, (0, 0), 1.5)
        amount  = defi / 100.0
        result  = cv2.addWeighted(result, 1.0 + amount, blurred, -amount, 0)
        result  = np.clip(result, 0, 255).astype(np.uint8)

    return result


class _SDKPreviewWorker(QThread):
    """
    Background thread: uses SDKExtractor to stitch 1 frame from an INSV file.
    Uses 'draft' quality for speed — this is preview only.
    """
    frame_ready = pyqtSignal(str)   # absolute path to the extracted JPEG
    failed      = pyqtSignal(str)   # human-readable error message

    def __init__(self, sdk_extractor, input_path: str, parent=None):
        super().__init__(parent)
        self._extractor  = sdk_extractor
        self._input_path = input_path

    def run(self):
        import tempfile
        tmp = tempfile.mkdtemp(prefix="360tk_prev_")
        try:
            frames = self._extractor.extract_frames(
                input_path   = self._input_path,
                output_dir   = tmp,
                fps          = 0.5,      # 0.5 fps → 1 frame in first 2 s
                quality      = 'good',   # dynamicstitch + flowstate → same orientation as pipeline
                output_format= 'jpg',
                start_time   = 0.0,
                end_time     = 2.0,
            )
            if frames and Path(frames[0]).exists():
                self.frame_ready.emit(str(frames[0]))
            else:
                self.failed.emit("SDK returned no frames (check SDK path in Settings)")
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class _FFmpegLensPreviewWorker(QThread):
    """
    Background thread: uses FFmpeg to grab 1 preview frame from fisheye stream(s).
    mode: 'both' | 'lens1' | 'lens2'
    """
    frames_ready = pyqtSignal(dict)   # {'lens_1': path, 'lens_2': path} (subset)
    failed       = pyqtSignal(str)

    def __init__(self, ffmpeg_path: str, input_path: str, mode: str = 'both', parent=None):
        super().__init__(parent)
        self._ffmpeg = ffmpeg_path
        self._path   = input_path
        self._mode   = mode

    def run(self):
        import tempfile
        tmp = tempfile.mkdtemp(prefix="360tk_lens_prev_")
        result = {}
        try:
            lenses = []
            if self._mode in ('both', 'lens1'):
                lenses.append(('lens_1', 0))
            if self._mode in ('both', 'lens2'):
                lenses.append(('lens_2', 1))

            for name, stream in lenses:
                out = str(Path(tmp) / f"{name}.jpg")
                cmd = [
                    self._ffmpeg,
                    '-y', '-ss', '0',
                    '-i', self._path,
                    '-map', f'0:v:{stream}',
                    '-frames:v', '1',
                    '-q:v', '3',
                    out,
                ]
                proc = subprocess.run(
                    cmd, capture_output=True, timeout=30
                )
                if proc.returncode == 0 and Path(out).exists():
                    result[name] = out

            if result:
                self.frames_ready.emit(result)
            else:
                self.failed.emit("FFmpeg could not extract lens frames (is the file an .insv with dual streams?)")
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class _PanoRenderWorker(QThread):
    """
    Background thread: applies colour corrections to the equirectangular image
    and horizontally shifts it by Yaw so the panorama view scrolls correctly.
    Emits a BGR numpy array ready to display.
    """
    result_ready = pyqtSignal(object)   # np.ndarray BGR uint8

    def __init__(self, orig_bgr: np.ndarray, yaw: float, color_opts: dict,
                 skip_roll: bool = False, parent=None):
        super().__init__(parent)
        self._orig       = orig_bgr
        self._yaw        = float(yaw)
        self._color_opts = dict(color_opts)
        self._skip_roll  = skip_roll

    def run(self):
        try:
            src = self._orig
            # Downscale for display speed (preview only)
            pano_w = 1280
            pano_h = pano_w // 2
            if src.shape[1] != pano_w or src.shape[0] != pano_h:
                src = cv2.resize(src, (pano_w, pano_h), interpolation=cv2.INTER_AREA)

            # Apply colour corrections first
            corrected = _apply_color_corrections(src, self._color_opts)

            # Horizontal shift by Yaw — disabled for fisheye (skip_roll) mode
            if self._skip_roll:
                self.result_ready.emit(corrected)
            else:
                # yaw range -180..180 maps to pixel shift 0..width
                shift = int((self._yaw / 360.0) * pano_w)
                self.result_ready.emit(np.roll(corrected, shift, axis=1))
        except Exception as exc:  # noqa: BLE001
            import logging
            logging.getLogger(__name__).warning(f"Pano render failed: {exc}")



class _EquirectCanvas(QWidget):
    """Internal canvas: renders the current processed preview frame."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap: Optional[QPixmap] = None
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(280, 160)

    def set_pixmap(self, pixmap: Optional[QPixmap]):
        self._pixmap = pixmap
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(27, 31, 36))

        if self._pixmap is None:
            p.setPen(QColor(107, 114, 128))
            f = p.font()
            f.setPointSize(11)
            p.setFont(f)
            p.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "360° PREVIEW\n\nLoad INSV file — stitched frame extracts automatically",
            )
            return

        pix = self._pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        p.drawPixmap(
            (self.width()  - pix.width())  // 2,
            (self.height() - pix.height()) // 2,
            pix,
        )


class EquirectPreviewWidget(QWidget):
    """
    Panoramic 360° preview for the Frame Extraction tab.

    Shows the equirectangular image with:
      - Colour corrections applied (matching the exported frames)
      - Horizontal scroll driven by the Yaw slider (np.roll)

    No perspective re-projection is performed here; the panorama is
    always shown as a flat equirectangular strip.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._video_path: Optional[str]        = None
        self._orig_bgr:   Optional[np.ndarray] = None   # raw stitched equirectangular
        self._yaw:        float                = -180.0  # default: left edge (same as slider default)
        self._color_opts: dict                 = {}
        self._extraction_method: str                     = 'sdk_stitching'
        self._fisheye_mode:      bool                     = False
        self._rotation:          int                      = 0   # 0 | 90 | 180 | 270
        self._fisheye_labels:    list                     = []  # [(text, x_fraction), ...]
        self._sdk         = None
        self._sdk_worker: Optional[_SDKPreviewWorker]     = None
        self._ffmpeg_extractor                            = None
        self._ffmpeg_lens_worker                          = None
        self._render_worker: Optional[_PanoRenderWorker]  = None

        # 120 ms debounce so slider drags don't flood the worker thread
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(120)
        self._render_timer.timeout.connect(self._start_render)

        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._canvas = _EquirectCanvas(self)
        root.addWidget(self._canvas, stretch=1)

        # Action bar
        bar_widget = QWidget()
        bar_widget.setObjectName("previewBar")
        bar = QHBoxLayout(bar_widget)
        bar.setContentsMargins(8, 4, 8, 4)
        bar.setSpacing(8)

        self._status_lbl = QLabel("Load input file to enable preview")
        self._status_lbl.setProperty("role", "muted")
        bar.addWidget(self._status_lbl, stretch=1)

        # Rotation button — cycles 0°→90°→180°→270°→0°
        self._rotate_btn = QToolButton()
        self._rotate_btn.setText("↻ 0°")
        self._rotate_btn.setToolTip("Rotate preview — click to rotate 90° clockwise")
        self._rotate_btn.setFixedWidth(58)
        self._rotate_btn.clicked.connect(self._cycle_rotation)
        bar.addWidget(self._rotate_btn)

        open_btn = QToolButton()
        open_btn.setText("Open Image…")
        open_btn.clicked.connect(self._open_image_dialog)
        bar.addWidget(open_btn)

        self._extract_btn = QPushButton("Extract Preview Frame")
        self._extract_btn.setEnabled(False)
        self._extract_btn.setFixedWidth(168)
        self._extract_btn.clicked.connect(self._extract_preview)
        bar.addWidget(self._extract_btn)

        root.addWidget(bar_widget)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_view(self, yaw: float, pitch: float, roll: float, fov: float = 110.0):
        """Called when Yaw/Pitch/Roll sliders change (MediaProcessingPanel.view_changed).
        Only Yaw is used for the panorama — it scrolls the image horizontally."""
        self._yaw = yaw
        self._trigger_render()

    def set_color_opts(self, opts: dict):
        """Called when any colour slider changes (MediaProcessingPanel.values_changed)."""
        self._color_opts = dict(opts) if opts else {}
        self._trigger_render()

    def set_video_path(self, path: str):
        """Called when the input file path changes."""
        self._video_path = path.strip() if path else ""
        has_path = bool(self._video_path)
        self._extract_btn.setEnabled(has_path)
        if has_path:
            self._status_lbl.setText("Ready — extracting preview frame…")
            if self._video_path.lower().endswith(('.insv', '.mp4')):
                self._extract_preview()
        else:
            self._status_lbl.setText("Load input file to enable preview")

    def set_extraction_method(self, method: str):
        """Called when the Extraction Method combo changes.
        Updates the preview to match the selected lens/stitch mode."""
        if self._extraction_method == method:
            return
        self._extraction_method = method
        self._fisheye_mode = False   # will be set True when lens frames arrive
        # Re-trigger preview if we already have a loaded video
        if self._video_path and Path(self._video_path).exists():
            self._extract_preview()

    def load_image_path(self, path: str):
        """Load an equirectangular image file directly from disk."""
        try:
            img = cv2.imread(str(path))
            if img is None:
                self._status_lbl.setText("Failed to read image file")
                return
            self._store_orig(img)
            self._status_lbl.setText(f"{Path(path).name}  ({img.shape[1]}×{img.shape[0]})")
        except Exception as exc:
            self._status_lbl.setText(f"Error loading image: {exc}")

    def load_image_array(self, img_bgr: np.ndarray):
        """Load directly from a numpy BGR array."""
        self._store_orig(img_bgr)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _store_orig(self, img_bgr: np.ndarray):
        self._orig_bgr = img_bgr.copy()
        self._trigger_render()

    def _trigger_render(self):
        if self._orig_bgr is not None:
            self._render_timer.start()

    def _start_render(self):
        if self._orig_bgr is None:
            return
        if self._render_worker and self._render_worker.isRunning():
            self._render_worker.terminate()
            self._render_worker.wait()
        self._render_worker = _PanoRenderWorker(
            self._orig_bgr, self._yaw, self._color_opts,
            skip_roll=self._fisheye_mode, parent=self
        )
        self._render_worker.result_ready.connect(self._on_render_done)
        self._render_worker.start()

    def _cycle_rotation(self):
        """Rotate preview by 90° clockwise with each click."""
        self._rotation = (self._rotation + 90) % 360
        self._rotate_btn.setText(f"↻ {self._rotation}°")
        if self._orig_bgr is not None:
            self._trigger_render()

    def _on_render_done(self, rendered_bgr: np.ndarray):
        # Apply rotation to the image
        if self._rotation == 90:
            rendered_bgr = cv2.rotate(rendered_bgr, cv2.ROTATE_90_CLOCKWISE)
        elif self._rotation == 180:
            rendered_bgr = cv2.rotate(rendered_bgr, cv2.ROTATE_180)
        elif self._rotation == 270:
            rendered_bgr = cv2.rotate(rendered_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

        h, w = rendered_bgr.shape[:2]
        rgb  = cv2.cvtColor(rendered_bgr, cv2.COLOR_BGR2RGB)
        qimg  = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimg)

        # Draw fisheye labels with QPainter AFTER rotation — always upright
        if self._fisheye_labels:
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            font = painter.font()
            font.setPointSize(10)
            font.setBold(True)
            painter.setFont(font)
            for text, x_frac in self._fisheye_labels:
                x = int(x_frac * pixmap.width()) + 10
                y = pixmap.height() - 10   # bottom of the canvas
                # Shadow
                painter.setPen(QColor(0, 0, 0, 200))
                painter.drawText(x + 1, y + 1, text)
                # Text
                painter.setPen(QColor(220, 220, 220))
                painter.drawText(x, y, text)
            painter.end()

        self._canvas.set_pixmap(pixmap)

    def _extract_preview(self):
        # Route to lens preview for FFmpeg fisheye methods
        if self._extraction_method in ('ffmpeg_dual_lens', 'ffmpeg_lens1', 'ffmpeg_lens2'):
            self._extract_lens_preview()
            return

        path = self._video_path
        if not path or not Path(path).exists():
            self._status_lbl.setText("Input file not found")
            return

        if self._sdk is None:
            try:
                from ..extraction.sdk_extractor import SDKExtractor
                self._sdk = SDKExtractor()
            except Exception as exc:
                self._status_lbl.setText(f"SDK init failed: {exc}")
                return

        if not self._sdk.is_available():
            self._status_lbl.setText("SDK not available — configure SDK path in Settings")
            return

        if self._sdk_worker and self._sdk_worker.isRunning():
            self._sdk_worker.terminate()
            self._sdk_worker.wait()

        self._status_lbl.setText("Stitching equirectangular preview…")
        self._extract_btn.setEnabled(False)

        self._sdk_worker = _SDKPreviewWorker(self._sdk, path, self)
        self._sdk_worker.frame_ready.connect(self._on_sdk_frame_ready)
        self._sdk_worker.failed.connect(self._on_sdk_failed)
        self._sdk_worker.finished.connect(lambda: self._extract_btn.setEnabled(True))
        self._sdk_worker.start()

    def _extract_lens_preview(self):
        """Extract and display fisheye lens preview(s) using FFmpeg."""
        path = self._video_path
        if not path or not Path(path).exists():
            self._status_lbl.setText("Input file not found")
            return

        # Lazy-init FrameExtractor just to reuse its FFmpeg path detection
        if self._ffmpeg_extractor is None:
            try:
                from ..extraction.frame_extractor import FrameExtractor
                self._ffmpeg_extractor = FrameExtractor()
            except Exception as exc:
                self._status_lbl.setText(f"FFmpeg init failed: {exc}")
                return

        ffmpeg_path = self._ffmpeg_extractor.ffmpeg_path
        if not ffmpeg_path:
            self._status_lbl.setText("FFmpeg not found — install FFmpeg for lens preview")
            return

        mode = {'ffmpeg_dual_lens': 'both',
                'ffmpeg_lens1':     'lens1',
                'ffmpeg_lens2':     'lens2'}.get(self._extraction_method, 'both')

        if self._ffmpeg_lens_worker and self._ffmpeg_lens_worker.isRunning():
            self._ffmpeg_lens_worker.terminate()
            self._ffmpeg_lens_worker.wait()

        self._status_lbl.setText("Extracting fisheye lens preview…")
        self._extract_btn.setEnabled(False)

        self._ffmpeg_lens_worker = _FFmpegLensPreviewWorker(ffmpeg_path, path, mode, self)
        self._ffmpeg_lens_worker.frames_ready.connect(self._on_lens_frames_ready)
        self._ffmpeg_lens_worker.failed.connect(self._on_lens_preview_failed)
        self._ffmpeg_lens_worker.finished.connect(lambda: self._extract_btn.setEnabled(True))
        self._ffmpeg_lens_worker.start()

    def _on_sdk_frame_ready(self, img_path: str):
        img = cv2.imread(img_path)
        if img is None:
            self._status_lbl.setText("Could not load stitched frame")
            return
        src = Path(self._video_path).name if self._video_path else ""
        self._status_lbl.setText(
            f"Panorama: {src}  ({img.shape[1]}×{img.shape[0]})  — drag Yaw slider to pan"
        )
        self._fisheye_mode = False   # equirectangular: yaw rolling enabled
        self._fisheye_labels = []     # no labels for equirectangular
        self._store_orig(img)

    def _on_lens_frames_ready(self, frames: dict):
        """Build composite fisheye display (side-by-side for both, single for one lens)."""
        imgs = {}
        for name, img_path in frames.items():
            img = cv2.imread(img_path)
            if img is not None:
                imgs[name] = img

        if not imgs:
            self._status_lbl.setText("Could not load fisheye lens frames")
            return

        TARGET_H = 400

        def _resize(img):
            h, w = img.shape[:2]
            return cv2.resize(img, (int(w * TARGET_H / h), TARGET_H),
                              interpolation=cv2.INTER_AREA)

        if 'lens_1' in imgs and 'lens_2' in imgs:
            l1 = _resize(imgs['lens_1'])
            l2 = _resize(imgs['lens_2'])
            composite = np.hstack([l1, l2])
            self._fisheye_labels = [("Lens 1 (Front)", 0.0), ("Lens 2 (Back)", 0.5)]
            status = "Dual fisheye — Lens 1 (front) | Lens 2 (back)"
        else:
            name, raw = next(iter(imgs.items()))
            r = _resize(raw)
            label = "Lens 1 (Front)" if name == 'lens_1' else "Lens 2 (Back)"
            # Pad to dual-lens width so single lens renders at the same apparent scale
            pad = np.zeros_like(r)
            composite = np.hstack([r, pad]) if name == 'lens_1' else np.hstack([pad, r])
            x_frac = 0.0 if name == 'lens_1' else 0.5
            self._fisheye_labels = [(label, x_frac)]
            status = f"Fisheye — {label}"

        fn = Path(self._video_path).name if self._video_path else ""
        self._status_lbl.setText(f"{fn} — {status}")
        self._fisheye_mode = True   # disable yaw roll for fisheye
        self._store_orig(composite)

    def _on_lens_preview_failed(self, err: str):
        self._status_lbl.setText(f"Lens preview failed: {err}")
        self._extract_btn.setEnabled(True)

    def _on_sdk_failed(self, err: str):
        self._status_lbl.setText(f"SDK preview failed: {err}")
        self._extract_btn.setEnabled(True)

    def _open_image_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Equirectangular Image", "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff)",
        )
        if path:
            self.load_image_path(path)
