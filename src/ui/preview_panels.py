"""
Perspective Preview Panel Widget
Real-time preview for Stage 2 perspective mode with circular compass.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QSlider, QSpinBox, QPushButton, QGroupBox, QComboBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize, QPoint, QRect
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QBrush

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

