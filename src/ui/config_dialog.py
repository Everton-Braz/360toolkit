"""
Configuration Management Dialog for 360FrameTools
Allows users to save/load/manage pipeline configurations
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QListWidget, QListWidgetItem, QMessageBox, QLineEdit,
    QTextEdit, QFileDialog, QGroupBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from pathlib import Path
import logging

from ..config.config_manager import get_config_manager

logger = logging.getLogger(__name__)


class ConfigManagementDialog(QDialog):
    """Dialog for managing saved configurations"""
    
    config_loaded = pyqtSignal(dict)  # Emitted when config is loaded
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.config_manager = get_config_manager()
        self.selected_config_path = None
        
        self.init_ui()
        self.refresh_config_list()
    
    def init_ui(self):
        """Initialize the UI"""
        self.setWindowTitle("Configuration Manager")
        self.setMinimumSize(800, 600)
        
        layout = QVBoxLayout(self)
        
        # === SAVED CONFIGURATIONS LIST ===
        configs_group = QGroupBox("Saved Configurations")
        configs_layout = QVBoxLayout()
        
        self.config_list = QListWidget()
        self.config_list.itemClicked.connect(self.on_config_selected)
        self.config_list.itemDoubleClicked.connect(self.load_selected_config)
        configs_layout.addWidget(self.config_list)
        
        # Config info display
        self.config_info = QTextEdit()
        self.config_info.setReadOnly(True)
        self.config_info.setMaximumHeight(100)
        self.config_info.setPlaceholderText("Select a configuration to view details...")
        configs_layout.addWidget(self.config_info)
        
        configs_group.setLayout(configs_layout)
        layout.addWidget(configs_group)
        
        # === ACTION BUTTONS ===
        button_layout = QHBoxLayout()
        
        self.load_btn = QPushButton("📂 Load Selected")
        self.load_btn.setEnabled(False)
        self.load_btn.clicked.connect(self.load_selected_config)
        button_layout.addWidget(self.load_btn)
        
        self.import_btn = QPushButton("📥 Import from File...")
        self.import_btn.clicked.connect(self.import_config)
        button_layout.addWidget(self.import_btn)
        
        self.export_btn = QPushButton("📤 Export Selected...")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export_selected_config)
        button_layout.addWidget(self.export_btn)
        
        self.delete_btn = QPushButton("🗑️ Delete Selected")
        self.delete_btn.setEnabled(False)
        self.delete_btn.clicked.connect(self.delete_selected_config)
        button_layout.addWidget(self.delete_btn)
        
        self.refresh_btn = QPushButton("🔄 Refresh")
        self.refresh_btn.clicked.connect(self.refresh_config_list)
        button_layout.addWidget(self.refresh_btn)
        
        layout.addLayout(button_layout)
        
        # === CLOSE BUTTON ===
        close_layout = QHBoxLayout()
        close_layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.setMinimumWidth(100)
        close_btn.clicked.connect(self.accept)
        close_layout.addWidget(close_btn)
        
        layout.addLayout(close_layout)
    
    def refresh_config_list(self):
        """Refresh the list of saved configurations"""
        self.config_list.clear()
        
        configs = self.config_manager.list_saved_configs()
        
        if not configs:
            item = QListWidgetItem("No saved configurations")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.config_list.addItem(item)
            return
        
        for filepath, config_name, saved_at in configs:
            display_text = f"{config_name} ({saved_at})"
            item = QListWidgetItem(display_text)
            item.setData(Qt.ItemDataRole.UserRole, filepath)
            self.config_list.addItem(item)
    
    def on_config_selected(self, item: QListWidgetItem):
        """Handle config selection"""
        filepath = item.data(Qt.ItemDataRole.UserRole)
        
        if filepath is None:
            self.load_btn.setEnabled(False)
            self.export_btn.setEnabled(False)
            self.delete_btn.setEnabled(False)
            return
        
        self.selected_config_path = filepath
        self.load_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        self.delete_btn.setEnabled(True)
        
        # Load and display config info
        config = self.config_manager.load_config(filepath)
        if config:
            info_lines = [
                f"Configuration: {filepath.stem}",
                f"Path: {filepath}",
                "",
                "Settings:",
                f"  • Stage 1 Enabled: {config.get('stage1_enabled', False)}",
                f"  • Stage 2 Enabled: {config.get('stage2_enabled', False)}",
                f"  • Stage 3 Enabled: {config.get('stage3_enabled', False)}",
                f"  • Extraction Method: {config.get('extraction_method', 'N/A')}",
                f"  • FPS: {config.get('fps_interval', 'N/A')}",
                f"  • Split Count: {config.get('split_count', 'N/A')}",
                f"  • FOV: {config.get('h_fov', 'N/A')}°",
                f"  • Model Size: {config.get('model_size', 'N/A')}",
            ]
            self.config_info.setText("\n".join(info_lines))
    
    def load_selected_config(self):
        """Load the selected configuration"""
        if not self.selected_config_path:
            return
        
        config = self.config_manager.load_config(self.selected_config_path)
        if config:
            QMessageBox.information(
                self,
                "Configuration Loaded",
                f"Configuration '{self.selected_config_path.stem}' loaded successfully!"
            )
            self.config_loaded.emit(config)
            self.accept()
        else:
            QMessageBox.critical(
                self,
                "Load Failed",
                "Failed to load configuration. Check the log for details."
            )
    
    def import_config(self):
        """Import configuration from external file"""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Import Configuration",
            str(Path.home()),
            "JSON Files (*.json)"
        )
        
        if not filepath:
            return
        
        config = self.config_manager.load_config(Path(filepath))
        if config:
            QMessageBox.information(
                self,
                "Configuration Imported",
                f"Configuration imported successfully!"
            )
            self.config_loaded.emit(config)
            self.accept()
        else:
            QMessageBox.critical(
                self,
                "Import Failed",
                "Failed to import configuration. Check the log for details."
            )
    
    def export_selected_config(self):
        """Export selected configuration to file"""
        if not self.selected_config_path:
            return
        
        config = self.config_manager.load_config(self.selected_config_path)
        if not config:
            QMessageBox.critical(self, "Export Failed", "Could not load configuration.")
            return
        
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export Configuration",
            str(Path.home() / f"{self.selected_config_path.stem}.json"),
            "JSON Files (*.json)"
        )
        
        if not filepath:
            return
        
        success = self.config_manager.save_config(config, Path(filepath))
        if success:
            QMessageBox.information(
                self,
                "Export Successful",
                f"Configuration exported to:\n{filepath}"
            )
        else:
            QMessageBox.critical(
                self,
                "Export Failed",
                "Failed to export configuration. Check the log for details."
            )
    
    def delete_selected_config(self):
        """Delete the selected configuration"""
        if not self.selected_config_path:
            return
        
        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to delete configuration '{self.selected_config_path.stem}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            success = self.config_manager.delete_config(self.selected_config_path)
            if success:
                QMessageBox.information(
                    self,
                    "Delete Successful",
                    "Configuration deleted successfully."
                )
                self.refresh_config_list()
                self.config_info.clear()
                self.selected_config_path = None
            else:
                QMessageBox.critical(
                    self,
                    "Delete Failed",
                    "Failed to delete configuration. Check the log for details."
                )


class SaveConfigDialog(QDialog):
    """Dialog for saving current configuration"""
    
    def __init__(self, current_config: dict, parent=None):
        super().__init__(parent)
        
        self.current_config = current_config
        self.config_manager = get_config_manager()
        self.saved_path = None
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI"""
        self.setWindowTitle("Save Configuration")
        self.setMinimumSize(500, 250)
        
        layout = QVBoxLayout(self)
        
        # === CONFIG NAME ===
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Configuration Name:"))
        
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g., MyPreset_8Cameras_110FOV")
        name_layout.addWidget(self.name_edit, stretch=1)
        
        layout.addLayout(name_layout)
        
        # === DESCRIPTION ===
        layout.addWidget(QLabel("Description (optional):"))
        self.description_edit = QTextEdit()
        self.description_edit.setMaximumHeight(100)
        self.description_edit.setPlaceholderText("Add notes about this configuration...")
        layout.addWidget(self.description_edit)
        
        # === BUTTONS ===
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        save_btn = QPushButton("💾 Save")
        save_btn.setMinimumWidth(100)
        save_btn.clicked.connect(self.save_config)
        button_layout.addWidget(save_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setMinimumWidth(100)
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
    
    def save_config(self):
        """Save the configuration"""
        config_name = self.name_edit.text().strip()
        
        if not config_name:
            QMessageBox.warning(
                self,
                "Name Required",
                "Please enter a configuration name."
            )
            return
        
        description = self.description_edit.toPlainText().strip()
        
        # Check if file already exists
        filepath = self.config_manager.DEFAULT_CONFIG_DIR / f"{config_name}.json"
        if filepath.exists():
            reply = QMessageBox.question(
                self,
                "Overwrite?",
                f"Configuration '{config_name}' already exists. Overwrite?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        
        # Save config
        success = self.config_manager.export_config_with_description(
            self.current_config,
            filepath,
            description
        )
        
        if success:
            self.saved_path = filepath
            QMessageBox.information(
                self,
                "Save Successful",
                f"Configuration '{config_name}' saved successfully!"
            )
            self.accept()
        else:
            QMessageBox.critical(
                self,
                "Save Failed",
                "Failed to save configuration. Check the log for details."
            )
