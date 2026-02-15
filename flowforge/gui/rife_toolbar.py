"""
RIFE Player RIFE Toolbar Widget
Controls for RIFE interpolation settings.
"""

from typing import Dict, Any

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QToolButton, QComboBox, 
    QSpinBox, QCheckBox, QLabel, QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from .settings import FlowForgeSettings
from .utils import get_gpu_info


class RifeToolbar(QWidget):
    """Toolbar widget for RIFE interpolation controls."""
    
    # Signals
    rife_toggled = pyqtSignal(bool)  # RIFE enabled/disabled
    preset_changed = pyqtSignal(str)  # Preset selection changed
    fps_changed = pyqtSignal(int)  # Custom FPS changed
    scene_detection_toggled = pyqtSignal(bool)  # Scene detection toggled
    
    def __init__(self, settings: FlowForgeSettings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.rife_enabled = False
        
        self.setup_ui()
        self.update_status()
    
    def setup_ui(self) -> None:
        """Setup the RIFE toolbar UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(12)
        
        # RIFE toggle button
        self.rife_button = QToolButton()
        self.rife_button.setText("RIFE OFF")
        self.rife_button.setCheckable(True)
        self.rife_button.setStyleSheet("""
            QToolButton {
                font-weight: bold;
                min-width: 80px;
                padding: 6px 12px;
            }
            QToolButton:checked {
                background-color: #0078d4;
                border-color: #106ebe;
            }
        """)
        self.rife_button.toggled.connect(self._on_rife_toggled)
        layout.addWidget(self.rife_button)
        
        # Separator
        separator1 = QFrame(); separator1.setFrameShape(QFrame.Shape.VLine); separator1.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(separator1)
        
        # Preset label and combo
        preset_label = QLabel("Preset:")
        layout.addWidget(preset_label)
        
        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "Film 60fps",
            "Anime 60fps", 
            "Sports 60fps",
            "Smooth 144fps",
            "Custom"
        ])
        self.preset_combo.setMinimumWidth(120)
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        layout.addWidget(self.preset_combo)
        
        # Custom FPS spinner (only visible for Custom preset)
        self.fps_label = QLabel("FPS:")
        self.fps_label.setVisible(False)
        layout.addWidget(self.fps_label)
        
        self.fps_spinner = QSpinBox()
        self.fps_spinner.setRange(30, 240)
        self.fps_spinner.setValue(60)
        self.fps_spinner.setSuffix(" fps")
        self.fps_spinner.setVisible(False)
        self.fps_spinner.valueChanged.connect(self._on_fps_changed)
        layout.addWidget(self.fps_spinner)
        
        # Separator
        separator2 = QFrame(); separator2.setFrameShape(QFrame.Shape.VLine); separator2.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(separator2)
        
        # Scene detection checkbox
        self.scene_detection = QCheckBox("Scene Detection")
        self.scene_detection.setChecked(self.settings.scene_detection)
        self.scene_detection.toggled.connect(self._on_scene_detection_toggled)
        layout.addWidget(self.scene_detection)
        
        # Flexible spacer
        layout.addStretch()
        
        # Status indicator
        self.status_label = QLabel("Ready")
        status_font = QFont()
        status_font.setPointSize(9)
        self.status_label.setFont(status_font)
        self.status_label.setStyleSheet("color: #aaaaaa;")
        layout.addWidget(self.status_label)
        
        # Initialize preset selection
        preset_index = self.preset_combo.findText(self.settings.default_preset)
        if preset_index >= 0:
            self.preset_combo.setCurrentIndex(preset_index)
        
        self._update_custom_fps_visibility()
    
    def _on_rife_toggled(self, checked: bool) -> None:
        """Handle RIFE toggle button."""
        self.rife_enabled = checked
        self.rife_button.setText("RIFE ON" if checked else "RIFE OFF")
        self.update_status()
        self.rife_toggled.emit(checked)
    
    def _on_preset_changed(self, preset: str) -> None:
        """Handle preset selection change."""
        self._update_custom_fps_visibility()
        self.update_status()
        self.preset_changed.emit(preset)
    
    def _on_fps_changed(self, fps: int) -> None:
        """Handle custom FPS change."""
        self.update_status()
        self.fps_changed.emit(fps)
    
    def _on_scene_detection_toggled(self, checked: bool) -> None:
        """Handle scene detection toggle."""
        self.scene_detection_toggled.emit(checked)
    
    def _update_custom_fps_visibility(self) -> None:
        """Show/hide custom FPS controls based on preset."""
        is_custom = self.preset_combo.currentText() == "Custom"
        self.fps_label.setVisible(is_custom)
        self.fps_spinner.setVisible(is_custom)
    
    def update_status(self) -> None:
        """Update the status indicator text."""
        if not self.rife_enabled:
            self.status_label.setText("Ready")
            return
        
        preset = self.preset_combo.currentText()
        
        if preset == "Custom":
            fps = self.fps_spinner.value()
            status_text = f"RIFE: {fps}fps Custom"
        else:
            status_text = f"RIFE: {preset}"
        
        # Add GPU info if available
        gpu_info = get_gpu_info()
        if gpu_info:
            # Shorten GPU name for status
            gpu_name = gpu_info.replace("NVIDIA GeForce ", "").replace("RTX ", "RTX")
            if len(gpu_name) > 20:
                gpu_name = gpu_name[:17] + "..."
            status_text += f" | GPU: {gpu_name}"
        
        self.status_label.setText(status_text)
    
    def get_rife_config(self) -> Dict[str, Any]:
        """Get current RIFE configuration."""
        preset = self.preset_combo.currentText()
        
        # Determine target FPS based on preset
        preset_fps = {
            "Film 60fps": 60,
            "Anime 60fps": 60,
            "Sports 60fps": 60,
            "Smooth 144fps": 144,
            "Custom": self.fps_spinner.value()
        }
        
        target_fps = preset_fps.get(preset, 60)
        
        # Determine RIFE model settings based on preset
        if preset == "Anime 60fps":
            # Anime-optimized settings
            model_scale = 1.0
            scene_threshold = 0.15  # More sensitive for anime
        elif preset == "Sports 60fps":
            # Sports-optimized settings  
            model_scale = 1.0
            scene_threshold = 0.25  # Less sensitive for fast motion
        elif preset == "Smooth 144fps":
            # High framerate settings
            model_scale = 1.0
            scene_threshold = 0.20
        else:
            # Film/Custom default settings
            model_scale = 1.0
            scene_threshold = 0.20
        
        return {
            "enabled": self.rife_enabled,
            "preset": preset,
            "target_fps": target_fps,
            "model_scale": model_scale,
            "scene_threshold": scene_threshold,
            "scene_detection": self.scene_detection.isChecked(),
            "gpu_id": self.settings.gpu_id,
            "rife_binary_path": self.settings.rife_binary_path,
            "rife_model_path": self.settings.rife_model_path,
            "vs_plugin_path": self.settings.vs_plugin_path,
        }
    
    def set_rife_enabled(self, enabled: bool) -> None:
        """Set RIFE enabled state."""
        self.rife_button.setChecked(enabled)
        # The signal will be emitted automatically
    
    def get_preset(self) -> str:
        """Get current preset."""
        return self.preset_combo.currentText()
    
    def set_preset(self, preset: str) -> None:
        """Set current preset."""
        index = self.preset_combo.findText(preset)
        if index >= 0:
            self.preset_combo.setCurrentIndex(index)
    
    def get_target_fps(self) -> int:
        """Get target FPS for current configuration."""
        config = self.get_rife_config()
        return config["target_fps"]
    
    def is_scene_detection_enabled(self) -> bool:
        """Check if scene detection is enabled."""
        return self.scene_detection.isChecked()
    
    def set_scene_detection_enabled(self, enabled: bool) -> None:
        """Set scene detection enabled state."""
        self.scene_detection.setChecked(enabled)