"""
RIFE Player Settings Management
Configuration storage and settings dialog.
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, 
    QLabel, QLineEdit, QPushButton, QComboBox, QSpinBox,
    QGroupBox, QFileDialog, QMessageBox, QTabWidget, QWidget
)
from PyQt6.QtCore import Qt

from .utils import _find_tool, _find_mpv, get_available_gpus, get_gpu_info


@dataclass
class FlowForgeSettings:
    """RIFE Player configuration settings."""
    
    # Paths
    ffmpeg_path: str = ""
    ffprobe_path: str = ""
    mpv_path: str = ""
    rife_binary_path: str = ""
    rife_model_path: str = ""
    vs_plugin_path: str = ""
    
    # RIFE settings
    default_preset: str = "Film 60fps"
    gpu_id: int = 0
    scene_detection: bool = True
    
    # Player settings
    window_width: int = 900
    window_height: int = 650
    
    # Export settings
    export_threads: int = 4
    
    @classmethod
    def load(cls) -> "FlowForgeSettings":
        """Load settings from config file."""
        config_path = get_config_path()
        
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return cls(**data)
            except (json.JSONDecodeError, TypeError, FileNotFoundError):
                pass
        
        # Return default settings with auto-detected paths
        return cls._create_default()
    
    def save(self) -> None:
        """Save settings to config file."""
        config_path = get_config_path()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(self), f, indent=2)
        except (OSError, json.JSONEncodeError) as e:
            print(f"Error saving settings: {e}")
    
    @classmethod
    def _create_default(cls) -> "FlowForgeSettings":
        """Create default settings with auto-detected paths."""
        # Auto-detect tool paths
        ffmpeg_path = _find_tool("ffmpeg")
        ffprobe_path = _find_tool("ffprobe")
        mpv_path = _find_mpv()
        
        # Known deployment paths
        rife_binary = Path(r"C:\Users\Kad\Desktop\FlowForge\bin\rife-ncnn-vulkan.exe")
        rife_model = Path(r"C:\Users\Kad\Desktop\FlowForge\models\rife-v4.6")
        vs_plugin = Path(r"C:\Users\Kad\Desktop\FlowForge\vs-plugins\librife.dll")
        
        return cls(
            ffmpeg_path=ffmpeg_path,
            ffprobe_path=ffprobe_path,
            mpv_path=mpv_path,
            rife_binary_path=str(rife_binary) if rife_binary.exists() else "",
            rife_model_path=str(rife_model) if rife_model.exists() else "",
            vs_plugin_path=str(vs_plugin) if vs_plugin.exists() else "",
            gpu_id=0,
        )


def get_config_path() -> Path:
    """Get path to configuration file."""
    config_dir = Path.home() / ".rifeplayer"
    return config_dir / "config.json"


class SettingsDialog(QDialog):
    """Settings dialog for RIFE Player."""
    
    def __init__(self, settings: FlowForgeSettings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle("RIFE Player Settings")
        self.setModal(True)
        self.resize(500, 400)
        
        self.setup_ui()
        self.load_settings()
    
    def setup_ui(self) -> None:
        """Setup the settings dialog UI."""
        layout = QVBoxLayout(self)
        
        # Tab widget
        tab_widget = QTabWidget()
        layout.addWidget(tab_widget)
        
        # Paths tab
        paths_tab = QWidget()
        tab_widget.addTab(paths_tab, "Paths")
        self.setup_paths_tab(paths_tab)
        
        # RIFE tab
        rife_tab = QWidget()
        tab_widget.addTab(rife_tab, "RIFE")
        self.setup_rife_tab(rife_tab)
        
        # Player tab
        player_tab = QWidget()
        tab_widget.addTab(player_tab, "Player")
        self.setup_player_tab(player_tab)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        auto_detect_btn = QPushButton("Auto-Detect Paths")
        auto_detect_btn.clicked.connect(self.auto_detect_paths)
        button_layout.addWidget(auto_detect_btn)
        
        button_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        ok_btn = QPushButton("OK")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(ok_btn)
        
        layout.addLayout(button_layout)
    
    def setup_paths_tab(self, parent: QWidget) -> None:
        """Setup paths configuration tab."""
        layout = QVBoxLayout(parent)
        
        # Tools group
        tools_group = QGroupBox("Tool Paths")
        tools_layout = QFormLayout(tools_group)
        
        self.ffmpeg_edit = QLineEdit()
        self.browse_button(tools_layout, "FFmpeg:", self.ffmpeg_edit, "ffmpeg.exe")
        
        self.ffprobe_edit = QLineEdit()
        self.browse_button(tools_layout, "FFprobe:", self.ffprobe_edit, "ffprobe.exe")
        
        self.mpv_edit = QLineEdit()
        self.browse_button(tools_layout, "mpv:", self.mpv_edit, "mpv.exe")
        
        layout.addWidget(tools_group)
        
        # RIFE group
        rife_group = QGroupBox("RIFE Paths")
        rife_layout = QFormLayout(rife_group)
        
        self.rife_binary_edit = QLineEdit()
        self.browse_button(rife_layout, "RIFE Binary:", self.rife_binary_edit, "rife-ncnn-vulkan.exe")
        
        self.rife_model_edit = QLineEdit()
        self.browse_directory_button(rife_layout, "RIFE Model:", self.rife_model_edit)
        
        self.vs_plugin_edit = QLineEdit()
        self.browse_button(rife_layout, "VapourSynth Plugin:", self.vs_plugin_edit, "librife.dll")
        
        layout.addWidget(rife_group)
        
        layout.addStretch()
    
    def setup_rife_tab(self, parent: QWidget) -> None:
        """Setup RIFE configuration tab."""
        layout = QVBoxLayout(parent)
        
        # RIFE settings group
        rife_group = QGroupBox("RIFE Configuration")
        rife_layout = QFormLayout(rife_group)
        
        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "Film 60fps", "Anime 60fps", "Sports 60fps", 
            "Smooth 144fps", "Custom"
        ])
        rife_layout.addRow("Default Preset:", self.preset_combo)
        
        self.gpu_combo = QComboBox()
        gpu_ids = get_available_gpus()
        gpu_info = get_gpu_info()
        for gpu_id in gpu_ids:
            label = f"GPU {gpu_id}"
            if gpu_info and gpu_id == 0:
                label += f" ({gpu_info})"
            self.gpu_combo.addItem(label, gpu_id)
        rife_layout.addRow("GPU:", self.gpu_combo)
        
        layout.addWidget(rife_group)
        
        # Export settings group
        export_group = QGroupBox("Export Settings")
        export_layout = QFormLayout(export_group)
        
        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(1, 32)
        self.threads_spin.setValue(4)
        export_layout.addRow("Threads:", self.threads_spin)
        
        layout.addWidget(export_group)
        
        layout.addStretch()
    
    def setup_player_tab(self, parent: QWidget) -> None:
        """Setup player configuration tab."""
        layout = QVBoxLayout(parent)
        
        # Window settings group
        window_group = QGroupBox("Window Settings")
        window_layout = QFormLayout(window_group)
        
        self.width_spin = QSpinBox()
        self.width_spin.setRange(640, 3840)
        self.width_spin.setValue(900)
        window_layout.addRow("Default Width:", self.width_spin)
        
        self.height_spin = QSpinBox()
        self.height_spin.setRange(480, 2160)
        self.height_spin.setValue(650)
        window_layout.addRow("Default Height:", self.height_spin)
        
        layout.addWidget(window_group)
        
        layout.addStretch()
    
    def browse_button(self, layout: QFormLayout, label: str, 
                     line_edit: QLineEdit, filename: str) -> None:
        """Add a browse button for file selection."""
        container = QWidget()
        container_layout = QHBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        
        container_layout.addWidget(line_edit)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(
            lambda: self.browse_file(line_edit, filename)
        )
        container_layout.addWidget(browse_btn)
        
        layout.addRow(label, container)
    
    def browse_directory_button(self, layout: QFormLayout, label: str, 
                               line_edit: QLineEdit) -> None:
        """Add a browse button for directory selection."""
        container = QWidget()
        container_layout = QHBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        
        container_layout.addWidget(line_edit)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(
            lambda: self.browse_directory(line_edit)
        )
        container_layout.addWidget(browse_btn)
        
        layout.addRow(label, container)
    
    def browse_file(self, line_edit: QLineEdit, filename: str) -> None:
        """Open file dialog and update line edit."""
        current_path = line_edit.text() or str(Path.home())
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, f"Select {filename}", current_path,
            f"{filename} (*.exe);;All Files (*)"
        )
        
        if file_path:
            line_edit.setText(file_path)
    
    def browse_directory(self, line_edit: QLineEdit) -> None:
        """Open directory dialog and update line edit."""
        current_path = line_edit.text() or str(Path.home())
        
        directory = QFileDialog.getExistingDirectory(
            self, "Select Directory", current_path
        )
        
        if directory:
            line_edit.setText(directory)
    
    def auto_detect_paths(self) -> None:
        """Auto-detect tool paths."""
        # Auto-detect and update paths
        self.ffmpeg_edit.setText(_find_tool("ffmpeg"))
        self.ffprobe_edit.setText(_find_tool("ffprobe"))
        self.mpv_edit.setText(_find_mpv())
        
        # Known deployment paths
        rife_binary = Path(r"C:\Users\Kad\Desktop\FlowForge\bin\rife-ncnn-vulkan.exe")
        if rife_binary.exists():
            self.rife_binary_edit.setText(str(rife_binary))
        
        rife_model = Path(r"C:\Users\Kad\Desktop\FlowForge\models\rife-v4.6")
        if rife_model.exists():
            self.rife_model_edit.setText(str(rife_model))
        
        vs_plugin = Path(r"C:\Users\Kad\Desktop\FlowForge\vs-plugins\librife.dll")
        if vs_plugin.exists():
            self.vs_plugin_edit.setText(str(vs_plugin))
        
        QMessageBox.information(self, "Auto-Detection", 
                               "Paths have been auto-detected.")
    
    def load_settings(self) -> None:
        """Load current settings into the dialog."""
        self.ffmpeg_edit.setText(self.settings.ffmpeg_path)
        self.ffprobe_edit.setText(self.settings.ffprobe_path)
        self.mpv_edit.setText(self.settings.mpv_path)
        self.rife_binary_edit.setText(self.settings.rife_binary_path)
        self.rife_model_edit.setText(self.settings.rife_model_path)
        self.vs_plugin_edit.setText(self.settings.vs_plugin_path)
        
        # Set combo box selection
        index = self.preset_combo.findText(self.settings.default_preset)
        if index >= 0:
            self.preset_combo.setCurrentIndex(index)
        
        # Set GPU selection
        gpu_index = self.gpu_combo.findData(self.settings.gpu_id)
        if gpu_index >= 0:
            self.gpu_combo.setCurrentIndex(gpu_index)
        
        self.threads_spin.setValue(self.settings.export_threads)
        self.width_spin.setValue(self.settings.window_width)
        self.height_spin.setValue(self.settings.window_height)
    
    def accept(self) -> None:
        """Save settings and close dialog."""
        # Update settings from UI
        self.settings.ffmpeg_path = self.ffmpeg_edit.text()
        self.settings.ffprobe_path = self.ffprobe_edit.text()
        self.settings.mpv_path = self.mpv_edit.text()
        self.settings.rife_binary_path = self.rife_binary_edit.text()
        self.settings.rife_model_path = self.rife_model_edit.text()
        self.settings.vs_plugin_path = self.vs_plugin_edit.text()
        
        self.settings.default_preset = self.preset_combo.currentText()
        self.settings.gpu_id = self.gpu_combo.currentData()
        self.settings.export_threads = self.threads_spin.value()
        self.settings.window_width = self.width_spin.value()
        self.settings.window_height = self.height_spin.value()
        
        # Save to disk
        self.settings.save()
        
        super().accept()