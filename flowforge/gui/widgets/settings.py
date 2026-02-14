"""
Settings Dialog Widget
Configuration dialog for FlowForge settings.
"""

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, 
                            QWidget, QLabel, QLineEdit, QPushButton, QSpinBox,
                            QComboBox, QCheckBox, QSlider, QGroupBox, QFileDialog,
                            QMessageBox, QDoubleSpinBox, QFormLayout)
from PyQt6.QtGui import QFont

from ..settings import settings


class SettingsDialog(QDialog):
    """Main settings configuration dialog."""
    
    settings_changed = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("FlowForge Settings")
        self.setModal(True)
        self.setMinimumSize(600, 500)
        
        self._setup_ui()
        self._load_current_settings()
    
    def _setup_ui(self) -> None:
        """Setup the settings dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(16, 16, 16, 16)
        
        # Tab widget
        self.tab_widget = QTabWidget()
        
        # Paths tab
        self.paths_tab = PathsTab()
        self.tab_widget.addTab(self.paths_tab, "Paths")
        
        # Processing tab  
        self.processing_tab = ProcessingTab()
        self.tab_widget.addTab(self.processing_tab, "Processing")
        
        # Export tab
        self.export_tab = ExportTab()
        self.tab_widget.addTab(self.export_tab, "Export")
        
        # Advanced tab
        self.advanced_tab = AdvancedTab()
        self.tab_widget.addTab(self.advanced_tab, "Advanced")
        
        layout.addWidget(self.tab_widget)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.restore_button = QPushButton("Restore Defaults")
        self.restore_button.clicked.connect(self._restore_defaults)
        
        button_layout.addWidget(self.restore_button)
        button_layout.addStretch()
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self._save_and_close)
        self.ok_button.setDefault(True)
        
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.ok_button)
        
        layout.addLayout(button_layout)
        
        # Style buttons
        self._style_buttons()
    
    def _style_buttons(self) -> None:
        """Apply button styles."""
        button_style = """
            QPushButton {
                background-color: #353535;
                border: 1px solid #3c3c3c;
                border-radius: 6px;
                padding: 8px 16px;
                color: #dcdcdc;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #404040;
            }
            QPushButton:pressed {
                background-color: #2a2a2a;
            }
        """
        
        ok_style = """
            QPushButton {
                background-color: #2a82da;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3a92da;
            }
            QPushButton:pressed {
                background-color: #1a72ca;
            }
        """
        
        self.restore_button.setStyleSheet(button_style)
        self.cancel_button.setStyleSheet(button_style)
        self.ok_button.setStyleSheet(ok_style)
    
    def _load_current_settings(self) -> None:
        """Load current settings into the dialog."""
        self.paths_tab.load_settings()
        self.processing_tab.load_settings()
        self.export_tab.load_settings()
        self.advanced_tab.load_settings()
    
    def _save_and_close(self) -> None:
        """Save all settings and close dialog."""
        # Validate settings first
        if not self._validate_settings():
            return
        
        # Save from all tabs
        self.paths_tab.save_settings()
        self.processing_tab.save_settings()
        self.export_tab.save_settings()
        self.advanced_tab.save_settings()
        
        # Persist to file
        settings.save_settings()
        
        # Emit signal and close
        self.settings_changed.emit()
        self.accept()
    
    def _validate_settings(self) -> bool:
        """Validate all settings before saving."""
        # Check required paths
        rife_path = self.paths_tab.rife_binary_edit.text().strip()
        if not rife_path or not Path(rife_path).exists():
            QMessageBox.warning(
                self, 
                "Invalid Path", 
                "RIFE binary path is required and must exist."
            )
            self.tab_widget.setCurrentWidget(self.paths_tab)
            return False
        
        model_path = self.paths_tab.model_dir_edit.text().strip()
        if not model_path or not Path(model_path).exists():
            QMessageBox.warning(
                self, 
                "Invalid Path", 
                "RIFE model directory path is required and must exist."
            )
            self.tab_widget.setCurrentWidget(self.paths_tab)
            return False
        
        return True
    
    def _restore_defaults(self) -> None:
        """Restore all settings to defaults."""
        result = QMessageBox.question(
            self,
            "Restore Defaults",
            "Are you sure you want to restore all settings to their default values?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if result == QMessageBox.StandardButton.Yes:
            # Reset settings to defaults
            settings._settings = settings._get_default_settings()
            settings._auto_detect_paths()
            
            # Reload UI
            self._load_current_settings()


class PathsTab(QWidget):
    """Paths configuration tab."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Setup paths tab UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        
        # RIFE paths group
        rife_group = QGroupBox("RIFE Configuration")
        rife_layout = QFormLayout(rife_group)
        
        # RIFE binary
        rife_binary_layout = QHBoxLayout()
        self.rife_binary_edit = QLineEdit()
        rife_binary_browse = QPushButton("Browse...")
        rife_binary_browse.clicked.connect(lambda: self._browse_file(
            self.rife_binary_edit, "RIFE Binary", "Executable Files (*.exe);;All Files (*)"
        ))
        
        rife_binary_layout.addWidget(self.rife_binary_edit)
        rife_binary_layout.addWidget(rife_binary_browse)
        rife_layout.addRow("RIFE Binary:", rife_binary_layout)
        
        # Model directory
        model_dir_layout = QHBoxLayout()
        self.model_dir_edit = QLineEdit()
        model_dir_browse = QPushButton("Browse...")
        model_dir_browse.clicked.connect(lambda: self._browse_folder(
            self.model_dir_edit, "RIFE Model Directory"
        ))
        
        model_dir_layout.addWidget(self.model_dir_edit)
        model_dir_layout.addWidget(model_dir_browse)
        rife_layout.addRow("Model Directory:", model_dir_layout)
        
        layout.addWidget(rife_group)
        
        # Playback paths group
        playback_group = QGroupBox("Playback Configuration")
        playback_layout = QFormLayout(playback_group)
        
        # mpv path
        mpv_layout = QHBoxLayout()
        self.mpv_edit = QLineEdit()
        mpv_browse = QPushButton("Browse...")
        mpv_browse.clicked.connect(lambda: self._browse_file(
            self.mpv_edit, "mpv Executable", "Executable Files (*.exe);;All Files (*)"
        ))
        
        mpv_layout.addWidget(self.mpv_edit)
        mpv_layout.addWidget(mpv_browse)
        playback_layout.addRow("mpv Path:", mpv_layout)
        
        # VapourSynth plugin
        vs_plugin_layout = QHBoxLayout()
        self.vs_plugin_edit = QLineEdit()
        vs_plugin_browse = QPushButton("Browse...")
        vs_plugin_browse.clicked.connect(lambda: self._browse_file(
            self.vs_plugin_edit, "VapourSynth Plugin", "Library Files (*.dll *.so);;All Files (*)"
        ))
        
        vs_plugin_layout.addWidget(self.vs_plugin_edit)
        vs_plugin_layout.addWidget(vs_plugin_browse)
        playback_layout.addRow("VS Plugin:", vs_plugin_layout)
        
        layout.addWidget(playback_group)
        
        layout.addStretch()
    
    def _browse_file(self, line_edit: QLineEdit, title: str, filter: str) -> None:
        """Browse for a file."""
        current_path = line_edit.text().strip()
        start_dir = str(Path(current_path).parent) if current_path and Path(current_path).parent.exists() else ""
        
        file_path, _ = QFileDialog.getOpenFileName(self, f"Select {title}", start_dir, filter)
        if file_path:
            line_edit.setText(file_path)
    
    def _browse_folder(self, line_edit: QLineEdit, title: str) -> None:
        """Browse for a folder."""
        current_path = line_edit.text().strip()
        start_dir = current_path if current_path and Path(current_path).exists() else ""
        
        folder_path = QFileDialog.getExistingDirectory(self, f"Select {title}", start_dir)
        if folder_path:
            line_edit.setText(folder_path)
    
    def load_settings(self) -> None:
        """Load settings into UI."""
        self.rife_binary_edit.setText(settings.get("rife_binary", ""))
        self.model_dir_edit.setText(settings.get("rife_model_dir", ""))
        self.mpv_edit.setText(settings.get("mpv_path", ""))
        self.vs_plugin_edit.setText(settings.get("vs_plugin_path", ""))
    
    def save_settings(self) -> None:
        """Save settings from UI."""
        settings.set("rife_binary", self.rife_binary_edit.text().strip())
        settings.set("rife_model_dir", self.model_dir_edit.text().strip())
        settings.set("mpv_path", self.mpv_edit.text().strip())
        settings.set("vs_plugin_path", self.vs_plugin_edit.text().strip())


class ProcessingTab(QWidget):
    """Processing configuration tab."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Setup processing tab UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        
        # GPU settings
        gpu_group = QGroupBox("GPU Configuration")
        gpu_layout = QFormLayout(gpu_group)
        
        self.gpu_spin = QSpinBox()
        self.gpu_spin.setMinimum(-1)
        self.gpu_spin.setMaximum(8)
        self.gpu_spin.setSpecialValueText("CPU")
        gpu_layout.addRow("Default GPU ID:", self.gpu_spin)
        
        layout.addWidget(gpu_group)
        
        # Threading settings
        thread_group = QGroupBox("Threading")
        thread_layout = QFormLayout(thread_group)
        
        self.threads_edit = QLineEdit()
        self.threads_edit.setPlaceholderText("load:process:save (e.g., 1:2:2)")
        thread_layout.addRow("Thread Pattern:", self.threads_edit)
        
        layout.addWidget(thread_group)
        
        # Scene detection
        scene_group = QGroupBox("Scene Detection")
        scene_layout = QVBoxLayout(scene_group)
        
        self.scene_detection_check = QCheckBox("Enable scene detection")
        scene_layout.addWidget(self.scene_detection_check)
        
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Threshold:"))
        
        self.scene_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.scene_threshold_slider.setMinimum(10)
        self.scene_threshold_slider.setMaximum(100)
        self.scene_threshold_slider.setValue(30)
        self.scene_threshold_slider.valueChanged.connect(self._update_threshold_label)
        
        self.threshold_label = QLabel("0.30")
        self.threshold_label.setMinimumWidth(40)
        
        threshold_layout.addWidget(self.scene_threshold_slider)
        threshold_layout.addWidget(self.threshold_label)
        
        scene_layout.addLayout(threshold_layout)
        layout.addWidget(scene_group)
        
        # Default preset
        preset_group = QGroupBox("Default Settings")
        preset_layout = QFormLayout(preset_group)
        
        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "Film (24→60fps)",
            "Anime (24→60fps)", 
            "Sports (30→60fps)",
            "Smooth (→144fps)",
            "Custom"
        ])
        preset_layout.addRow("Default Preset:", self.preset_combo)
        
        layout.addWidget(preset_group)
        
        layout.addStretch()
    
    def _update_threshold_label(self) -> None:
        """Update threshold label when slider changes."""
        value = self.scene_threshold_slider.value() / 100.0
        self.threshold_label.setText(f"{value:.2f}")
    
    def load_settings(self) -> None:
        """Load settings into UI."""
        self.gpu_spin.setValue(settings.get("default_gpu", 0))
        self.threads_edit.setText(settings.get("default_threads", "1:2:2"))
        self.scene_detection_check.setChecked(settings.get("scene_detection", True))
        
        threshold = settings.get("scene_threshold", 0.3)
        self.scene_threshold_slider.setValue(int(threshold * 100))
        self._update_threshold_label()
        
        preset = settings.get("default_preset", "Film (24→60fps)")
        index = self.preset_combo.findText(preset)
        if index >= 0:
            self.preset_combo.setCurrentIndex(index)
    
    def save_settings(self) -> None:
        """Save settings from UI."""
        settings.set("default_gpu", self.gpu_spin.value())
        settings.set("default_threads", self.threads_edit.text().strip())
        settings.set("scene_detection", self.scene_detection_check.isChecked())
        settings.set("scene_threshold", self.scene_threshold_slider.value() / 100.0)
        settings.set("default_preset", self.preset_combo.currentText())


class ExportTab(QWidget):
    """Export configuration tab."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Setup export tab UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        
        # Encoding presets
        preset_group = QGroupBox("Encoding Presets")
        preset_layout = QFormLayout(preset_group)
        
        self.encoding_preset_combo = QComboBox()
        self.encoding_preset_combo.addItems(["Quality", "Balanced", "Fast"])
        preset_layout.addRow("Default Preset:", self.encoding_preset_combo)
        
        layout.addWidget(preset_group)
        
        # Quality settings
        quality_group = QGroupBox("Quality Settings")
        quality_layout = QFormLayout(quality_group)
        
        self.crf_spin = QSpinBox()
        self.crf_spin.setMinimum(0)
        self.crf_spin.setMaximum(51)
        self.crf_spin.setValue(18)
        quality_layout.addRow("Default CRF:", self.crf_spin)
        
        layout.addWidget(quality_group)
        
        # Hardware acceleration
        hw_group = QGroupBox("Hardware Acceleration")
        hw_layout = QVBoxLayout(hw_group)
        
        self.nvenc_check = QCheckBox("Use NVENC hardware encoding by default")
        hw_layout.addWidget(self.nvenc_check)
        
        layout.addWidget(hw_group)
        
        layout.addStretch()
    
    def load_settings(self) -> None:
        """Load settings into UI."""
        preset = settings.get("default_encoding_preset", "Balanced")
        index = self.encoding_preset_combo.findText(preset)
        if index >= 0:
            self.encoding_preset_combo.setCurrentIndex(index)
        
        self.crf_spin.setValue(settings.get("default_crf", 18))
        self.nvenc_check.setChecked(settings.get("default_nvenc", True))
    
    def save_settings(self) -> None:
        """Save settings from UI."""
        settings.set("default_encoding_preset", self.encoding_preset_combo.currentText())
        settings.set("default_crf", self.crf_spin.value())
        settings.set("default_nvenc", self.nvenc_check.isChecked())


class AdvancedTab(QWidget):
    """Advanced configuration tab."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Setup advanced tab UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        
        # Platform info (read-only)
        platform_group = QGroupBox("Platform Information")
        platform_layout = QFormLayout(platform_group)
        
        self.platform_label = QLabel()
        self.platform_label.setStyleSheet("color: #bbb;")
        platform_layout.addRow("Platform:", self.platform_label)
        
        self.wsl_label = QLabel()
        self.wsl_label.setStyleSheet("color: #bbb;")
        platform_layout.addRow("WSL Detected:", self.wsl_label)
        
        layout.addWidget(platform_group)
        
        # GPU info (read-only)
        gpu_group = QGroupBox("GPU Information")
        gpu_layout = QVBoxLayout(gpu_group)
        
        self.gpu_info_label = QLabel("Detecting...")
        self.gpu_info_label.setStyleSheet("color: #bbb; font-family: monospace;")
        self.gpu_info_label.setWordWrap(True)
        gpu_layout.addWidget(self.gpu_info_label)
        
        refresh_button = QPushButton("Refresh GPU Info")
        refresh_button.clicked.connect(self._refresh_gpu_info)
        gpu_layout.addWidget(refresh_button)
        
        layout.addWidget(gpu_group)
        
        layout.addStretch()
    
    def _refresh_gpu_info(self) -> None:
        """Refresh GPU information display."""
        self.gpu_info_label.setText("Detecting...")
        
        gpu_info = settings.get_gpu_info()
        
        if gpu_info["detected"] and gpu_info["gpus"]:
            info_text = "GPUs detected:\n"
            for gpu in gpu_info["gpus"]:
                info_text += f"  GPU {gpu['id']}: {gpu['name']} ({gpu['memory']})\n"
        else:
            info_text = "No NVIDIA GPUs detected or nvidia-smi not available."
        
        self.gpu_info_label.setText(info_text.strip())
    
    def load_settings(self) -> None:
        """Load settings into UI."""
        self.platform_label.setText(settings.get("platform", "Unknown"))
        self.wsl_label.setText("Yes" if settings.get("is_wsl", False) else "No")
        self._refresh_gpu_info()
    
    def save_settings(self) -> None:
        """Save settings from UI (nothing to save in advanced tab)."""
        pass