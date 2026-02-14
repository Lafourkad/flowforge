"""
FlowForge Main Window
The main GUI window with all processing controls.
"""

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QSplitter, QTabWidget, QGroupBox, QLabel, QPushButton,
                            QComboBox, QSpinBox, QDoubleSpinBox, QSlider, QCheckBox,
                            QFileDialog, QMessageBox, QStatusBar, QMenuBar, QMenu,
                            QFrame, QFormLayout, QStackedWidget, QScrollArea, QLineEdit)
from PyQt6.QtGui import QAction, QFont, QIcon
from PyQt6.QtCore import QThread

from .widgets.dragdrop import VideoDropZone
from .widgets.progress import ProcessingProgressWidget, CompactProgressWidget
from .widgets.settings import SettingsDialog
from .worker import VideoProcessorWorker, PlaybackWorker
from .settings import settings


class FlowForgeMainWindow(QMainWindow):
    """Main application window for FlowForge."""
    
    def __init__(self):
        super().__init__()
        
        # State
        self._current_video: Optional[Path] = None
        self._video_info: Optional[dict] = None
        self._processing_worker: Optional[VideoProcessorWorker] = None
        self._playback_worker: Optional[PlaybackWorker] = None
        
        self.setWindowTitle("FlowForge - Video Frame Interpolation")
        self.setMinimumSize(900, 700)
        
        # Load window geometry
        self._load_window_state()
        
        self._setup_ui()
        self._setup_menu()
        self._setup_status_bar()
        self._connect_signals()
        
        # Auto-detect GPU info for status bar
        QTimer.singleShot(1000, self._update_gpu_status)
    
    def _setup_ui(self) -> None:
        """Setup the main UI layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(16)
        main_layout.setContentsMargins(16, 16, 16, 16)
        
        # Main splitter (horizontal)
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel - Video input and controls
        left_panel = self._create_left_panel()
        main_splitter.addWidget(left_panel)
        
        # Right panel - Processing and progress
        right_panel = self._create_right_panel()
        main_splitter.addWidget(right_panel)
        
        # Set splitter proportions
        main_splitter.setStretchFactor(0, 2)
        main_splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(main_splitter)
    
    def _create_left_panel(self) -> QWidget:
        """Create the left control panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(16)
        
        # Video input section
        input_group = QGroupBox("Video Input")
        input_layout = QVBoxLayout(input_group)
        
        self.drop_zone = VideoDropZone()
        self.drop_zone.file_dropped.connect(self._on_video_loaded)
        input_layout.addWidget(self.drop_zone)
        
        layout.addWidget(input_group)
        
        # Playback section
        playback_group = QGroupBox("Real-Time Playback")
        playback_layout = QVBoxLayout(playback_group)
        
        # Play button
        self.play_button = QPushButton("▶️ Play Real-Time")
        self.play_button.setMinimumHeight(40)
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self._start_realtime_playback)
        self.play_button.setStyleSheet("""
            QPushButton {
                background-color: #4caf50;
                border: none;
                border-radius: 8px;
                color: white;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5cbf60;
            }
            QPushButton:pressed {
                background-color: #3c9f43;
            }
            QPushButton:disabled {
                background-color: #666;
                color: #999;
            }
        """)
        playback_layout.addWidget(self.play_button)
        
        # Preset controls
        preset_layout = QFormLayout()
        
        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "Film (24→60fps)",
            "Anime (24→60fps)",
            "Sports (30→60fps)", 
            "Smooth (→144fps)",
            "Custom"
        ])
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        preset_layout.addRow("Preset:", self.preset_combo)
        
        # Custom FPS (initially hidden)
        self.custom_fps_spin = QDoubleSpinBox()
        self.custom_fps_spin.setMinimum(30.0)
        self.custom_fps_spin.setMaximum(240.0)
        self.custom_fps_spin.setValue(60.0)
        self.custom_fps_spin.setSuffix(" fps")
        self.custom_fps_spin.hide()
        self.custom_fps_label = QLabel("Target FPS:")
        self.custom_fps_label.hide()
        preset_layout.addRow(self.custom_fps_label, self.custom_fps_spin)
        
        # Scene detection
        scene_layout = QHBoxLayout()
        self.scene_detection_check = QCheckBox("Scene Detection")
        self.scene_detection_check.setChecked(True)
        scene_layout.addWidget(self.scene_detection_check)
        
        self.scene_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.scene_threshold_slider.setMinimum(10)
        self.scene_threshold_slider.setMaximum(100)
        self.scene_threshold_slider.setValue(30)
        self.scene_threshold_slider.valueChanged.connect(self._update_scene_threshold_label)
        scene_layout.addWidget(self.scene_threshold_slider)
        
        self.scene_threshold_label = QLabel("0.30")
        self.scene_threshold_label.setMinimumWidth(40)
        scene_layout.addWidget(self.scene_threshold_label)
        
        preset_layout.addRow("", scene_layout)
        
        # GPU threads
        self.gpu_threads_spin = QSpinBox()
        self.gpu_threads_spin.setMinimum(1)
        self.gpu_threads_spin.setMaximum(4)
        self.gpu_threads_spin.setValue(2)
        preset_layout.addRow("GPU Threads:", self.gpu_threads_spin)
        
        playback_layout.addLayout(preset_layout)
        
        layout.addWidget(playback_group)
        
        # Export section
        export_group = QGroupBox("Export Settings")
        export_layout = QVBoxLayout(export_group)
        
        # Export button
        self.export_button = QPushButton("🎬 Export Video")
        self.export_button.setMinimumHeight(40)
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._start_export)
        self.export_button.setStyleSheet("""
            QPushButton {
                background-color: #ff9800;
                border: none;
                border-radius: 8px;
                color: white;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #ffa726;
            }
            QPushButton:pressed {
                background-color: #f57c00;
            }
            QPushButton:disabled {
                background-color: #666;
                color: #999;
            }
        """)
        export_layout.addWidget(self.export_button)
        
        # Export controls
        export_controls = QFormLayout()
        
        # Output path
        output_layout = QHBoxLayout()
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setPlaceholderText("Select output location...")
        output_browse = QPushButton("Browse...")
        output_browse.clicked.connect(self._browse_output_path)
        output_layout.addWidget(self.output_path_edit)
        output_layout.addWidget(output_browse)
        export_controls.addRow("Output:", output_layout)
        
        # FPS multiplier
        multiplier_layout = QHBoxLayout()
        self.fps_mode_combo = QComboBox()
        self.fps_mode_combo.addItems(["Multiplier", "Target FPS"])
        self.fps_mode_combo.currentTextChanged.connect(self._on_fps_mode_changed)
        multiplier_layout.addWidget(self.fps_mode_combo)
        
        self.multiplier_spin = QSpinBox()
        self.multiplier_spin.setMinimum(2)
        self.multiplier_spin.setMaximum(8)
        self.multiplier_spin.setValue(2)
        self.multiplier_spin.setSuffix("x")
        multiplier_layout.addWidget(self.multiplier_spin)
        
        self.target_fps_spin = QDoubleSpinBox()
        self.target_fps_spin.setMinimum(30.0)
        self.target_fps_spin.setMaximum(240.0)
        self.target_fps_spin.setValue(60.0)
        self.target_fps_spin.setSuffix(" fps")
        self.target_fps_spin.hide()
        multiplier_layout.addWidget(self.target_fps_spin)
        
        export_controls.addRow("FPS:", multiplier_layout)
        
        # Encoding preset
        self.encoding_preset_combo = QComboBox()
        self.encoding_preset_combo.addItems(["Quality (slow/crf16)", "Balanced (medium/crf18)", "Fast (veryfast/crf20)"])
        self.encoding_preset_combo.setCurrentText("Balanced (medium/crf18)")
        export_controls.addRow("Preset:", self.encoding_preset_combo)
        
        # NVENC toggle
        self.nvenc_check = QCheckBox("Use NVENC hardware encoding")
        self.nvenc_check.setChecked(True)
        export_controls.addRow("", self.nvenc_check)
        
        export_layout.addLayout(export_controls)
        
        layout.addWidget(export_group)
        
        layout.addStretch()
        
        return panel
    
    def _create_right_panel(self) -> QWidget:
        """Create the right processing panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(16)
        
        # Stacked widget for different views
        self.right_stack = QStackedWidget()
        
        # Welcome/idle view
        welcome_widget = QWidget()
        welcome_layout = QVBoxLayout(welcome_widget)
        welcome_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        welcome_label = QLabel("🎬")
        welcome_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        welcome_label.setStyleSheet("font-size: 64px; color: #666; margin: 20px;")
        welcome_layout.addWidget(welcome_label)
        
        welcome_text = QLabel("FlowForge")
        welcome_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = welcome_text.font()
        font.setPointSize(24)
        font.setBold(True)
        welcome_text.setFont(font)
        welcome_text.setStyleSheet("color: #dcdcdc; margin-bottom: 8px;")
        welcome_layout.addWidget(welcome_text)
        
        subtitle = QLabel("GPU-accelerated video frame interpolation using RIFE")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #999; font-size: 14px; margin-bottom: 40px;")
        welcome_layout.addWidget(subtitle)
        
        # Quick start instructions
        instructions = QLabel("""
        <b>Quick Start:</b><br>
        1. Drop a video file or click Browse<br>
        2. Choose your preset and settings<br>
        3. Click "Play Real-Time" for preview<br>
        4. Click "Export Video" for final output
        """)
        instructions.setStyleSheet("color: #bbb; font-size: 12px; line-height: 1.4;")
        instructions.setAlignment(Qt.AlignmentFlag.AlignCenter)
        welcome_layout.addWidget(instructions)
        
        self.right_stack.addWidget(welcome_widget)
        
        # Processing view
        self.progress_widget = ProcessingProgressWidget()
        self.progress_widget.cancel_requested.connect(self._cancel_processing)
        self.right_stack.addWidget(self.progress_widget)
        
        layout.addWidget(self.right_stack)
        
        return panel
    
    def _setup_menu(self) -> None:
        """Setup the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        open_action = QAction("&Open Video...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_video_dialog)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Settings menu
        settings_menu = menubar.addMenu("&Settings")
        
        preferences_action = QAction("&Preferences...", self)
        preferences_action.setShortcut("Ctrl+,")
        preferences_action.triggered.connect(self._show_settings)
        settings_menu.addAction(preferences_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        about_action = QAction("&About FlowForge", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _setup_status_bar(self) -> None:
        """Setup the status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # GPU info label
        self.gpu_status_label = QLabel("GPU: Detecting...")
        self.gpu_status_label.setStyleSheet("color: #bbb; font-size: 11px;")
        self.status_bar.addWidget(self.gpu_status_label)
        
        # RIFE version label
        self.rife_status_label = QLabel("")
        self.rife_status_label.setStyleSheet("color: #bbb; font-size: 11px;")
        self.status_bar.addWidget(self.rife_status_label)
        
        # Processing status
        self.compact_progress = CompactProgressWidget()
        self.status_bar.addPermanentWidget(self.compact_progress)
        
        # Initial status
        self.status_bar.showMessage("Ready")
    
    def _connect_signals(self) -> None:
        """Connect various signals."""
        # Load default settings
        default_preset = settings.get("default_preset", "Film (24→60fps)")
        index = self.preset_combo.findText(default_preset)
        if index >= 0:
            self.preset_combo.setCurrentIndex(index)
        
        # Load other defaults
        self.nvenc_check.setChecked(settings.get("default_nvenc", True))
        self.scene_detection_check.setChecked(settings.get("scene_detection", True))
        
        threshold = settings.get("scene_threshold", 0.3)
        self.scene_threshold_slider.setValue(int(threshold * 100))
        self._update_scene_threshold_label()
    
    def _on_video_loaded(self, video_path: str) -> None:
        """Handle video file loaded."""
        self._current_video = Path(video_path)
        self._video_info = self.drop_zone.get_video_info()
        
        # Enable controls
        self.play_button.setEnabled(True)
        self.export_button.setEnabled(True)
        
        # Auto-generate output path
        if self._current_video:
            output_name = f"{self._current_video.stem}_interpolated{self._current_video.suffix}"
            output_path = self._current_video.parent / output_name
            self.output_path_edit.setText(str(output_path))
        
        self.status_bar.showMessage(f"Loaded: {self._current_video.name}")
    
    def _on_preset_changed(self, preset: str) -> None:
        """Handle preset selection change."""
        is_custom = preset == "Custom"
        self.custom_fps_spin.setVisible(is_custom)
        self.custom_fps_label.setVisible(is_custom)
    
    def _on_fps_mode_changed(self, mode: str) -> None:
        """Handle FPS mode change."""
        is_target = mode == "Target FPS"
        self.multiplier_spin.setVisible(not is_target)
        self.target_fps_spin.setVisible(is_target)
    
    def _update_scene_threshold_label(self) -> None:
        """Update scene threshold label."""
        value = self.scene_threshold_slider.value() / 100.0
        self.scene_threshold_label.setText(f"{value:.2f}")
    
    def _browse_output_path(self) -> None:
        """Browse for output file path."""
        if not self._current_video:
            return
        
        suggested_name = f"{self._current_video.stem}_interpolated{self._current_video.suffix}"
        start_path = str(self._current_video.parent / suggested_name)
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Video As",
            start_path,
            "Video Files (*.mp4 *.mkv *.avi);;All Files (*)"
        )
        
        if file_path:
            self.output_path_edit.setText(file_path)
    
    def _open_video_dialog(self) -> None:
        """Open video file dialog."""
        last_dir = settings.get("last_input_dir", str(Path.home()))
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video File",
            last_dir,
            "Video Files (*.mp4 *.mkv *.avi *.mov *.wmv *.flv *.webm *.m4v *.mpg *.mpeg *.3gp *.ogv *.ts *.mts);;All Files (*)"
        )
        
        if file_path:
            settings.set("last_input_dir", str(Path(file_path).parent))
            self.drop_zone._load_video_file(Path(file_path))
    
    def _start_realtime_playback(self) -> None:
        """Start real-time playback with mpv."""
        if not self._current_video:
            return
        
        # Collect playback settings
        preset = self.preset_combo.currentText()
        custom_fps = self.custom_fps_spin.value() if preset == "Custom" else 60.0
        gpu_threads = self.gpu_threads_spin.value()
        scene_detection = self.scene_detection_check.isChecked()
        scene_threshold = self.scene_threshold_slider.value() / 100.0
        
        # Create and start playback worker
        self._playback_worker = PlaybackWorker()
        self._playback_worker.setup_playback(
            self._current_video,
            preset=preset,
            custom_fps=custom_fps,
            gpu_threads=gpu_threads,
            scene_detection=scene_detection,
            scene_threshold=scene_threshold
        )
        
        self._playback_worker.playback_started.connect(self._on_playback_result)
        self._playback_worker.start()
        
        self.status_bar.showMessage("Starting real-time playback...")
    
    def _on_playback_result(self, success: bool, message: str) -> None:
        """Handle playback result."""
        if success:
            self.status_bar.showMessage("Real-time playback started")
        else:
            self.status_bar.showMessage(f"Playback failed: {message}")
            QMessageBox.warning(self, "Playback Error", f"Failed to start playback:\n{message}")
    
    def _start_export(self) -> None:
        """Start video export process."""
        if not self._current_video:
            return
        
        output_path = self.output_path_edit.text().strip()
        if not output_path:
            QMessageBox.warning(self, "No Output Path", "Please select an output path.")
            return
        
        # Collect export settings
        fps_mode = self.fps_mode_combo.currentText()
        if fps_mode == "Multiplier":
            multiplier = self.multiplier_spin.value()
            target_fps = None
        else:
            multiplier = 2  # Will be recalculated
            target_fps = self.target_fps_spin.value()
        
        encoding_preset_text = self.encoding_preset_combo.currentText()
        preset_map = {
            "Quality (slow/crf16)": "Quality",
            "Balanced (medium/crf18)": "Balanced", 
            "Fast (veryfast/crf20)": "Fast"
        }
        encoding_preset = preset_map.get(encoding_preset_text, "Balanced")
        
        nvenc = self.nvenc_check.isChecked()
        scene_detection = self.scene_detection_check.isChecked()
        scene_threshold = self.scene_threshold_slider.value() / 100.0
        gpu_id = settings.get("default_gpu", 0)
        threads = settings.get("default_threads", "1:2:2")
        
        # Switch to processing view
        self.right_stack.setCurrentWidget(self.progress_widget)
        
        # Create and start processing worker
        self._processing_worker = VideoProcessorWorker()
        self._processing_worker.setup_processing(
            Path(self._current_video),
            Path(output_path),
            fps_multiplier=multiplier,
            target_fps=target_fps,
            gpu_id=gpu_id,
            threads=threads,
            scene_detection=scene_detection,
            scene_threshold=scene_threshold,
            encoding_preset=encoding_preset,
            nvenc=nvenc
        )
        
        # Connect signals
        self._processing_worker.progress_updated.connect(self._on_processing_progress)
        self._processing_worker.processing_finished.connect(self._on_processing_finished)
        self._processing_worker.frame_extracted.connect(self.progress_widget.update_frame_count)
        self._processing_worker.rife_progress.connect(self.progress_widget.update_rife_progress)
        self._processing_worker.encoding_started.connect(self.progress_widget.set_encoding_status)
        
        # Start processing
        self.progress_widget.start_processing()
        self._processing_worker.start()
        
        # Disable UI controls
        self._set_processing_ui_state(True)
        
        self.status_bar.showMessage("Processing started...")
    
    def _on_processing_progress(self, percentage: int, status: str, extra_info: dict) -> None:
        """Handle processing progress updates."""
        self.progress_widget.update_progress(percentage, status, extra_info)
        self.compact_progress.update_progress(percentage, status)
        
        if percentage < 100:
            self.status_bar.showMessage(f"Processing... {percentage}%")
    
    def _on_processing_finished(self, success: bool, message: str) -> None:
        """Handle processing completion."""
        self.progress_widget.finish_processing(success, message)
        self.compact_progress.reset()
        
        if success:
            self.status_bar.showMessage("Processing completed successfully!")
            
            # Show completion dialog
            result = QMessageBox.information(
                self,
                "Export Complete",
                f"Video export completed successfully!\n\n{message}",
                QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Open,
                QMessageBox.StandardButton.Ok
            )
            
            if result == QMessageBox.StandardButton.Open:
                # Open output folder
                output_path = Path(self.output_path_edit.text())
                if output_path.exists():
                    import subprocess
                    import sys
                    
                    if sys.platform == "win32":
                        subprocess.run(["explorer", "/select,", str(output_path)])
                    elif sys.platform == "darwin":
                        subprocess.run(["open", "-R", str(output_path)])
                    else:
                        subprocess.run(["xdg-open", str(output_path.parent)])
        else:
            self.status_bar.showMessage(f"Processing failed: {message}")
            QMessageBox.critical(self, "Processing Error", f"Processing failed:\n{message}")
        
        # Re-enable UI
        self._set_processing_ui_state(False)
        
        # Return to welcome view after a delay
        QTimer.singleShot(3000, lambda: self.right_stack.setCurrentIndex(0))
    
    def _cancel_processing(self) -> None:
        """Cancel ongoing processing."""
        if self._processing_worker and self._processing_worker.isRunning():
            self._processing_worker.stop_processing()
            self.progress_widget.set_cancelled()
            self.status_bar.showMessage("Cancelling processing...")
    
    def _set_processing_ui_state(self, processing: bool) -> None:
        """Enable/disable UI during processing."""
        self.play_button.setEnabled(not processing and self._current_video is not None)
        self.export_button.setEnabled(not processing and self._current_video is not None)
        self.drop_zone.setEnabled(not processing)
        
        # Disable menu items during processing
        if hasattr(self, 'menuBar'):
            self.menuBar().setEnabled(not processing)
    
    def _show_settings(self) -> None:
        """Show settings dialog."""
        dialog = SettingsDialog(self)
        dialog.settings_changed.connect(self._on_settings_changed)
        dialog.exec()
    
    def _on_settings_changed(self) -> None:
        """Handle settings changes."""
        # Update GPU status
        self._update_gpu_status()
        self._update_rife_status()
    
    def _show_about(self) -> None:
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About FlowForge",
            """
            <h3>FlowForge</h3>
            <p>GPU-accelerated video frame interpolation using RIFE</p>
            <p>Version 1.0.0</p>
            <p>Built with PyQt6 and powered by RIFE neural network</p>
            <p><b>Features:</b></p>
            <ul>
            <li>Real-time playback with VapourSynth</li>
            <li>Batch processing with progress tracking</li>
            <li>NVENC hardware encoding support</li>
            <li>Scene detection for better quality</li>
            <li>Multiple quality presets</li>
            </ul>
            """
        )
    
    def _update_gpu_status(self) -> None:
        """Update GPU status in status bar."""
        gpu_info = settings.get_gpu_info()
        
        if gpu_info["detected"] and gpu_info["gpus"]:
            gpu_text = f"GPU: {gpu_info['gpus'][0]['name']}"
            if len(gpu_info['gpus']) > 1:
                gpu_text += f" (+{len(gpu_info['gpus'])-1} more)"
        else:
            gpu_text = "GPU: Not detected"
        
        self.gpu_status_label.setText(gpu_text)
    
    def _update_rife_status(self) -> None:
        """Update RIFE status in status bar."""
        rife_version = settings.get_rife_version()
        if rife_version:
            self.rife_status_label.setText(f"RIFE: {rife_version}")
        else:
            self.rife_status_label.setText("RIFE: Not found")
    
    def _load_window_state(self) -> None:
        """Load saved window geometry and state."""
        geometry = settings.get("window_geometry")
        if geometry:
            try:
                # geometry is expected to be [x, y, width, height]
                if len(geometry) == 4:
                    self.setGeometry(*geometry)
            except (TypeError, ValueError):
                pass  # Use default geometry
        
        window_state = settings.get("window_state")
        if window_state:
            try:
                # This would be window state bytes, but for simplicity we'll skip
                pass
            except:
                pass
    
    def _save_window_state(self) -> None:
        """Save current window geometry and state."""
        geometry = self.geometry()
        settings.set("window_geometry", [geometry.x(), geometry.y(), geometry.width(), geometry.height()])
        settings.save_settings()
    
    def closeEvent(self, event) -> None:
        """Handle window close event."""
        # Stop any running workers
        if self._processing_worker and self._processing_worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Processing In Progress",
                "Video processing is still running. Do you want to cancel and exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self._processing_worker.stop_processing()
                self._processing_worker.wait(3000)  # Wait up to 3 seconds
                if self._processing_worker.isRunning():
                    self._processing_worker.terminate()
            else:
                event.ignore()
                return
        
        # Save window state
        self._save_window_state()
        
        event.accept()