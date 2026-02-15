"""
RIFE Player VLC-Style Player Window
Main application window with VLC-inspired layout and controls.
"""

from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, 
    QMenuBar, QMenu, QToolBar, QStatusBar, QFileDialog,
    QMessageBox, QPushButton, QLabel, QSizePolicy, QFrame
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QAction, QKeySequence, QFont, QIcon

from .settings import FlowForgeSettings, SettingsDialog
from .drop_zone import DropZoneWidget
from .rife_toolbar import RifeToolbar
from .playback import PlaybackWorker
from .export_dialog import ExportDialog
from .utils import get_video_resolution_name, format_duration


class FlowForgePlayer(QMainWindow):
    """
    VLC-style main player window.
    Layout: Menu bar → Main area → RIFE toolbar → Playback controls
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Load settings
        self.settings = FlowForgeSettings.load()
        
        # State
        self.current_video_path: Optional[Path] = None
        self.playback_worker: Optional[PlaybackWorker] = None
        self.playback_thread: Optional[QThread] = None
        
        self.setup_window()
        self.setup_ui()
        self.setup_menu_bar()
        self.setup_status_bar()
        
        # Apply initial settings
        self.resize(self.settings.window_width, self.settings.window_height)
    
    def setup_window(self) -> None:
        """Setup main window properties."""
        self.setWindowTitle("RIFE Player")
        self.setMinimumSize(700, 500)
        
        # Center on screen
        screen_geometry = self.screen().geometry()
        x = (screen_geometry.width() - self.settings.window_width) // 2
        y = (screen_geometry.height() - self.settings.window_height) // 2
        self.move(x, y)
    
    def setup_ui(self) -> None:
        """Setup the main UI layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main vertical layout
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Main drop zone area (large, center)
        self.drop_zone = DropZoneWidget()
        self.drop_zone.file_dropped.connect(self.load_video)
        layout.addWidget(self.drop_zone, 1)  # Takes most space
        
        # RIFE toolbar (slim bar below main area)
        self.rife_toolbar = RifeToolbar(self.settings)
        self.rife_toolbar.rife_toggled.connect(self.on_rife_toggled)
        self.rife_toolbar.preset_changed.connect(self.on_preset_changed)
        layout.addWidget(self.rife_toolbar)
        
        # Playback controls (bottom bar, VLC-style)
        self.playback_controls = self.create_playback_controls()
        layout.addWidget(self.playback_controls)
    
    def setup_menu_bar(self) -> None:
        """Setup the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        open_action = QAction("&Open...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)
        
        # Recent files (placeholder - could implement later)
        recent_menu = QMenu("Recent", self)
        recent_menu.addAction("(No recent files)").setEnabled(False)
        file_menu.addMenu(recent_menu)
        
        file_menu.addSeparator()
        
        export_action = QAction("&Export...", self)
        export_action.setShortcut(QKeySequence("Ctrl+E"))
        export_action.triggered.connect(self.open_export_dialog)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)
        
        # Playback menu
        playback_menu = menubar.addMenu("&Playback")
        
        play_action = QAction("&Play", self)
        play_action.setShortcut(QKeySequence("Space"))
        play_action.triggered.connect(self.play_video)
        playback_menu.addAction(play_action)
        
        stop_action = QAction("&Stop", self)
        stop_action.setShortcut(QKeySequence("S"))
        stop_action.triggered.connect(self.stop_playback)
        playback_menu.addAction(stop_action)
        
        # RIFE menu
        rife_menu = menubar.addMenu("&RIFE")
        
        toggle_rife_action = QAction("&Enable RIFE", self)
        toggle_rife_action.setCheckable(True)
        toggle_rife_action.setShortcut(QKeySequence("R"))
        toggle_rife_action.triggered.connect(self.toggle_rife)
        rife_menu.addAction(toggle_rife_action)
        self.toggle_rife_action = toggle_rife_action
        
        rife_menu.addSeparator()
        
        # Preset submenu
        presets_menu = QMenu("Presets", self)
        for preset in ["Film 60fps", "Anime 60fps", "Sports 60fps", "Smooth 144fps", "Custom"]:
            action = QAction(preset, self)
            action.triggered.connect(lambda checked, p=preset: self.set_preset(p))
            presets_menu.addAction(action)
        rife_menu.addMenu(presets_menu)
        
        rife_menu.addSeparator()
        
        settings_action = QAction("&Settings...", self)
        settings_action.triggered.connect(self.open_settings)
        rife_menu.addAction(settings_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        about_action = QAction("&About RIFE Player", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def setup_status_bar(self) -> None:
        """Setup the status bar."""
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")
    
    def create_playback_controls(self) -> QWidget:
        """Create VLC-style playback controls."""
        controls_frame = QFrame()
        controls_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        controls_frame.setStyleSheet("""
            QFrame {
                background-color: #2d2d2d;
                border-top: 1px solid #3c3c3c;
                padding: 8px;
            }
        """)
        
        layout = QHBoxLayout(controls_frame)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(12)
        
        # Big play button (VLC-style)
        self.play_button = QPushButton("▶")
        self.play_button.setFixedSize(50, 40)
        play_font = QFont()
        play_font.setPointSize(16)
        self.play_button.setFont(play_font)
        self.play_button.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                border: 1px solid #106ebe;
                border-radius: 6px;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1084d4;
            }
            QPushButton:pressed {
                background-color: #006cc4;
            }
            QPushButton:disabled {
                background-color: #404040;
                border-color: #505050;
                color: #808080;
            }
        """)
        self.play_button.clicked.connect(self.play_video)
        self.play_button.setEnabled(False)
        layout.addWidget(self.play_button)
        
        # Video title/info
        info_layout = QVBoxLayout()
        
        self.video_title = QLabel("No video loaded")
        title_font = QFont()
        title_font.setPointSize(11)
        title_font.setBold(True)
        self.video_title.setFont(title_font)
        self.video_title.setStyleSheet("color: #ffffff;")
        info_layout.addWidget(self.video_title)
        
        self.video_info = QLabel("Select a video file to get started")
        self.video_info.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        info_layout.addWidget(self.video_info)
        
        layout.addLayout(info_layout)
        
        # Flexible spacer
        layout.addStretch()
        
        # RIFE status badge
        self.rife_badge = QLabel("RIFE OFF")
        self.rife_badge.setStyleSheet("""
            QLabel {
                background-color: #404040;
                color: #aaaaaa;
                border: 1px solid #505050;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 10px;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.rife_badge)
        
        return controls_frame
    
    def open_file(self) -> None:
        """Open file dialog to select video."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video File",
            str(Path.home()),
            "Video Files (*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.m4v *.webm *.ts *.mts *.m2ts);;All Files (*)"
        )
        
        if file_path:
            self.load_video(file_path)
    
    def load_video(self, file_path: str) -> None:
        """Load video file."""
        video_path = Path(file_path)
        
        if not video_path.exists():
            QMessageBox.warning(self, "File Not Found", f"Video file not found:\n{file_path}")
            return
        
        # Load video in drop zone
        self.drop_zone.load_video(video_path)
        self.current_video_path = video_path
        
        # Update UI
        self.video_title.setText(video_path.name)
        
        # Get video info for display
        video_info = self.drop_zone.video_info.video_info
        if video_info:
            resolution = get_video_resolution_name(video_info.get("width"), video_info.get("height"))
            duration = format_duration(video_info.get("duration"))
            fps = video_info.get("fps", "Unknown")
            
            info_text = f"{resolution}"
            if duration != "Unknown":
                info_text += f" • {duration}"
            if fps != "Unknown":
                info_text += f" • {fps} fps"
            
            self.video_info.setText(info_text)
        else:
            self.video_info.setText("Video information unavailable")
        
        # Enable play button
        self.play_button.setEnabled(True)
        
        self.status_bar.showMessage(f"Loaded: {video_path.name}")
    
    def play_video(self) -> None:
        """Start video playback."""
        if not self.current_video_path:
            QMessageBox.information(self, "No Video", "Please load a video file first.")
            return
        
        # Stop any existing playback
        self.stop_playback()
        
        # Get RIFE configuration
        rife_config = self.rife_toolbar.get_rife_config()
        
        # Validate RIFE paths if enabled
        if rife_config["enabled"]:
            missing_paths = []
            if not rife_config["rife_binary_path"] or not Path(rife_config["rife_binary_path"]).exists():
                missing_paths.append("RIFE binary")
            if not rife_config["rife_model_path"] or not Path(rife_config["rife_model_path"]).exists():
                missing_paths.append("RIFE model")
            if not rife_config["vs_plugin_path"] or not Path(rife_config["vs_plugin_path"]).exists():
                missing_paths.append("VapourSynth plugin")
            
            if missing_paths:
                missing_text = ", ".join(missing_paths)
                reply = QMessageBox.question(
                    self, "Missing RIFE Components",
                    f"The following RIFE components are missing:\n{missing_text}\n\nPlay without RIFE?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
                
                if reply == QMessageBox.StandardButton.Yes:
                    rife_config["enabled"] = False
                    self.rife_toolbar.set_rife_enabled(False)
                else:
                    return
        
        # Create and start playback worker
        self.playback_worker = PlaybackWorker(self.current_video_path, rife_config)
        self.playback_thread = QThread()
        self.playback_worker.moveToThread(self.playback_thread)
        
        # Connect signals
        self.playback_thread.started.connect(self.playback_worker.start_playback)
        self.playback_worker.playback_started.connect(self.on_playback_started)
        self.playback_worker.playback_finished.connect(self.on_playback_finished)
        self.playback_worker.error_occurred.connect(self.on_playback_error)
        
        # Start playback
        self.playback_thread.start()
    
    def stop_playback(self) -> None:
        """Stop current playback."""
        if self.playback_worker and self.playback_thread:
            self.playback_worker.stop_playback()
            self.playback_thread.quit()
            self.playback_thread.wait(5000)  # Wait up to 5 seconds
            
            self.playback_worker = None
            self.playback_thread = None
            
            self.status_bar.showMessage("Playback stopped")
    
    def on_playback_started(self) -> None:
        """Handle playback started."""
        rife_status = "with RIFE" if self.rife_toolbar.rife_enabled else "normal"
        self.status_bar.showMessage(f"Playing {rife_status}...")
    
    def on_playback_finished(self) -> None:
        """Handle playback finished."""
        self.playback_worker = None
        self.playback_thread = None
        self.status_bar.showMessage("Playback finished")
    
    def on_playback_error(self, error_message: str) -> None:
        """Handle playback error."""
        self.playback_worker = None
        self.playback_thread = None
        self.status_bar.showMessage("Playback error")
        QMessageBox.critical(self, "Playback Error", error_message)
    
    def toggle_rife(self) -> None:
        """Toggle RIFE on/off."""
        current_state = self.rife_toolbar.rife_enabled
        self.rife_toolbar.set_rife_enabled(not current_state)
    
    def on_rife_toggled(self, enabled: bool) -> None:
        """Handle RIFE toggle."""
        self.toggle_rife_action.setChecked(enabled)
        self.update_rife_badge()
    
    def on_preset_changed(self, preset: str) -> None:
        """Handle preset change."""
        self.update_rife_badge()
    
    def set_preset(self, preset: str) -> None:
        """Set RIFE preset."""
        self.rife_toolbar.set_preset(preset)
    
    def update_rife_badge(self) -> None:
        """Update RIFE status badge."""
        if self.rife_toolbar.rife_enabled:
            preset = self.rife_toolbar.get_preset()
            if preset == "Custom":
                fps = self.rife_toolbar.get_target_fps()
                badge_text = f"RIFE {fps}fps"
            else:
                badge_text = f"RIFE {preset.split()[1]}"  # Extract FPS from preset name
            
            self.rife_badge.setText(badge_text)
            self.rife_badge.setStyleSheet("""
                QLabel {
                    background-color: #0078d4;
                    color: #ffffff;
                    border: 1px solid #106ebe;
                    border-radius: 4px;
                    padding: 4px 8px;
                    font-size: 10px;
                    font-weight: bold;
                }
            """)
        else:
            self.rife_badge.setText("RIFE OFF")
            self.rife_badge.setStyleSheet("""
                QLabel {
                    background-color: #404040;
                    color: #aaaaaa;
                    border: 1px solid #505050;
                    border-radius: 4px;
                    padding: 4px 8px;
                    font-size: 10px;
                    font-weight: bold;
                }
            """)
    
    def open_export_dialog(self) -> None:
        """Open batch export dialog."""
        rife_config = self.rife_toolbar.get_rife_config()
        
        dialog = ExportDialog(rife_config, self)
        
        # Pre-load current video if available
        if self.current_video_path:
            dialog.add_file(self.current_video_path)
        
        dialog.exec()
    
    def open_settings(self) -> None:
        """Open settings dialog."""
        dialog = SettingsDialog(self.settings, self)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Reload settings
            self.settings = FlowForgeSettings.load()
            
            # Update RIFE toolbar with new settings
            self.rife_toolbar.settings = self.settings
            self.rife_toolbar.update_status()
            
            QMessageBox.information(self, "Settings", "Settings saved successfully.")
    
    def show_about(self) -> None:
        """Show about dialog."""
        QMessageBox.about(self, "About RIFE Player", 
                         "RIFE Player\nVLC-style video player with RIFE interpolation\n\nVersion 1.0.0")
    
    def closeEvent(self, event) -> None:
        """Handle window close event."""
        # Save current window size
        self.settings.window_width = self.width()
        self.settings.window_height = self.height()
        self.settings.save()
        
        # Stop any running playback
        self.stop_playback()
        
        event.accept()