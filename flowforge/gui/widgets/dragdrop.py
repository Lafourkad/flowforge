"""
Drag & Drop Video File Widget
Modern drag and drop zone with video file validation.
"""

import json
import subprocess
from pathlib import Path
from typing import Optional, List

from PyQt6.QtCore import Qt, pyqtSignal, QMimeData, QSize
from PyQt6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent, QPainter, QPen, QBrush, QColor, QFont, QPixmap
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QSizePolicy


class VideoDropZone(QWidget):
    """Drag and drop zone for video files with preview and info display."""
    
    file_dropped = pyqtSignal(str)  # Emitted when a valid video file is dropped/selected
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setAcceptDrops(True)
        self.setMinimumSize(400, 200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        
        # State
        self._current_file: Optional[Path] = None
        self._video_info: Optional[dict] = None
        self._is_hovering = False
        
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Setup the UI layout."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)
        
        # Drop zone area
        self.drop_area = QWidget()
        self.drop_area.setMinimumHeight(120)
        self.drop_area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding)
        
        # Drop zone layout
        drop_layout = QVBoxLayout(self.drop_area)
        drop_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drop_layout.setSpacing(8)
        
        # Icon/status label
        self.status_label = QLabel("📹")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("font-size: 48px; color: #666;")
        drop_layout.addWidget(self.status_label)
        
        # Main text
        self.main_text = QLabel("Drop video files here")
        self.main_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = self.main_text.font()
        font.setPointSize(14)
        font.setBold(True)
        self.main_text.setFont(font)
        self.main_text.setStyleSheet("color: #dcdcdc; margin: 4px;")
        drop_layout.addWidget(self.main_text)
        
        # Subtitle text
        self.sub_text = QLabel("or click to browse")
        self.sub_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sub_text.setStyleSheet("color: #999; font-size: 12px;")
        drop_layout.addWidget(self.sub_text)
        
        # Browse button
        self.browse_button = QPushButton("Browse Files")
        self.browse_button.setMaximumWidth(120)
        self.browse_button.clicked.connect(self._browse_files)
        self.browse_button.setStyleSheet("""
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
        """)
        
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.browse_button)
        button_layout.addStretch()
        drop_layout.addLayout(button_layout)
        
        layout.addWidget(self.drop_area)
        
        # Video info panel (initially hidden)
        self.info_widget = VideoInfoWidget()
        self.info_widget.hide()
        layout.addWidget(self.info_widget)
        
        # Clear button (initially hidden)
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self._clear_selection)
        self.clear_button.hide()
        layout.addWidget(self.clear_button)
    
    def paintEvent(self, event) -> None:
        """Custom paint event for drag and drop visual feedback."""
        super().paintEvent(event)
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw border
        rect = self.drop_area.geometry()
        rect.adjust(2, 2, -2, -2)
        
        pen = QPen()
        pen.setWidth(2)
        pen.setStyle(Qt.PenStyle.DashLine)
        
        if self._is_hovering:
            pen.setColor(QColor(42, 130, 218))  # Blue when hovering
            brush = QBrush(QColor(42, 130, 218, 20))  # Semi-transparent blue
            painter.setBrush(brush)
        else:
            pen.setColor(QColor(100, 100, 100))  # Gray normally
        
        painter.setPen(pen)
        painter.drawRoundedRect(rect, 8, 8)
        
        painter.end()
    
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """Handle drag enter events."""
        if self._has_valid_video_files(event.mimeData()):
            event.acceptProposedAction()
            self._is_hovering = True
            self._update_hover_state(True)
            self.update()
    
    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        """Handle drag move events."""
        if self._has_valid_video_files(event.mimeData()):
            event.acceptProposedAction()
    
    def dragLeaveEvent(self, event) -> None:
        """Handle drag leave events."""
        self._is_hovering = False
        self._update_hover_state(False)
        self.update()
    
    def dropEvent(self, event: QDropEvent) -> None:
        """Handle drop events."""
        self._is_hovering = False
        self._update_hover_state(False)
        
        mime_data = event.mimeData()
        
        if mime_data.hasUrls():
            urls = mime_data.urls()
            for url in urls:
                if url.isLocalFile():
                    file_path = Path(url.toLocalFile())
                    if self._is_video_file(file_path):
                        self._load_video_file(file_path)
                        event.acceptProposedAction()
                        self.update()
                        return
        
        event.ignore()
    
    def _has_valid_video_files(self, mime_data: QMimeData) -> bool:
        """Check if mime data contains valid video files."""
        if not mime_data.hasUrls():
            return False
        
        for url in mime_data.urls():
            if url.isLocalFile():
                file_path = Path(url.toLocalFile())
                if self._is_video_file(file_path):
                    return True
        
        return False
    
    def _is_video_file(self, file_path: Path) -> bool:
        """Check if file is a supported video format."""
        video_extensions = {
            '.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', 
            '.m4v', '.mpg', '.mpeg', '.3gp', '.ogv', '.ts', '.mts'
        }
        return file_path.suffix.lower() in video_extensions
    
    def _update_hover_state(self, hovering: bool) -> None:
        """Update UI for hover state."""
        if hovering:
            self.main_text.setText("Release to load video")
            self.status_label.setText("📥")
        else:
            if self._current_file:
                self.main_text.setText(f"Current: {self._current_file.name}")
                self.status_label.setText("✅")
            else:
                self.main_text.setText("Drop video files here")
                self.status_label.setText("📹")
    
    def _browse_files(self) -> None:
        """Open file browser dialog."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.mkv *.avi *.mov *.wmv *.flv *.webm *.m4v *.mpg *.mpeg *.3gp *.ogv *.ts *.mts);;All Files (*)"
        )
        
        if file_path:
            self._load_video_file(Path(file_path))
    
    def _load_video_file(self, file_path: Path) -> None:
        """Load and analyze video file."""
        self._current_file = file_path
        
        # Update UI immediately
        self.main_text.setText(f"Loading: {file_path.name}")
        self.status_label.setText("⏳")
        
        # Get video info
        self._video_info = self._get_video_info(file_path)
        
        if self._video_info:
            # Update UI with loaded state
            self.main_text.setText(f"Current: {file_path.name}")
            self.status_label.setText("✅")
            
            # Show video info
            self.info_widget.set_video_info(self._video_info, file_path)
            self.info_widget.show()
            self.clear_button.show()
            
            # Emit signal
            self.file_dropped.emit(str(file_path))
        else:
            # Failed to load
            self.main_text.setText("Error: Could not analyze video")
            self.status_label.setText("❌")
            self._current_file = None
    
    def _find_ffprobe(self) -> str:
        """Find ffprobe binary."""
        import sys
        # Check next to the exe / script
        if getattr(sys, 'frozen', False):
            app_dir = Path(sys.executable).parent
        else:
            app_dir = Path(__file__).resolve().parent.parent.parent.parent
        
        candidates = [
            app_dir / "bin" / "ffprobe.exe",
            app_dir / "bin" / "ffprobe",
            Path(r"C:\Users\Kad\Desktop\FlowForge\bin\ffprobe.exe"),
            Path.home() / ".flowforge" / "bin" / "ffprobe.exe",
        ]
        for c in candidates:
            if c.exists():
                return str(c)
        return "ffprobe"  # fallback to PATH

    def _get_video_info(self, file_path: Path) -> Optional[dict]:
        """Get video metadata using ffprobe."""
        try:
            ffprobe = self._find_ffprobe()
            cmd = [
                ffprobe, "-v", "quiet", "-print_format", "json",
                "-show_streams", "-show_format", str(file_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                return None
            
            data = json.loads(result.stdout)
            
            video_stream = None
            audio_streams = []
            
            for stream in data["streams"]:
                if stream["codec_type"] == "video" and not video_stream:
                    video_stream = stream
                elif stream["codec_type"] == "audio":
                    audio_streams.append(stream)
            
            if not video_stream:
                return None
            
            # Calculate FPS
            fps_str = video_stream.get("r_frame_rate", "0/1")
            fps_parts = fps_str.split("/")
            fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 and fps_parts[1] != '0' else 0
            
            # Calculate duration and frame count
            duration = float(data["format"].get("duration", 0))
            frame_count = int(duration * fps) if duration > 0 and fps > 0 else 0
            
            return {
                "width": int(video_stream.get("width", 0)),
                "height": int(video_stream.get("height", 0)),
                "fps": fps,
                "duration": duration,
                "frame_count": frame_count,
                "codec": video_stream.get("codec_name", "unknown"),
                "pix_fmt": video_stream.get("pix_fmt", "unknown"),
                "bitrate": int(data["format"].get("bit_rate", 0)),
                "size": file_path.stat().st_size,
                "audio_tracks": len(audio_streams),
                "format": data["format"].get("format_name", "unknown")
            }
        
        except (subprocess.SubprocessError, json.JSONDecodeError, KeyError, ValueError, OSError):
            return None
    
    def _clear_selection(self) -> None:
        """Clear the current video selection."""
        self._current_file = None
        self._video_info = None
        
        # Reset UI
        self.main_text.setText("Drop video files here")
        self.status_label.setText("📹")
        self.info_widget.hide()
        self.clear_button.hide()
        
        self.update()
    
    def get_current_file(self) -> Optional[Path]:
        """Get the currently selected video file."""
        return self._current_file
    
    def get_video_info(self) -> Optional[dict]:
        """Get information about the current video."""
        return self._video_info


class VideoInfoWidget(QWidget):
    """Widget for displaying video information."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Setup the video info display."""
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(16, 12, 16, 12)
        
        # Header
        header = QLabel("Video Information")
        header.setStyleSheet("font-size: 14px; font-weight: bold; color: #dcdcdc; margin-bottom: 8px;")
        layout.addWidget(header)
        
        # Info grid
        info_layout = QHBoxLayout()
        
        # Left column
        left_col = QVBoxLayout()
        self.resolution_label = QLabel()
        self.fps_label = QLabel()
        self.duration_label = QLabel()
        left_col.addWidget(self.resolution_label)
        left_col.addWidget(self.fps_label)
        left_col.addWidget(self.duration_label)
        
        # Right column
        right_col = QVBoxLayout()
        self.codec_label = QLabel()
        self.audio_label = QLabel()
        self.size_label = QLabel()
        right_col.addWidget(self.codec_label)
        right_col.addWidget(self.audio_label)
        right_col.addWidget(self.size_label)
        
        info_layout.addLayout(left_col)
        info_layout.addStretch()
        info_layout.addLayout(right_col)
        
        layout.addLayout(info_layout)
        
        # Style all labels
        for widget in [self.resolution_label, self.fps_label, self.duration_label,
                      self.codec_label, self.audio_label, self.size_label]:
            widget.setStyleSheet("color: #bbb; font-size: 12px; margin: 2px;")
    
    def set_video_info(self, info: dict, file_path: Path) -> None:
        """Update the display with video information."""
        # Format duration
        duration = info.get("duration", 0)
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)
        duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours > 0 else f"{minutes:02d}:{seconds:02d}"
        
        # Format file size
        size_bytes = info.get("size", 0)
        if size_bytes >= 1024**3:  # GB
            size_str = f"{size_bytes / (1024**3):.1f} GB"
        elif size_bytes >= 1024**2:  # MB
            size_str = f"{size_bytes / (1024**2):.1f} MB"
        else:  # KB
            size_str = f"{size_bytes / 1024:.1f} KB"
        
        # Update labels
        self.resolution_label.setText(f"Resolution: {info.get('width', 0)}×{info.get('height', 0)}")
        self.fps_label.setText(f"Frame Rate: {info.get('fps', 0):.3f} fps")
        self.duration_label.setText(f"Duration: {duration_str}")
        self.codec_label.setText(f"Codec: {info.get('codec', 'unknown')}")
        self.audio_label.setText(f"Audio Tracks: {info.get('audio_tracks', 0)}")
        self.size_label.setText(f"File Size: {size_str}")