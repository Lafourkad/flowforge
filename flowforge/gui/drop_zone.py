"""
RIFE Player Drop Zone Widget
Main area for drag & drop and video info display.
"""

from pathlib import Path
from typing import Optional, Dict, Any

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QFrame, QSizePolicy, QGridLayout
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap, QPainter, QColor, QDragEnterEvent, QDropEvent

from .utils import (
    probe_video, format_duration, format_filesize, 
    format_bitrate, get_video_resolution_name
)


class VideoInfoWidget(QFrame):
    """Widget to display video information with thumbnail."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.video_info: Optional[Dict[str, Any]] = None
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        self.setup_ui()
    
    def setup_ui(self) -> None:
        """Setup the video info UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # Thumbnail area (left side)
        self.thumbnail_label = QLabel()
        self.thumbnail_label.setFixedSize(200, 112)  # 16:9 aspect ratio
        self.thumbnail_label.setScaledContents(True)
        self.thumbnail_label.setStyleSheet("""
            QLabel {
                background-color: #1a1a1a;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
            }
        """)
        self.thumbnail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.thumbnail_label)
        
        # Info area (right side)
        info_layout = QVBoxLayout()
        
        # File name (large, prominent)
        self.filename_label = QLabel("No video loaded")
        filename_font = QFont()
        filename_font.setPointSize(14)
        filename_font.setBold(True)
        self.filename_label.setFont(filename_font)
        self.filename_label.setWordWrap(True)
        info_layout.addWidget(self.filename_label)
        
        # Technical details grid
        details_grid = QGridLayout()
        details_grid.setHorizontalSpacing(20)
        details_grid.setVerticalSpacing(8)
        
        self.details_labels = {}
        details = [
            ("Resolution:", "resolution"),
            ("Duration:", "duration"), 
            ("Frame Rate:", "fps"),
            ("Codec:", "codec"),
            ("Bitrate:", "bitrate"),
            ("File Size:", "filesize"),
        ]
        
        for i, (label_text, key) in enumerate(details):
            row = i // 2
            col = (i % 2) * 2
            
            label = QLabel(label_text)
            label.setStyleSheet("color: #aaaaaa;")
            details_grid.addWidget(label, row, col)
            
            value_label = QLabel("—")
            value_label.setStyleSheet("color: #ffffff;")
            details_grid.addWidget(value_label, row, col + 1)
            
            self.details_labels[key] = value_label
        
        info_layout.addLayout(details_grid)
        info_layout.addStretch()
        
        layout.addLayout(info_layout)
        layout.addStretch()
    
    def set_video_info(self, video_info: Dict[str, Any]) -> None:
        """Set video information to display."""
        self.video_info = video_info
        
        # Update filename
        self.filename_label.setText(video_info.get("filename", "Unknown"))
        
        # Update details
        width = video_info.get("width")
        height = video_info.get("height")
        resolution = get_video_resolution_name(width, height)
        if width and height:
            resolution += f" ({width}×{height})"
        self.details_labels["resolution"].setText(resolution)
        
        duration = format_duration(video_info.get("duration"))
        self.details_labels["duration"].setText(duration)
        
        fps = video_info.get("fps")
        fps_text = f"{fps} fps" if fps else "Unknown"
        self.details_labels["fps"].setText(fps_text)
        
        codec = video_info.get("codec", "Unknown")
        self.details_labels["codec"].setText(codec.upper() if codec != "Unknown" else codec)
        
        bitrate = format_bitrate(video_info.get("bitrate"))
        self.details_labels["bitrate"].setText(bitrate)
        
        filesize = format_filesize(video_info.get("size"))
        self.details_labels["filesize"].setText(filesize)
        
        # Create placeholder thumbnail (we don't extract actual frames for simplicity)
        self.create_placeholder_thumbnail(width, height)
    
    def create_placeholder_thumbnail(self, width: Optional[int], height: Optional[int]) -> None:
        """Create a placeholder thumbnail with resolution info."""
        pixmap = QPixmap(200, 112)
        pixmap.fill(QColor(26, 26, 26))
        
        painter = QPainter(pixmap)
        painter.setPen(QColor(255, 255, 255))
        
        # Draw video icon placeholder
        font = painter.font()
        font.setPointSize(10)
        font.setBold(True)
        painter.setFont(font)
        
        if width and height:
            resolution_text = f"{width}×{height}"
            painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, 
                           f"📹\n{resolution_text}")
        else:
            painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "📹")
        
        painter.end()
        
        self.thumbnail_label.setPixmap(pixmap)
    
    def clear(self) -> None:
        """Clear video information."""
        self.video_info = None
        self.filename_label.setText("No video loaded")
        
        for label in self.details_labels.values():
            label.setText("—")
        
        # Clear thumbnail
        self.thumbnail_label.clear()
        self.thumbnail_label.setText("📹")
        self.thumbnail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)


class DropZoneWidget(QWidget):
    """
    Main drop zone widget that handles drag & drop and displays video info.
    Shows drop prompt when empty, video info when loaded.
    """
    
    file_dropped = pyqtSignal(str)  # Emitted when file is dropped
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_video_path: Optional[Path] = None
        self.setAcceptDrops(True)
        self.setup_ui()
    
    def setup_ui(self) -> None:
        """Setup the drop zone UI."""
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Create drop prompt and video info widgets
        self.drop_prompt = self.create_drop_prompt()
        self.video_info = VideoInfoWidget()
        
        # Show drop prompt initially
        self.layout.addWidget(self.drop_prompt)
        self.video_info.hide()
        
        # Set size policy
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    
    def create_drop_prompt(self) -> QWidget:
        """Create the drop prompt widget."""
        prompt_widget = QWidget()
        prompt_layout = QVBoxLayout(prompt_widget)
        
        # Large file icon and text
        icon_label = QLabel("🎬")
        icon_font = QFont()
        icon_font.setPointSize(48)
        icon_label.setFont(icon_font)
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_label.setStyleSheet("color: #666666;")
        prompt_layout.addWidget(icon_label)
        
        # Main prompt text
        prompt_label = QLabel("Drop video here or File → Open")
        prompt_font = QFont()
        prompt_font.setPointSize(16)
        prompt_label.setFont(prompt_font)
        prompt_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        prompt_label.setStyleSheet("color: #aaaaaa; margin: 20px;")
        prompt_layout.addWidget(prompt_label)
        
        # Supported formats hint
        formats_label = QLabel("Supported: MP4, AVI, MKV, MOV, WMV, FLV, M4V")
        formats_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        formats_label.setStyleSheet("color: #666666; font-size: 12px;")
        prompt_layout.addWidget(formats_label)
        
        # Center everything
        prompt_layout.addStretch()
        
        return prompt_widget
    
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """Handle drag enter event."""
        if event.mimeData().hasUrls():
            # Check if any URL is a video file
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    file_path = Path(url.toLocalFile())
                    if self.is_video_file(file_path):
                        event.acceptProposedAction()
                        return
        
        event.ignore()
    
    def dropEvent(self, event: QDropEvent) -> None:
        """Handle drop event."""
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    file_path = Path(url.toLocalFile())
                    if self.is_video_file(file_path):
                        self.load_video(file_path)
                        event.acceptProposedAction()
                        return
        
        event.ignore()
    
    def is_video_file(self, file_path: Path) -> bool:
        """Check if file is a supported video format."""
        video_extensions = {
            '.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', 
            '.m4v', '.webm', '.ts', '.mts', '.m2ts'
        }
        return file_path.suffix.lower() in video_extensions
    
    def load_video(self, video_path: Path) -> None:
        """Load and display video information."""
        if not video_path.exists():
            return
        
        # Probe video file
        video_info = probe_video(video_path)
        if not video_info:
            return
        
        # Store current video path
        self.current_video_path = video_path
        
        # Switch from drop prompt to video info
        if not self.video_info.isVisible():
            self.drop_prompt.hide()
            self.layout.addWidget(self.video_info)
            self.video_info.show()
        
        # Update video info display
        self.video_info.set_video_info(video_info)
        
        # Emit signal
        self.file_dropped.emit(str(video_path))
    
    def clear_video(self) -> None:
        """Clear current video and show drop prompt."""
        self.current_video_path = None
        
        if self.video_info.isVisible():
            self.video_info.hide()
            self.layout.removeWidget(self.video_info)
            self.drop_prompt.show()
        
        self.video_info.clear()
    
    def get_current_video_path(self) -> Optional[Path]:
        """Get the currently loaded video path."""
        return self.current_video_path
    
    def has_video(self) -> bool:
        """Check if a video is currently loaded."""
        return self.current_video_path is not None