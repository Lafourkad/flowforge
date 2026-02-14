"""
Progress Display Widget
Real-time progress tracking for video processing.
"""

import time
from typing import Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QProgressBar, QPushButton, QFrame)
from PyQt6.QtGui import QFont


class ProcessingProgressWidget(QWidget):
    """Widget for displaying real-time processing progress."""
    
    cancel_requested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._start_time: Optional[float] = None
        self._last_fps_update = 0
        self._frames_processed = 0
        self._total_frames = 0
        self._rife_fps = 0.0
        
        self._setup_ui()
        self._setup_timer()
    
    def _setup_ui(self) -> None:
        """Setup the progress display UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)
        
        # Title
        self.title_label = QLabel("Processing Video")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        self.title_label.setFont(title_font)
        self.title_label.setStyleSheet("color: #dcdcdc; margin-bottom: 8px;")
        layout.addWidget(self.title_label)
        
        # Status text
        self.status_label = QLabel("Preparing...")
        self.status_label.setStyleSheet("color: #bbb; font-size: 12px; margin-bottom: 8px;")
        layout.addWidget(self.status_label)
        
        # Main progress bar
        self.main_progress = QProgressBar()
        self.main_progress.setMinimum(0)
        self.main_progress.setMaximum(100)
        self.main_progress.setValue(0)
        self.main_progress.setTextVisible(True)
        self.main_progress.setMinimumHeight(24)
        layout.addWidget(self.main_progress)
        
        # Stats grid
        stats_frame = QFrame()
        stats_frame.setFrameStyle(QFrame.Shape.Box)
        stats_frame.setStyleSheet("""
            QFrame {
                background-color: #2d2d2d;
                border: 1px solid #3c3c3c;
                border-radius: 6px;
                padding: 8px;
            }
        """)
        
        stats_layout = QVBoxLayout(stats_frame)
        stats_layout.setSpacing(6)
        
        # Row 1: Progress stats
        row1 = QHBoxLayout()
        
        self.percentage_label = QLabel("0%")
        self.percentage_label.setStyleSheet("font-weight: bold; color: #2a82da;")
        
        self.eta_label = QLabel("ETA: --:--")
        self.eta_label.setStyleSheet("color: #bbb;")
        
        self.elapsed_label = QLabel("Elapsed: 00:00")
        self.elapsed_label.setStyleSheet("color: #bbb;")
        
        row1.addWidget(self.percentage_label)
        row1.addStretch()
        row1.addWidget(self.eta_label)
        row1.addStretch() 
        row1.addWidget(self.elapsed_label)
        
        stats_layout.addLayout(row1)
        
        # Row 2: Processing stats
        row2 = QHBoxLayout()
        
        self.frames_label = QLabel("Frames: 0/0")
        self.frames_label.setStyleSheet("color: #bbb; font-size: 11px;")
        
        self.rife_fps_label = QLabel("RIFE: 0.0 fps")
        self.rife_fps_label.setStyleSheet("color: #bbb; font-size: 11px;")
        
        self.phase_label = QLabel("Phase: Preparing")
        self.phase_label.setStyleSheet("color: #bbb; font-size: 11px;")
        
        row2.addWidget(self.frames_label)
        row2.addStretch()
        row2.addWidget(self.rife_fps_label)
        row2.addStretch()
        row2.addWidget(self.phase_label)
        
        stats_layout.addLayout(row2)
        
        layout.addWidget(stats_frame)
        
        # Cancel button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setMaximumWidth(100)
        self.cancel_button.clicked.connect(self.cancel_requested.emit)
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #d32f2f;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e53935;
            }
            QPushButton:pressed {
                background-color: #c62828;
            }
            QPushButton:disabled {
                background-color: #666;
                color: #999;
            }
        """)
        
        button_layout.addWidget(self.cancel_button)
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
    
    def _setup_timer(self) -> None:
        """Setup update timer."""
        self._timer = QTimer()
        self._timer.timeout.connect(self._update_time_display)
        self._timer.start(1000)  # Update every second
    
    def start_processing(self) -> None:
        """Start processing and reset all counters."""
        self._start_time = time.time()
        self._last_fps_update = 0
        self._frames_processed = 0
        self._total_frames = 0
        self._rife_fps = 0.0
        
        self.main_progress.setValue(0)
        self.cancel_button.setEnabled(True)
    
    def update_progress(self, percentage: int, status: str, extra_info: dict = None) -> None:
        """Update the progress display."""
        # Update main progress
        self.main_progress.setValue(min(100, max(0, percentage)))
        self.status_label.setText(status)
        
        # Update percentage
        self.percentage_label.setText(f"{percentage}%")
        
        # Update phase based on percentage
        if percentage < 15:
            phase = "Analyzing"
        elif percentage < 30:
            phase = "Extracting"
        elif percentage < 35:
            phase = "Scene Detection"
        elif percentage < 85:
            phase = "RIFE Interpolation"
        elif percentage < 100:
            phase = "Encoding"
        else:
            phase = "Complete"
        
        self.phase_label.setText(f"Phase: {phase}")
        
        # Update extra info if provided
        if extra_info:
            if "rife_fps" in extra_info:
                self._rife_fps = extra_info["rife_fps"]
                self.rife_fps_label.setText(f"RIFE: {self._rife_fps:.1f} fps")
    
    def update_frame_count(self, current: int, total: int = None) -> None:
        """Update frame processing counters."""
        self._frames_processed = current
        if total is not None:
            self._total_frames = total
        
        self.frames_label.setText(f"Frames: {self._frames_processed}/{self._total_frames}")
    
    def update_rife_progress(self, current: int, total: int, fps: float) -> None:
        """Update RIFE-specific progress."""
        self.update_frame_count(current, total)
        self._rife_fps = fps
        self.rife_fps_label.setText(f"RIFE: {fps:.1f} fps")
    
    def set_encoding_status(self, encoder_name: str) -> None:
        """Update status for encoding phase."""
        self.status_label.setText(f"Encoding with {encoder_name}...")
        self.phase_label.setText("Phase: Encoding")
    
    def finish_processing(self, success: bool, message: str) -> None:
        """Finish processing and show final status."""
        if success:
            self.main_progress.setValue(100)
            self.status_label.setText("✅ " + message)
            self.phase_label.setText("Phase: Complete")
            self.cancel_button.setText("Close")
        else:
            self.status_label.setText("❌ " + message)
            self.phase_label.setText("Phase: Error")
            self.cancel_button.setText("Close")
        
        self.cancel_button.setEnabled(True)
    
    def _update_time_display(self) -> None:
        """Update elapsed time and ETA."""
        if not self._start_time:
            return
        
        elapsed = time.time() - self._start_time
        
        # Format elapsed time
        elapsed_min = int(elapsed // 60)
        elapsed_sec = int(elapsed % 60)
        self.elapsed_label.setText(f"Elapsed: {elapsed_min:02d}:{elapsed_sec:02d}")
        
        # Calculate ETA based on current progress
        progress = self.main_progress.value()
        if progress > 0 and progress < 100:
            total_estimated = (elapsed * 100) / progress
            remaining = total_estimated - elapsed
            
            if remaining > 0:
                eta_min = int(remaining // 60)
                eta_sec = int(remaining % 60)
                self.eta_label.setText(f"ETA: {eta_min:02d}:{eta_sec:02d}")
            else:
                self.eta_label.setText("ETA: --:--")
        else:
            self.eta_label.setText("ETA: --:--")
    
    def set_cancelled(self) -> None:
        """Set the display to cancelled state."""
        self.status_label.setText("❌ Processing cancelled")
        self.phase_label.setText("Phase: Cancelled")
        self.cancel_button.setText("Close")
        self.cancel_button.setEnabled(True)


class CompactProgressWidget(QWidget):
    """Compact progress widget for status bar or small spaces."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Setup compact progress display."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        # Small progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumHeight(16)
        self.progress_bar.setMaximumWidth(100)
        self.progress_bar.setTextVisible(False)
        
        # Status text
        self.status_text = QLabel("Ready")
        self.status_text.setStyleSheet("color: #bbb; font-size: 11px;")
        
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_text)
        layout.addStretch()
    
    def update_progress(self, percentage: int, status: str) -> None:
        """Update compact progress display."""
        self.progress_bar.setValue(percentage)
        self.status_text.setText(status)
    
    def reset(self) -> None:
        """Reset to default state."""
        self.progress_bar.setValue(0)
        self.status_text.setText("Ready")