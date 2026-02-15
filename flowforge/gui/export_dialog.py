"""
RIFE Player Export Dialog
Modal dialog for batch export with progress tracking.
"""

from pathlib import Path
from typing import List, Dict, Any

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QPushButton, QListWidget, QListWidgetItem,
    QProgressBar, QTextEdit, QGroupBox, QFileDialog, QMessageBox,
    QSizePolicy, QCheckBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont

from .playback import ExportWorker
from .utils import format_filesize, get_video_resolution_name


class ExportDialog(QDialog):
    """Modal dialog for batch video export with RIFE processing."""
    
    def __init__(self, rife_config: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.rife_config = rife_config
        self.input_files: List[Path] = []
        self.output_dir: Path = Path.home() / "Videos" / "RIFE_Player_Export"
        self.export_worker: ExportWorker = None
        self.export_thread: QThread = None
        
        self.setWindowTitle("RIFE Player - Batch Export")
        self.setModal(True)
        self.resize(600, 500)
        
        self.setup_ui()
    
    def setup_ui(self) -> None:
        """Setup the export dialog UI."""
        layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel("Batch Export with RIFE")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        layout.addWidget(title_label)
        
        # Input files section
        input_group = QGroupBox("Input Files")
        input_layout = QVBoxLayout(input_group)
        
        # File list
        self.file_list = QListWidget()
        self.file_list.setMinimumHeight(150)
        input_layout.addWidget(self.file_list)
        
        # File buttons
        file_buttons_layout = QHBoxLayout()
        
        add_files_btn = QPushButton("Add Files...")
        add_files_btn.clicked.connect(self.add_files)
        file_buttons_layout.addWidget(add_files_btn)
        
        add_folder_btn = QPushButton("Add Folder...")
        add_folder_btn.clicked.connect(self.add_folder)
        file_buttons_layout.addWidget(add_folder_btn)
        
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear_files)
        file_buttons_layout.addWidget(clear_btn)
        
        file_buttons_layout.addStretch()
        
        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self.remove_selected)
        file_buttons_layout.addWidget(remove_btn)
        
        input_layout.addLayout(file_buttons_layout)
        layout.addWidget(input_group)
        
        # Output settings section
        output_group = QGroupBox("Output Settings")
        output_layout = QFormLayout(output_group)
        
        # Output directory
        output_dir_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit(str(self.output_dir))
        output_dir_layout.addWidget(self.output_dir_edit)
        
        browse_output_btn = QPushButton("Browse...")
        browse_output_btn.clicked.connect(self.browse_output_directory)
        output_dir_layout.addWidget(browse_output_btn)
        
        output_layout.addRow("Output Directory:", output_dir_layout)
        
        # RIFE settings display
        rife_info = self._get_rife_info_text()
        rife_label = QLabel(rife_info)
        rife_label.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        rife_label.setWordWrap(True)
        output_layout.addRow("RIFE Settings:", rife_label)
        
        layout.addWidget(output_group)
        
        # Progress section
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready to export")
        self.status_label.setStyleSheet("color: #aaaaaa;")
        progress_layout.addWidget(self.status_label)
        
        # Log area
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(100)
        self.log_text.setVisible(False)
        font = QFont("Consolas", 9)
        self.log_text.setFont(font)
        progress_layout.addWidget(self.log_text)
        
        layout.addWidget(progress_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.export_btn = QPushButton("Start Export")
        self.export_btn.setDefault(True)
        self.export_btn.clicked.connect(self.start_export)
        button_layout.addWidget(self.export_btn)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel_or_close)
        button_layout.addWidget(self.cancel_btn)
        
        button_layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
    
    def _get_rife_info_text(self) -> str:
        """Get RIFE configuration info text."""
        if not self.rife_config.get("enabled", False):
            return "RIFE disabled - files will be copied without processing"
        
        preset = self.rife_config.get("preset", "Unknown")
        target_fps = self.rife_config.get("target_fps", 60)
        scene_detection = "enabled" if self.rife_config.get("scene_detection", False) else "disabled"
        gpu_id = self.rife_config.get("gpu_id", 0)
        
        return f"Preset: {preset} ({target_fps} fps), Scene Detection: {scene_detection}, GPU: {gpu_id}"
    
    def add_files(self) -> None:
        """Add video files to the export list."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Video Files",
            str(Path.home()),
            "Video Files (*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.m4v *.webm *.ts *.mts *.m2ts);;All Files (*)"
        )
        
        for file_path in file_paths:
            self.add_file(Path(file_path))
    
    def add_folder(self) -> None:
        """Add all video files from a folder."""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Folder with Videos", str(Path.home())
        )
        
        if folder_path:
            folder = Path(folder_path)
            video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.m4v', '.webm', '.ts', '.mts', '.m2ts'}
            
            video_files = []
            for ext in video_extensions:
                video_files.extend(folder.glob(f"*{ext}"))
                video_files.extend(folder.glob(f"*{ext.upper()}"))
            
            for video_file in sorted(video_files):
                self.add_file(video_file)
    
    def add_file(self, file_path: Path) -> None:
        """Add a single file to the list."""
        # Check if file already in list
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == str(file_path):
                return  # Already in list
        
        # Create list item
        file_size = format_filesize(file_path.stat().st_size if file_path.exists() else None)
        item_text = f"{file_path.name}\n{file_size} - {file_path.parent}"
        
        item = QListWidgetItem(item_text)
        item.setData(Qt.ItemDataRole.UserRole, str(file_path))
        item.setToolTip(str(file_path))
        
        self.file_list.addItem(item)
        self.input_files.append(file_path)
        
        self._update_export_button()
    
    def remove_selected(self) -> None:
        """Remove selected files from the list."""
        for item in self.file_list.selectedItems():
            file_path = Path(item.data(Qt.ItemDataRole.UserRole))
            if file_path in self.input_files:
                self.input_files.remove(file_path)
            
            row = self.file_list.row(item)
            self.file_list.takeItem(row)
        
        self._update_export_button()
    
    def clear_files(self) -> None:
        """Clear all files from the list."""
        self.file_list.clear()
        self.input_files.clear()
        self._update_export_button()
    
    def browse_output_directory(self) -> None:
        """Browse for output directory."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", self.output_dir_edit.text()
        )
        
        if directory:
            self.output_dir_edit.setText(directory)
            self.output_dir = Path(directory)
    
    def _update_export_button(self) -> None:
        """Update export button state."""
        has_files = len(self.input_files) > 0
        self.export_btn.setEnabled(has_files)
        
        if has_files:
            self.status_label.setText(f"Ready to export {len(self.input_files)} file(s)")
        else:
            self.status_label.setText("Add files to export")
    
    def start_export(self) -> None:
        """Start the batch export process."""
        if not self.input_files:
            QMessageBox.warning(self, "No Files", "Please add some video files to export.")
            return
        
        # Validate output directory
        output_dir = Path(self.output_dir_edit.text())
        if not output_dir.parent.exists():
            QMessageBox.warning(self, "Invalid Output", 
                               "Output directory parent does not exist.")
            return
        
        # Create output directory if it doesn't exist
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot create output directory:\n{str(e)}")
            return
        
        # Setup UI for export
        self.export_btn.setEnabled(False)
        self.cancel_btn.setText("Cancel Export")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.log_text.setVisible(True)
        self.log_text.clear()
        
        # Create export settings
        export_settings = {
            "threads": 4,  # Could be configurable
            "quality": "medium"
        }
        
        # Create worker and thread
        self.export_worker = ExportWorker(
            self.input_files.copy(),
            output_dir,
            self.rife_config,
            export_settings
        )
        
        self.export_thread = QThread()
        self.export_worker.moveToThread(self.export_thread)
        
        # Connect signals
        self.export_thread.started.connect(self.export_worker.start_export)
        self.export_worker.progress_updated.connect(self.progress_bar.setValue)
        self.export_worker.status_updated.connect(self.status_label.setText)
        self.export_worker.error_occurred.connect(self.on_export_error)
        self.export_worker.export_finished.connect(self.on_export_finished)
        
        # Start export
        self.export_thread.start()
        
        self.log("Export started...")
    
    def cancel_or_close(self) -> None:
        """Cancel export or close dialog."""
        if self.export_thread and self.export_thread.isRunning():
            # Cancel export
            if self.export_worker:
                self.export_worker.cancel_export()
            self.log("Cancelling export...")
        else:
            # Close dialog
            self.reject()
    
    def on_export_error(self, error_message: str) -> None:
        """Handle export error."""
        self.log(f"Error: {error_message}")
        QMessageBox.warning(self, "Export Error", error_message)
    
    def on_export_finished(self, success: bool) -> None:
        """Handle export completion."""
        # Clean up thread
        if self.export_thread:
            self.export_thread.quit()
            self.export_thread.wait()
            self.export_thread = None
            self.export_worker = None
        
        # Reset UI
        self.export_btn.setEnabled(True)
        self.cancel_btn.setText("Cancel")
        
        if success:
            self.log("Export completed successfully!")
            self.progress_bar.setValue(100)
            QMessageBox.information(self, "Export Complete", 
                                   f"Export completed successfully!\nFiles saved to:\n{self.output_dir_edit.text()}")
        else:
            self.log("Export failed or was cancelled.")
    
    def log(self, message: str) -> None:
        """Add message to log."""
        self.log_text.append(message)
        # Auto-scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def closeEvent(self, event) -> None:
        """Handle dialog close event."""
        if self.export_thread and self.export_thread.isRunning():
            reply = QMessageBox.question(
                self, "Export in Progress",
                "Export is still running. Cancel it and close?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                if self.export_worker:
                    self.export_worker.cancel_export()
                self.export_thread.quit()
                self.export_thread.wait(5000)  # Wait up to 5 seconds
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()