"""
RIFE Player GUI Application Setup
VLC-style dark theme and application initialization.
"""

import sys
from typing import Optional
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QPalette, QColor


class FlowForgeApplication(QApplication):
    """Custom QApplication with VLC-style dark theme."""
    
    def __init__(self, argv: list[str]):
        super().__init__(argv)
        
        # Set application properties
        self.setApplicationName("RIFE Player")
        self.setApplicationVersion("1.0.0")
        self.setOrganizationName("FlowForge")
        self.setOrganizationDomain("flowforge.ai")
        
        # Apply VLC-style dark theme
        self._setup_vlc_theme()
        
        # Set modern font
        font = QFont("Segoe UI", 9)
        font.setHintingPreference(QFont.HintingPreference.PreferDefaultHinting)
        self.setFont(font)
    
    def _setup_vlc_theme(self) -> None:
        """Apply VLC-inspired dark theme."""
        self.setStyle("Fusion")
        
        # VLC-style color palette
        palette = QPalette()
        
        # Main colors - VLC inspired
        background = QColor(30, 30, 30)          # #1e1e1e - main background
        panel = QColor(45, 45, 45)               # #2d2d2d - panels
        accent = QColor(0, 120, 212)             # #0078d4 - VLC blue accent
        text = QColor(255, 255, 255)             # White text
        disabled_text = QColor(128, 128, 128)    # Disabled text
        highlight = QColor(0, 120, 212)          # Selection highlight
        button = QColor(45, 45, 45)              # Button background
        border = QColor(60, 60, 60)              # Borders
        
        # Window colors
        palette.setColor(QPalette.ColorRole.Window, background)
        palette.setColor(QPalette.ColorRole.WindowText, text)
        
        # Base colors (for input fields, lists, etc.)
        palette.setColor(QPalette.ColorRole.Base, background)
        palette.setColor(QPalette.ColorRole.AlternateBase, panel)
        palette.setColor(QPalette.ColorRole.Text, text)
        
        # Button colors
        palette.setColor(QPalette.ColorRole.Button, button)
        palette.setColor(QPalette.ColorRole.ButtonText, text)
        
        # Selection colors
        palette.setColor(QPalette.ColorRole.Highlight, highlight)
        palette.setColor(QPalette.ColorRole.HighlightedText, text)
        
        # Disabled colors
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, disabled_text)
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, disabled_text)
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, disabled_text)
        
        self.setPalette(palette)
        
        # Additional stylesheet for modern look
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            
            QMenuBar {
                background-color: #2d2d2d;
                border-bottom: 1px solid #3c3c3c;
                padding: 2px;
            }
            
            QMenuBar::item {
                background: transparent;
                padding: 4px 8px;
                border-radius: 3px;
            }
            
            QMenuBar::item:selected {
                background-color: #0078d4;
            }
            
            QMenu {
                background-color: #2d2d2d;
                border: 1px solid #3c3c3c;
                padding: 4px;
            }
            
            QMenu::item {
                padding: 6px 20px 6px 8px;
                border-radius: 3px;
            }
            
            QMenu::item:selected {
                background-color: #0078d4;
            }
            
            QMenu::separator {
                height: 1px;
                background: #3c3c3c;
                margin: 4px 0px;
            }
            
            QToolBar {
                background-color: #2d2d2d;
                border: none;
                spacing: 4px;
                padding: 4px;
            }
            
            QToolButton {
                background-color: transparent;
                border: 1px solid transparent;
                border-radius: 4px;
                padding: 6px;
                min-width: 60px;
            }
            
            QToolButton:hover {
                background-color: #404040;
                border-color: #5c5c5c;
            }
            
            QToolButton:pressed {
                background-color: #0078d4;
            }
            
            QToolButton:checked {
                background-color: #0078d4;
                border-color: #106ebe;
            }
            
            QPushButton {
                background-color: #2d2d2d;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
                padding: 6px 12px;
                min-height: 20px;
            }
            
            QPushButton:hover {
                background-color: #404040;
                border-color: #5c5c5c;
            }
            
            QPushButton:pressed {
                background-color: #0078d4;
            }
            
            QPushButton:disabled {
                background-color: #1a1a1a;
                border-color: #2c2c2c;
                color: #808080;
            }
            
            QComboBox {
                background-color: #2d2d2d;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
                padding: 4px 8px;
                min-height: 20px;
            }
            
            QComboBox:hover {
                border-color: #5c5c5c;
            }
            
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 4px solid #ffffff;
                margin-right: 6px;
            }
            
            QComboBox QAbstractItemView {
                background-color: #2d2d2d;
                border: 1px solid #3c3c3c;
                selection-background-color: #0078d4;
            }
            
            QSpinBox, QDoubleSpinBox {
                background-color: #2d2d2d;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
                padding: 4px;
                min-height: 20px;
            }
            
            QSpinBox:hover, QDoubleSpinBox:hover {
                border-color: #5c5c5c;
            }
            
            QCheckBox {
                spacing: 8px;
            }
            
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #3c3c3c;
                border-radius: 3px;
                background-color: #2d2d2d;
            }
            
            QCheckBox::indicator:hover {
                border-color: #5c5c5c;
            }
            
            QCheckBox::indicator:checked {
                background-color: #0078d4;
                border-color: #106ebe;
                image: none;
            }
            
            QLabel {
                color: #ffffff;
                background: transparent;
            }
            
            QScrollBar:vertical {
                background-color: #1e1e1e;
                width: 12px;
                border: none;
            }
            
            QScrollBar::handle:vertical {
                background-color: #404040;
                border-radius: 6px;
                min-height: 20px;
                margin: 2px;
            }
            
            QScrollBar::handle:vertical:hover {
                background-color: #5c5c5c;
            }
            
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
            }
        """)


def create_application(argv: list[str]) -> FlowForgeApplication:
    """Create and configure the RIFE Player application."""
    return FlowForgeApplication(argv)