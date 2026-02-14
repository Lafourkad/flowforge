"""
FlowForge GUI Application Setup
Modern dark theme and application initialization.
"""

import sys
from typing import Optional
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QPalette, QColor


class FlowForgeApplication(QApplication):
    """Custom QApplication with dark theme and app-specific settings."""
    
    def __init__(self, argv: list[str]):
        super().__init__(argv)
        
        # Set application properties
        self.setApplicationName("FlowForge")
        self.setApplicationVersion("1.0.0")
        self.setOrganizationName("FlowForge")
        self.setOrganizationDomain("flowforge.ai")
        
        # Apply dark theme
        self._setup_dark_theme()
        
        # Set font
        font = QFont("Segoe UI", 9)
        font.setHintingPreference(QFont.HintingPreference.PreferDefaultHinting)
        self.setFont(font)
    
    def _setup_dark_theme(self) -> None:
        """Apply a modern dark theme similar to DaVinci Resolve / HandBrake."""
        # Set style to Fusion for better customization
        self.setStyle("Fusion")
        
        # Define color palette
        dark_palette = QPalette()
        
        # Base colors
        base_color = QColor(37, 37, 37)          # #252525 - main background
        alt_base_color = QColor(45, 45, 45)      # #2d2d2d - alternate background
        text_color = QColor(220, 220, 220)       # #dcdcdc - main text
        disabled_text = QColor(127, 127, 127)    # #7f7f7f - disabled text
        highlight_color = QColor(42, 130, 218)   # #2a82da - selection/accent
        button_color = QColor(53, 53, 53)        # #353535 - buttons
        window_color = QColor(32, 32, 32)        # #202020 - window background
        
        # Set palette colors
        dark_palette.setColor(QPalette.ColorRole.Window, window_color)
        dark_palette.setColor(QPalette.ColorRole.WindowText, text_color)
        dark_palette.setColor(QPalette.ColorRole.Base, base_color)
        dark_palette.setColor(QPalette.ColorRole.AlternateBase, alt_base_color)
        dark_palette.setColor(QPalette.ColorRole.ToolTipBase, base_color)
        dark_palette.setColor(QPalette.ColorRole.ToolTipText, text_color)
        dark_palette.setColor(QPalette.ColorRole.Text, text_color)
        dark_palette.setColor(QPalette.ColorRole.Button, button_color)
        dark_palette.setColor(QPalette.ColorRole.ButtonText, text_color)
        dark_palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.ColorRole.Link, highlight_color)
        dark_palette.setColor(QPalette.ColorRole.Highlight, highlight_color)
        dark_palette.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))
        
        # Disabled colors
        dark_palette.setColor(QPalette.ColorGroup.Disabled, 
                            QPalette.ColorRole.WindowText, disabled_text)
        dark_palette.setColor(QPalette.ColorGroup.Disabled, 
                            QPalette.ColorRole.Text, disabled_text)
        dark_palette.setColor(QPalette.ColorGroup.Disabled, 
                            QPalette.ColorRole.ButtonText, disabled_text)
        dark_palette.setColor(QPalette.ColorGroup.Disabled, 
                            QPalette.ColorRole.Highlight, QColor(80, 80, 80))
        dark_palette.setColor(QPalette.ColorGroup.Disabled, 
                            QPalette.ColorRole.HighlightedText, disabled_text)
        
        self.setPalette(dark_palette)
        
        # Additional stylesheet for fine-tuning
        self.setStyleSheet("""
            QToolTip {
                color: #ffffff;
                background-color: #2b2b2b;
                border: 1px solid #5a5a5a;
                border-radius: 3px;
                padding: 4px;
                opacity: 200;
            }
            
            QMenuBar {
                background-color: #2b2b2b;
                color: #dcdcdc;
                border-bottom: 1px solid #3c3c3c;
            }
            
            QMenuBar::item {
                background: transparent;
                padding: 4px 8px;
            }
            
            QMenuBar::item:selected {
                background: #404040;
                border-radius: 3px;
            }
            
            QMenu {
                background-color: #2b2b2b;
                color: #dcdcdc;
                border: 1px solid #3c3c3c;
            }
            
            QMenu::item {
                padding: 6px 16px;
            }
            
            QMenu::item:selected {
                background-color: #2a82da;
            }
            
            QStatusBar {
                background-color: #2b2b2b;
                color: #dcdcdc;
                border-top: 1px solid #3c3c3c;
            }
            
            QProgressBar {
                border: 1px solid #3c3c3c;
                border-radius: 4px;
                text-align: center;
                background-color: #252525;
            }
            
            QProgressBar::chunk {
                background-color: #2a82da;
                border-radius: 3px;
                margin: 1px;
            }
            
            QSlider::groove:horizontal {
                border: 1px solid #3c3c3c;
                height: 6px;
                background: #252525;
                border-radius: 3px;
            }
            
            QSlider::handle:horizontal {
                background: #2a82da;
                border: 1px solid #1a6bb8;
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }
            
            QSlider::handle:horizontal:hover {
                background: #3a92da;
            }
            
            QSpinBox, QDoubleSpinBox {
                background-color: #353535;
                border: 1px solid #3c3c3c;
                border-radius: 3px;
                padding: 2px 4px;
                selection-background-color: #2a82da;
            }
            
            QComboBox {
                background-color: #353535;
                border: 1px solid #3c3c3c;
                border-radius: 3px;
                padding: 2px 8px;
                selection-background-color: #2a82da;
            }
            
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            
            QComboBox::down-arrow {
                image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAcAAAAECAYAAABCxiV9AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAAdgAAAHYBTnsmCAAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAABHSURBVAiZY/z//z8DLsD4//9/BiYGBgYmIF0k8x8E4YpB+ljSCJ3Y0kjSyNIgNpI0sjSSNJI0sjSyNJI0sjSSNJI0slwDAJ5lJAOyG2zpAAAAAElFTkSuQmCC);
                width: 7px;
                height: 4px;
            }
            
            QComboBox QAbstractItemView {
                background-color: #2b2b2b;
                border: 1px solid #3c3c3c;
                color: #dcdcdc;
                selection-background-color: #2a82da;
            }
        """)


def create_application(argv: Optional[list[str]] = None) -> FlowForgeApplication:
    """Create and configure the FlowForge application."""
    if argv is None:
        argv = sys.argv
    
    # High-DPI is enabled by default in Qt6
    app = FlowForgeApplication(argv)
    return app