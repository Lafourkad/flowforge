"""
FlowForge GUI Main Entry Point
Runnable with: python -m flowforge.gui
"""

import sys
from typing import Optional

from .app import create_application
from .main_window import FlowForgeMainWindow


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point for FlowForge GUI."""
    if argv is None:
        argv = sys.argv
    
    # Create application
    app = create_application(argv)
    
    # Create and show main window
    window = FlowForgeMainWindow()
    window.show()
    
    # Run event loop
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())