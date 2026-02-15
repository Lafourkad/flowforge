"""
RIFE Player GUI Main Entry Point
VLC-style video player with RIFE interpolation.
"""

import sys
from typing import Optional

from .app import create_application
from .player import FlowForgePlayer


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point for RIFE Player GUI."""
    if argv is None:
        argv = sys.argv
    
    # Create application with dark theme
    app = create_application(argv)
    
    # Create and show main player window
    player = FlowForgePlayer()
    player.show()
    
    # Run event loop
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())