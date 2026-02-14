#!/usr/bin/env python3
"""
FlowForge GUI Launcher
Simple launcher script for the FlowForge GUI application.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from flowforge.gui.main import main

if __name__ == "__main__":
    sys.exit(main())