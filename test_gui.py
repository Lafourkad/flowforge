#!/usr/bin/env python3
"""
FlowForge GUI Test Script
Quick test to verify the GUI components load correctly.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all GUI components can be imported."""
    print("Testing FlowForge GUI imports...")
    
    try:
        # Test PyQt6 availability
        from PyQt6.QtWidgets import QApplication
        print("✅ PyQt6 available")
        
        # Test GUI components
        from flowforge.gui.app import create_application
        print("✅ App module imported")
        
        from flowforge.gui.main_window import FlowForgeMainWindow
        print("✅ Main window imported")
        
        from flowforge.gui.settings import settings
        print("✅ Settings module imported")
        
        from flowforge.gui.worker import VideoProcessorWorker, PlaybackWorker
        print("✅ Worker threads imported")
        
        from flowforge.gui.widgets.dragdrop import VideoDropZone
        from flowforge.gui.widgets.progress import ProcessingProgressWidget
        from flowforge.gui.widgets.settings import SettingsDialog
        print("✅ All widgets imported")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_settings():
    """Test settings functionality."""
    print("\nTesting settings system...")
    
    try:
        from flowforge.gui.settings import settings
        
        # Test basic get/set
        settings.set("test_key", "test_value")
        value = settings.get("test_key")
        assert value == "test_value", f"Expected 'test_value', got '{value}'"
        print("✅ Settings get/set works")
        
        # Test platform detection
        platform = settings.get("platform")
        is_wsl = settings.get("is_wsl")
        print(f"✅ Platform: {platform}, WSL: {is_wsl}")
        
        # Test GPU detection
        gpu_info = settings.get_gpu_info()
        print(f"✅ GPU detection: {len(gpu_info.get('gpus', []))} GPUs found")
        
        return True
        
    except Exception as e:
        print(f"❌ Settings test error: {e}")
        return False


def test_gui_creation():
    """Test GUI window creation (without showing)."""
    print("\nTesting GUI creation...")
    
    try:
        # Create application without showing
        from flowforge.gui.app import create_application
        app = create_application(["test"])
        print("✅ QApplication created")
        
        # Create main window
        from flowforge.gui.main_window import FlowForgeMainWindow
        window = FlowForgeMainWindow()
        print("✅ Main window created")
        
        # Test window properties
        title = window.windowTitle()
        size = window.size()
        print(f"✅ Window title: '{title}', size: {size.width()}x{size.height()}")
        
        # Clean up
        app.quit()
        print("✅ Application cleaned up")
        
        return True
        
    except Exception as e:
        print(f"❌ GUI creation error: {e}")
        return False


def main():
    """Run all tests."""
    print("FlowForge GUI Test Suite")
    print("=" * 40)
    
    # Add project root to path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    tests = [
        ("Import Test", test_imports),
        ("Settings Test", test_settings), 
        ("GUI Creation Test", test_gui_creation),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * len(test_name))
        
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print(f"\n{'=' * 40}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! FlowForge GUI is ready to use.")
        print("\nTo start the GUI:")
        print("  python -m flowforge.gui")
        print("  python run_gui.py")
        return 0
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())