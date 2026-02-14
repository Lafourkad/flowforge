"""
FlowForge Settings Management
Handles configuration persistence and platform detection.
"""

import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


class FlowForgeSettings:
    """Manages FlowForge configuration and settings."""
    
    def __init__(self):
        self.config_dir = Path.home() / ".flowforge"
        self.config_file = self.config_dir / "config.json"
        self._settings: Dict[str, Any] = {}
        self._load_settings()
        self._detect_platform()
        self._auto_detect_paths()
    
    def _load_settings(self) -> None:
        """Load settings from config file."""
        self._settings = self._get_default_settings()
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_settings = json.load(f)
                    self._settings.update(loaded_settings)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load settings: {e}")
    
    def save_settings(self) -> None:
        """Save current settings to config file."""
        self.config_dir.mkdir(exist_ok=True)
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self._settings, f, indent=2)
        except IOError as e:
            print(f"Warning: Could not save settings: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a setting value."""
        return self._settings.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a setting value."""
        self._settings[key] = value
    
    def _get_default_settings(self) -> Dict[str, Any]:
        """Get default settings."""
        return {
            # Paths
            "rife_binary": "",
            "rife_model_dir": "",
            "mpv_path": "",
            "vs_plugin_path": "",
            
            # Processing
            "default_gpu": 0,
            "default_threads": "1:2:2",
            "default_preset": "Film (24→60fps)",
            
            # Export
            "default_encoding_preset": "Balanced",
            "default_nvenc": True,
            "default_crf": 18,
            
            # UI
            "last_input_dir": str(Path.home()),
            "last_output_dir": str(Path.home()),
            "window_geometry": None,
            "window_state": None,
            
            # Scene detection
            "scene_detection": True,
            "scene_threshold": 0.3,
            
            # Platform
            "platform": platform.system(),
            "is_wsl": False,
        }
    
    def _detect_platform(self) -> None:
        """Detect if running on WSL."""
        is_wsl = False
        if platform.system() == "Linux":
            try:
                with open('/proc/version', 'r') as f:
                    version_info = f.read()
                    if 'microsoft' in version_info.lower() or 'wsl' in version_info.lower():
                        is_wsl = True
            except IOError:
                pass
        
        self._settings["is_wsl"] = is_wsl
    
    def _auto_detect_paths(self) -> None:
        """Auto-detect FlowForge component paths."""
        is_windows = platform.system() == "Windows"
        is_wsl = self._settings.get("is_wsl", False)
        
        # Try to find RIFE binary
        if is_windows:
            rife_paths = [
                "C:\\Users\\Kad\\Desktop\\FlowForge\\bin\\rife-ncnn-vulkan.exe",
                ".\\bin\\rife-ncnn-vulkan.exe",
            ]
        else:
            rife_paths = [
                "/mnt/c/Users/Kad/Desktop/FlowForge/bin/rife-ncnn-vulkan.exe",
                "./bin/rife-ncnn-vulkan.exe",
            ]
        
        for path in rife_paths:
            try:
                if Path(path).exists():
                    self._settings["rife_binary"] = path
                    break
            except (PermissionError, OSError):
                continue
        
        # Try to find RIFE model
        if is_windows:
            model_paths = [
                "C:\\Users\\Kad\\Desktop\\FlowForge\\models\\rife-v4.6",
                ".\\models\\rife-v4.6",
            ]
        else:
            model_paths = [
                "/mnt/c/Users/Kad/Desktop/FlowForge/models/rife-v4.6",
                "./models/rife-v4.6",
            ]
        
        for path in model_paths:
            try:
                if Path(path).exists():
                    self._settings["rife_model_dir"] = path
                    break
            except (PermissionError, OSError):
                continue
        
        # Try to find mpv
        if is_windows:
            mpv_paths = [
                "C:\\Program Files\\SVP 4\\mpv64\\mpv.exe",
                "C:\\Program Files\\mpv\\mpv.exe",
            ]
        else:
            mpv_paths = [
                "/usr/bin/mpv",
                "/usr/local/bin/mpv",
            ]
        
        for path in mpv_paths:
            try:
                if Path(path).exists():
                    self._settings["mpv_path"] = path
                    break
            except (PermissionError, OSError):
                continue
        
        # Try to find VS plugin
        if is_windows:
            vs_plugin_paths = [
                "C:\\Users\\Kad\\Desktop\\FlowForge\\vs-plugins\\librife.dll",
                ".\\vs-plugins\\librife.dll",
            ]
        else:
            vs_plugin_paths = [
                "/mnt/c/Users/Kad/Desktop/FlowForge/vs-plugins/librife.dll",
                "./vs-plugins/librife.dll",
            ]
        
        for path in vs_plugin_paths:
            try:
                if Path(path).exists():
                    self._settings["vs_plugin_path"] = path
                    break
            except (PermissionError, OSError):
                continue
    
    def to_windows_path(self, path: str) -> str:
        """Convert WSL path to Windows path for .exe binaries."""
        if not self._settings.get("is_wsl", False):
            return path
        
        path = str(path)
        if path.startswith("/mnt/"):
            # Convert /mnt/c/... to C:\...
            drive = path[5]
            rest = path[7:].replace("/", "\\")
            return f"{drive.upper()}:\\{rest}"
        return path
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information using nvidia-smi."""
        try:
            cmd = ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpus = []
                for i, line in enumerate(lines):
                    parts = line.split(', ')
                    if len(parts) >= 2:
                        name = parts[0].strip()
                        memory = parts[1].strip()
                        gpus.append({
                            "id": i,
                            "name": name,
                            "memory": f"{memory} MB"
                        })
                return {"gpus": gpus, "detected": True}
        
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
            pass
        
        return {"gpus": [], "detected": False}
    
    def get_rife_version(self) -> Optional[str]:
        """Get RIFE binary version information."""
        rife_path = self.get("rife_binary")
        if not rife_path or not Path(rife_path).exists():
            return None
        
        try:
            win_path = self.to_windows_path(rife_path)
            result = subprocess.run([win_path], capture_output=True, text=True, timeout=5)
            
            # RIFE prints help/version info to stderr usually
            output = result.stderr or result.stdout
            
            # Look for version information in output
            for line in output.split('\n'):
                if 'rife' in line.lower() and ('v' in line.lower() or 'version' in line.lower()):
                    return line.strip()
            
            # Fallback - just return that it's detected
            return "RIFE ncnn Vulkan (detected)"
        
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
            pass
        
        return None


# Global settings instance
settings = FlowForgeSettings()