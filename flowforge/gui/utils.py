"""
RIFE Player GUI Utilities
Path helpers, video probing, and tool location functions.
"""

import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Any


def _find_tool(name: str) -> str:
    """
    Find tool executable with fallback paths for Windows deployment.
    
    Critical for Windows where ffprobe/ffmpeg are not in PATH.
    Always checks the bin/ directory first.
    """
    # Determine application directory
    if getattr(sys, 'frozen', False):
        # PyInstaller bundle
        app_dir = Path(sys.executable).parent
    else:
        # Development mode - go up from gui/utils.py to project root
        app_dir = Path(__file__).resolve().parent.parent.parent
    
    # Search paths in order
    search_paths = [
        app_dir / "bin" / f"{name}.exe",  # Local bin directory
        Path(rf"C:\Users\Kad\Desktop\FlowForge\bin\{name}.exe"),  # Known deployment path
    ]
    
    for tool_path in search_paths:
        try:
            if tool_path.exists():
                return str(tool_path)
        except (PermissionError, OSError):
            continue
    
    # Fallback to system PATH
    return name


def _find_mpv() -> str:
    """Find mpv executable with SVP4 fallback."""
    # Check known SVP4 installation path first
    svp_mpv = Path(r"C:\Program Files\SVP 4\mpv64\mpv.exe")
    if platform.system() == "Windows" and svp_mpv.exists():
        return str(svp_mpv)
    
    # Fallback to system PATH
    return "mpv"


def _find_nvidia_smi() -> str:
    """Find nvidia-smi executable."""
    if platform.system() == "Windows":
        nvidia_smi = Path(r"C:\Windows\System32\nvidia-smi.exe")
        if nvidia_smi.exists():
            return str(nvidia_smi)
    
    return "nvidia-smi"


def probe_video(video_path: Path) -> Optional[Dict[str, Any]]:
    """
    Probe video file for metadata using ffprobe.
    
    Returns:
        Dictionary with video info or None if probe failed.
        Contains: duration, width, height, fps, codec, bitrate, etc.
    """
    try:
        ffprobe = _find_tool("ffprobe")
        
        cmd = [
            ffprobe,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(video_path)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=30
        )
        
        data = json.loads(result.stdout)
        
        # Extract video stream info
        video_stream = None
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                video_stream = stream
                break
        
        if not video_stream:
            return None
        
        # Extract useful metadata
        format_info = data.get("format", {})
        
        # Calculate duration
        duration = None
        duration_str = format_info.get("duration") or video_stream.get("duration")
        if duration_str:
            try:
                duration = float(duration_str)
            except (ValueError, TypeError):
                pass
        
        # Calculate FPS
        fps = None
        fps_str = video_stream.get("r_frame_rate", "0/0")
        if fps_str and "/" in fps_str:
            try:
                num, den = fps_str.split("/")
                if int(den) > 0:
                    fps = round(int(num) / int(den), 2)
            except (ValueError, ZeroDivisionError):
                pass
        
        # Calculate bitrate
        bitrate = None
        bitrate_str = format_info.get("bit_rate") or video_stream.get("bit_rate")
        if bitrate_str:
            try:
                bitrate = int(bitrate_str)
            except (ValueError, TypeError):
                pass
        
        return {
            "filename": video_path.name,
            "filepath": str(video_path),
            "duration": duration,
            "width": video_stream.get("width"),
            "height": video_stream.get("height"),
            "fps": fps,
            "codec": video_stream.get("codec_name"),
            "bitrate": bitrate,
            "size": video_path.stat().st_size if video_path.exists() else None,
            "pixel_format": video_stream.get("pix_fmt"),
        }
        
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, 
            json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error probing video {video_path}: {e}")
        return None


def format_duration(seconds: Optional[float]) -> str:
    """Format duration in seconds to HH:MM:SS string."""
    if seconds is None:
        return "Unknown"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


def format_filesize(size_bytes: Optional[int]) -> str:
    """Format file size in bytes to human readable string."""
    if size_bytes is None:
        return "Unknown"
    
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    
    return f"{size_bytes:.1f} PB"


def format_bitrate(bitrate: Optional[int]) -> str:
    """Format bitrate in bits/sec to human readable string."""
    if bitrate is None:
        return "Unknown"
    
    # Convert to kbps
    kbps = bitrate / 1000
    
    if kbps < 1000:
        return f"{kbps:.0f} kbps"
    else:
        return f"{kbps/1000:.1f} Mbps"


def get_video_resolution_name(width: Optional[int], height: Optional[int]) -> str:
    """Get common resolution name for width/height."""
    if not width or not height:
        return "Unknown"
    
    # Common resolution names
    resolutions = {
        (1920, 1080): "1080p",
        (1280, 720): "720p",
        (3840, 2160): "4K",
        (2560, 1440): "1440p",
        (1366, 768): "768p",
        (1024, 768): "768p (4:3)",
        (640, 480): "480p",
        (854, 480): "480p",
    }
    
    resolution = resolutions.get((width, height))
    if resolution:
        return resolution
    
    # Fallback to dimensions
    return f"{width}×{height}"


def is_windows() -> bool:
    """Check if running on Windows."""
    return platform.system() == "Windows"


def is_wsl_path(path: Path) -> bool:
    """Check if path is a WSL mount path that would crash on Windows."""
    return str(path).startswith("/mnt/") and is_windows()


def get_gpu_info() -> Optional[str]:
    """Get GPU information using nvidia-smi."""
    try:
        nvidia_smi = _find_nvidia_smi()
        
        cmd = [
            nvidia_smi,
            "--query-gpu=name",
            "--format=csv,noheader,nounits"
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=10
        )
        
        gpu_name = result.stdout.strip()
        if gpu_name:
            return gpu_name
        
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, 
            FileNotFoundError):
        pass
    
    return None


def get_available_gpus() -> list[int]:
    """Get list of available GPU IDs."""
    try:
        nvidia_smi = _find_nvidia_smi()
        
        cmd = [
            nvidia_smi,
            "--query-gpu=index",
            "--format=csv,noheader,nounits"
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=10
        )
        
        gpu_ids = []
        for line in result.stdout.strip().split('\n'):
            try:
                gpu_ids.append(int(line.strip()))
            except (ValueError, AttributeError):
                continue
        
        return gpu_ids
        
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, 
            FileNotFoundError):
        return [0]  # Fallback to GPU 0