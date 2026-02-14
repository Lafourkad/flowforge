"""FlowForge Phase 2 - Real-time video playback with frame interpolation.

This package provides real-time RIFE interpolation through mpv via VapourSynth.
"""

__version__ = "2.0.0"

from .launcher import MPVLauncher
from .presets import InterpolationPreset, PresetManager
from .realtime_engine import RealtimeEngine
from .stream_processor import StreamProcessor, create_stream_processor

__all__ = [
    "MPVLauncher",
    "InterpolationPreset", 
    "PresetManager",
    "RealtimeEngine",
    "StreamProcessor",
    "create_stream_processor"
]