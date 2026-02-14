"""VapourSynth filter for real-time RIFE interpolation."""

import logging
import os
import queue
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .presets import InterpolationPreset

logger = logging.getLogger(__name__)


class VapourSynthFilter:
    """VapourSynth filter manager for real-time RIFE interpolation."""
    
    def __init__(self, preset: InterpolationPreset, rife_binary_path: Optional[Path] = None):
        """Initialize VapourSynth filter.
        
        Args:
            preset: Interpolation preset configuration
            rife_binary_path: Path to rife-ncnn-vulkan binary
        """
        self.preset = preset
        self.rife_binary_path = rife_binary_path or self._find_rife_binary()
        
        # Performance monitoring
        self.stats = {
            "frames_processed": 0,
            "frames_interpolated": 0,
            "frames_dropped": 0,
            "processing_fps": 0.0,
            "avg_latency_ms": 0.0,
            "scene_changes_detected": 0,
            "gpu_utilization": 0.0
        }
        
        # Threading and queuing
        self.frame_queue = queue.Queue(maxsize=preset.max_queue_size)
        self.output_queue = queue.Queue(maxsize=preset.max_queue_size * 2)
        self.processing_thread = None
        self.is_processing = False
        
        # Scene detection
        self.last_frame = None
        self.scene_detector = SceneDetector() if preset.scene_detection else None
        
        # Performance monitoring
        self.frame_times = queue.Queue(maxsize=60)  # Last 60 frame times
        self.last_stats_time = time.time()
        
    def _find_rife_binary(self) -> Optional[Path]:
        """Find RIFE binary in common locations."""
        import platform
        
        is_windows = platform.system() == "Windows"
        is_wsl = "microsoft" in platform.uname().release.lower()
        
        common_paths = [
            Path("/mnt/c/Users/Kad/Desktop/FlowForge/bin/rife-ncnn-vulkan.exe"),  # WSL path
            Path.home() / ".flowforge" / "bin" / "rife-ncnn-vulkan",
            Path.home() / ".flowforge" / "bin" / "rife-ncnn-vulkan.exe"
        ]
        
        if is_windows or is_wsl:
            common_paths.extend([
                Path("C:/Program Files/FlowForge/bin/rife-ncnn-vulkan.exe"),
                Path("C:/FlowForge/bin/rife-ncnn-vulkan.exe")
            ])
        
        for path in common_paths:
            if path.exists():
                return path
        
        return None
    
    def generate_script(self) -> str:
        """Generate VapourSynth script content.
        
        Returns:
            VapourSynth script as string
        """
        try:
            # Try to import vs-rife plugin
            import vapoursynth as vs
            core = vs.core
            has_rife_plugin = hasattr(core, 'rife')
        except ImportError:
            has_rife_plugin = False
        
        if has_rife_plugin:
            return self._generate_plugin_script()
        else:
            return self._generate_binary_script()
    
    def _generate_plugin_script(self) -> str:
        """Generate script using vs-rife-ncnn-vulkan plugin."""
        script = f'''# FlowForge VapourSynth Script (Plugin Mode)
# Preset: {self.preset.name}
# Target FPS: {self.preset.target_fps}

import vapoursynth as vs
import math
import logging

core = vs.core
core.max_cache_size = 1024  # Limit cache for real-time

# Configuration
TARGET_FPS = {self.preset.target_fps}
MODEL_NAME = "{self.preset.model}"
TTA = {self.preset.tta}
UHD = {self.preset.uhd}
TILE_SIZE = {self.preset.tile_size}
TILE_PAD = {self.preset.tile_pad}
SCENE_DETECTION = {self.preset.scene_detection}
SCENE_THRESHOLD = {self.preset.scene_threshold}
GPU_ID = 0

def flowforge_interpolate(clip):
    """Apply FlowForge RIFE interpolation."""
    
    # Get input properties
    input_fps = clip.fps_num / clip.fps_den if clip.fps_den != 0 else 24.0
    multiplier = TARGET_FPS / input_fps
    
    # Skip interpolation if not needed
    if multiplier <= 1.05:  # Small tolerance
        return core.std.AssumeFPS(clip, fpsnum=int(TARGET_FPS), fpsden=1)
    
    # Apply scene detection if enabled
    if SCENE_DETECTION:
        try:
            # Scene change detection
            clip_sc = core.misc.SCDetect(clip, threshold=SCENE_THRESHOLD)
            
            # Apply RIFE with scene detection
            interpolated = core.rife.RIFE(
                clip,
                model=MODEL_NAME,
                factor_num=int(multiplier * 1000),
                factor_den=1000,
                gpu_id=GPU_ID,
                tta=TTA,
                uhd=UHD,
                sc=clip_sc,
                tile_w=TILE_SIZE,
                tile_h=TILE_SIZE,
                tile_pad_w=TILE_PAD,
                tile_pad_h=TILE_PAD
            )
        except Exception as e:
            # Fallback without scene detection
            interpolated = core.rife.RIFE(
                clip,
                model=MODEL_NAME,
                factor_num=int(multiplier * 1000),
                factor_den=1000,
                gpu_id=GPU_ID,
                tta=TTA,
                uhd=UHD,
                tile_w=TILE_SIZE,
                tile_h=TILE_SIZE,
                tile_pad_w=TILE_PAD,
                tile_pad_h=TILE_PAD
            )
    else:
        # Direct interpolation
        interpolated = core.rife.RIFE(
            clip,
            model=MODEL_NAME,
            factor_num=int(multiplier * 1000),
            factor_den=1000,
            gpu_id=GPU_ID,
            tta=TTA,
            uhd=UHD,
            tile_w=TILE_SIZE,
            tile_h=TILE_SIZE,
            tile_pad_w=TILE_PAD,
            tile_pad_h=TILE_PAD
        )
    
    # Set output frame rate
    output = core.std.AssumeFPS(interpolated, fpsnum=int(TARGET_FPS), fpsden=1)
    
    return output

# Main filter function
def main(video_in):
    """Main entry point for mpv."""
    return flowforge_interpolate(video_in)

# mpv integration
video_out = main(video_in)
'''
        return script
    
    def _generate_binary_script(self) -> str:
        """Generate script using subprocess calls to RIFE binary."""
        script = f'''# FlowForge VapourSynth Script (Binary Mode)
# Preset: {self.preset.name}
# Target FPS: {self.preset.target_fps}

import vapoursynth as vs
import subprocess
import tempfile
import os
import sys
import threading
import queue
import time
import numpy as np
from pathlib import Path

core = vs.core
core.max_cache_size = 512  # Smaller cache for binary mode

# Configuration
TARGET_FPS = {self.preset.target_fps}
MODEL_NAME = "{self.preset.model}"
TTA = {self.preset.tta}
UHD = {self.preset.uhd}
RIFE_BINARY = r"{self.rife_binary_path}"
BUFFER_SIZE = {self.preset.buffer_frames}
SCENE_DETECTION = {self.preset.scene_detection}
SCENE_THRESHOLD = {self.preset.scene_threshold}

class RIFEBinaryProcessor:
    """Processes frames using RIFE binary."""
    
    def __init__(self):
        self.frame_queue = queue.Queue(maxsize=BUFFER_SIZE * 2)
        self.result_queue = queue.Queue(maxsize=BUFFER_SIZE * 4)
        self.processing_thread = None
        self.is_running = False
        self.temp_dir = None
        self.frame_count = 0
        
    def start(self, input_fps):
        """Start processing thread."""
        self.input_fps = input_fps
        self.multiplier = TARGET_FPS / input_fps
        self.temp_dir = tempfile.mkdtemp(prefix="flowforge_")
        self.is_running = True
        
        self.processing_thread = threading.Thread(
            target=self._process_loop,
            daemon=True
        )
        self.processing_thread.start()
        
    def stop(self):
        """Stop processing and cleanup."""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
        if self.temp_dir:
            import shutil
            try:
                shutil.rmtree(self.temp_dir)
            except:
                pass
                
    def add_frame(self, frame_data, frame_num):
        """Add frame for processing."""
        try:
            self.frame_queue.put((frame_data, frame_num), timeout=0.1)
        except queue.Full:
            # Drop frame if queue full
            pass
            
    def get_result(self, timeout=0.1):
        """Get processed frame result."""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def _process_loop(self):
        """Main processing loop."""
        while self.is_running:
            try:
                # Get frame pair for interpolation
                frame_data, frame_num = self.frame_queue.get(timeout=0.5)
                
                # Simple passthrough for now (real implementation would call RIFE)
                # This is a simplified version - full implementation would:
                # 1. Save frames as images
                # 2. Call RIFE binary
                # 3. Load interpolated frames
                # 4. Queue results
                
                # For now, just return original frame
                self.result_queue.put((frame_data, frame_num))
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {{e}}", file=sys.stderr)
                continue

# Global processor instance
processor = RIFEBinaryProcessor()

def flowforge_interpolate(clip):
    """Apply FlowForge interpolation using binary."""
    
    input_fps = clip.fps_num / clip.fps_den if clip.fps_den != 0 else 24.0
    multiplier = TARGET_FPS / input_fps
    
    if multiplier <= 1.05:
        return core.std.AssumeFPS(clip, fpsnum=int(TARGET_FPS), fpsden=1)
    
    # Start processor
    processor.start(input_fps)
    
    def get_frame(n, f):
        """Frame getter with processing."""
        try:
            # Convert VapourSynth frame to numpy
            frame_array = np.array(f.get_read_array(0), copy=False)
            
            # Add to processing queue
            processor.add_frame(frame_array, n)
            
            # Try to get processed result
            result = processor.get_result()
            if result:
                # Convert back to VapourSynth frame (simplified)
                return f
            else:
                # Return original frame if no result ready
                return f
                
        except Exception as e:
            print(f"Frame processing error: {{e}}", file=sys.stderr)
            return f
    
    # Create output clip
    output = core.std.ModifyFrame(clip, clip, get_frame)
    output = core.std.AssumeFPS(output, fpsnum=int(TARGET_FPS), fpsden=1)
    
    return output

# Main filter function  
def main(video_in):
    """Main entry point for mpv."""
    try:
        return flowforge_interpolate(video_in)
    except Exception as e:
        print(f"FlowForge error: {{e}}", file=sys.stderr)
        return video_in  # Fallback to original

# mpv integration
video_out = main(video_in)
'''
        return script
    
    def create_filter_file(self, output_path: Path) -> Path:
        """Create VapourSynth filter file.
        
        Args:
            output_path: Path where to save the .vpy file
            
        Returns:
            Path to created filter file
        """
        script_content = self.generate_script()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        logger.info(f"Created VapourSynth filter: {output_path}")
        return output_path
    
    def update_stats(self, frame_time: float) -> None:
        """Update performance statistics.
        
        Args:
            frame_time: Time taken to process frame in seconds
        """
        current_time = time.time()
        
        # Add frame time to queue
        try:
            self.frame_times.put_nowait(frame_time)
        except queue.Full:
            # Remove oldest and add new
            try:
                self.frame_times.get_nowait()
                self.frame_times.put_nowait(frame_time)
            except queue.Empty:
                pass
        
        # Update stats every second
        if current_time - self.last_stats_time >= 1.0:
            self._calculate_stats()
            self.last_stats_time = current_time
    
    def _calculate_stats(self) -> None:
        """Calculate performance statistics."""
        if self.frame_times.empty():
            return
        
        # Get all frame times
        frame_times = []
        while not self.frame_times.empty():
            try:
                frame_times.append(self.frame_times.get_nowait())
            except queue.Empty:
                break
        
        if not frame_times:
            return
        
        # Calculate statistics
        avg_frame_time = sum(frame_times) / len(frame_times)
        self.stats["processing_fps"] = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
        self.stats["avg_latency_ms"] = avg_frame_time * 1000
        
        # Re-add frame times to queue
        for ft in frame_times[-30:]:  # Keep last 30
            try:
                self.frame_times.put_nowait(ft)
            except queue.Full:
                break
    
    def get_stats(self) -> Dict[str, float]:
        """Get current performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        return self.stats.copy()


class SceneDetector:
    """Simple scene change detector."""
    
    def __init__(self, threshold: float = 0.3, method: str = "ssim"):
        """Initialize scene detector.
        
        Args:
            threshold: Scene change threshold (0.0-1.0)
            method: Detection method (ssim, histogram, mse)
        """
        self.threshold = threshold
        self.method = method
        self.last_frame_hash = None
        
    def detect_scene_change(self, frame1: np.ndarray, frame2: np.ndarray) -> bool:
        """Detect if scene change occurred between frames.
        
        Args:
            frame1: First frame as numpy array
            frame2: Second frame as numpy array
            
        Returns:
            True if scene change detected
        """
        if self.method == "histogram":
            return self._histogram_difference(frame1, frame2) > self.threshold
        elif self.method == "mse":
            return self._mse_difference(frame1, frame2) > self.threshold
        else:  # ssim
            return self._ssim_difference(frame1, frame2) < (1.0 - self.threshold)
    
    def _histogram_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate histogram difference between frames."""
        try:
            # Convert to grayscale if needed
            if len(frame1.shape) == 3:
                gray1 = np.mean(frame1, axis=2)
                gray2 = np.mean(frame2, axis=2)
            else:
                gray1, gray2 = frame1, frame2
            
            # Calculate histograms
            hist1, _ = np.histogram(gray1.flatten(), bins=256, range=(0, 255))
            hist2, _ = np.histogram(gray2.flatten(), bins=256, range=(0, 255))
            
            # Normalize histograms
            hist1 = hist1.astype(float) / hist1.sum()
            hist2 = hist2.astype(float) / hist2.sum()
            
            # Calculate correlation
            correlation = np.corrcoef(hist1, hist2)[0, 1]
            return 1.0 - correlation if not np.isnan(correlation) else 1.0
            
        except Exception:
            return 0.0
    
    def _mse_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate MSE difference between frames."""
        try:
            diff = frame1.astype(float) - frame2.astype(float)
            mse = np.mean(diff ** 2)
            # Normalize to 0-1 range (assuming 8-bit images)
            return mse / (255.0 ** 2)
        except Exception:
            return 0.0
    
    def _ssim_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate SSIM between frames (simplified version)."""
        try:
            # Simplified SSIM calculation
            # Convert to grayscale if needed
            if len(frame1.shape) == 3:
                gray1 = np.mean(frame1, axis=2)
                gray2 = np.mean(frame2, axis=2)
            else:
                gray1, gray2 = frame1, frame2
            
            # Calculate means
            mu1 = np.mean(gray1)
            mu2 = np.mean(gray2)
            
            # Calculate variances and covariance
            sigma1_sq = np.var(gray1)
            sigma2_sq = np.var(gray2)
            sigma12 = np.mean((gray1 - mu1) * (gray2 - mu2))
            
            # SSIM constants
            c1 = (0.01 * 255) ** 2
            c2 = (0.03 * 255) ** 2
            
            # Calculate SSIM
            ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \\
                   ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2))
            
            return max(0.0, min(1.0, ssim))
            
        except Exception:
            return 0.0


def create_vapoursynth_filter(
    preset: InterpolationPreset,
    output_path: Path,
    rife_binary_path: Optional[Path] = None
) -> VapourSynthFilter:
    """Create and save VapourSynth filter.
    
    Args:
        preset: Interpolation preset
        output_path: Path to save .vpy file
        rife_binary_path: Path to RIFE binary
        
    Returns:
        VapourSynthFilter instance
    """
    filter_instance = VapourSynthFilter(preset, rife_binary_path)
    filter_instance.create_filter_file(output_path)
    return filter_instance