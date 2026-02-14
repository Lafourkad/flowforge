"""Real-time interpolation engine for FlowForge."""

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import psutil

from .presets import InterpolationPreset
from .vapoursynth_filter import SceneDetector

logger = logging.getLogger(__name__)


@dataclass
class FrameStats:
    """Statistics for a processed frame."""
    frame_number: int
    input_timestamp: float
    processing_start: float
    processing_end: float
    output_timestamp: float
    interpolated: bool = False
    scene_change: bool = False
    dropped: bool = False
    
    @property
    def processing_time_ms(self) -> float:
        """Processing time in milliseconds."""
        return (self.processing_end - self.processing_start) * 1000
    
    @property
    def total_latency_ms(self) -> float:
        """Total latency from input to output in milliseconds."""
        return (self.output_timestamp - self.input_timestamp) * 1000


@dataclass
class EngineStats:
    """Real-time engine performance statistics."""
    frames_processed: int = 0
    frames_interpolated: int = 0
    frames_dropped: int = 0
    scene_changes_detected: int = 0
    
    # Performance metrics
    current_fps: float = 0.0
    target_fps: float = 60.0
    processing_fps: float = 0.0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    
    # Resource usage
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_usage_percent: float = 0.0
    gpu_memory_mb: float = 0.0
    
    # Buffer status
    input_buffer_size: int = 0
    output_buffer_size: int = 0
    buffer_overruns: int = 0
    
    # Quality adaptations
    quality_level: str = "balanced"
    adaptive_changes: int = 0
    
    # Timing
    start_time: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    
    @property
    def uptime_seconds(self) -> float:
        """Engine uptime in seconds."""
        return time.time() - self.start_time
    
    @property
    def processing_load(self) -> float:
        """Processing load ratio (0.0-1.0)."""
        return min(1.0, self.current_fps / self.target_fps) if self.target_fps > 0 else 0.0


class AdaptiveQualityManager:
    """Manages adaptive quality based on performance."""
    
    def __init__(self, preset: InterpolationPreset):
        """Initialize adaptive quality manager.
        
        Args:
            preset: Base interpolation preset
        """
        self.base_preset = preset
        self.current_preset = preset
        
        # Quality levels (from lowest to highest)
        self.quality_levels = [
            {
                "name": "emergency",
                "tile_size": 768,
                "tta": False,
                "uhd": False,
                "buffer_frames": 2,
                "drop_threshold": 0.4
            },
            {
                "name": "fast",
                "tile_size": 640,
                "tta": False,
                "uhd": False,
                "buffer_frames": 4,
                "drop_threshold": 0.6
            },
            {
                "name": "balanced",
                "tile_size": 512,
                "tta": preset.tta,
                "uhd": preset.uhd,
                "buffer_frames": preset.buffer_frames,
                "drop_threshold": preset.drop_threshold
            },
            {
                "name": "quality",
                "tile_size": 384,
                "tta": True,
                "uhd": preset.uhd,
                "buffer_frames": preset.buffer_frames * 2,
                "drop_threshold": 0.9
            }
        ]
        
        self.current_level = 2  # Start with balanced
        self.adaptation_history = []
        
    def should_adapt(self, stats: EngineStats) -> Optional[str]:
        """Check if quality adaptation is needed.
        
        Args:
            stats: Current engine statistics
            
        Returns:
            Adaptation direction ("up" or "down") or None
        """
        current_load = stats.processing_load
        avg_latency = stats.avg_latency_ms
        buffer_overruns = stats.buffer_overruns
        
        # Conditions for lowering quality
        if (current_load > 0.85 or 
            avg_latency > 100 or 
            buffer_overruns > 5):
            
            if self.current_level > 0:
                return "down"
        
        # Conditions for raising quality
        elif (current_load < 0.7 and 
              avg_latency < 50 and 
              buffer_overruns == 0 and
              len(self.adaptation_history) > 10):
            
            # Only raise if we've been stable for a while
            recent_adaptations = [a for a in self.adaptation_history[-10:] 
                                if time.time() - a["timestamp"] < 10]
            
            if len(recent_adaptations) == 0 and self.current_level < len(self.quality_levels) - 1:
                return "up"
        
        return None
    
    def adapt_quality(self, direction: str) -> bool:
        """Adapt quality level.
        
        Args:
            direction: Adaptation direction ("up" or "down")
            
        Returns:
            True if adaptation was applied
        """
        if direction == "down" and self.current_level > 0:
            self.current_level -= 1
            self._apply_quality_level()
            self._log_adaptation(direction)
            return True
        
        elif direction == "up" and self.current_level < len(self.quality_levels) - 1:
            self.current_level += 1
            self._apply_quality_level()
            self._log_adaptation(direction)
            return True
        
        return False
    
    def _apply_quality_level(self) -> None:
        """Apply current quality level to preset."""
        level = self.quality_levels[self.current_level]
        
        # Create new preset with adapted settings
        adapted_preset = InterpolationPreset(
            name=f"{self.base_preset.name}_{level['name']}",
            description=f"Adapted {level['name']} quality",
            target_fps=self.base_preset.target_fps,
            multiplier=self.base_preset.multiplier,
            model=self.base_preset.model,
            tta=level["tta"],
            uhd=level["uhd"],
            tile_size=level["tile_size"],
            buffer_frames=level["buffer_frames"],
            drop_threshold=level["drop_threshold"],
            scene_detection=self.base_preset.scene_detection,
            scene_threshold=self.base_preset.scene_threshold
        )
        
        self.current_preset = adapted_preset
        logger.info(f"Adapted quality to {level['name']} level")
    
    def _log_adaptation(self, direction: str) -> None:
        """Log quality adaptation."""
        self.adaptation_history.append({
            "timestamp": time.time(),
            "direction": direction,
            "level": self.current_level,
            "level_name": self.quality_levels[self.current_level]["name"]
        })
        
        # Keep only recent history
        cutoff = time.time() - 300  # 5 minutes
        self.adaptation_history = [
            a for a in self.adaptation_history 
            if a["timestamp"] > cutoff
        ]
    
    def get_current_preset(self) -> InterpolationPreset:
        """Get current adapted preset."""
        return self.current_preset


class FrameBufferManager:
    """Manages frame buffers for real-time processing."""
    
    def __init__(self, max_buffer_size: int = 16):
        """Initialize frame buffer manager.
        
        Args:
            max_buffer_size: Maximum number of frames to buffer
        """
        self.max_buffer_size = max_buffer_size
        self.input_buffer = queue.Queue(maxsize=max_buffer_size)
        self.output_buffer = queue.Queue(maxsize=max_buffer_size * 2)
        
        self.buffer_stats = {
            "input_overruns": 0,
            "output_overruns": 0,
            "frames_dropped": 0
        }
        
    def add_input_frame(self, frame_data: any, timestamp: float, frame_number: int) -> bool:
        """Add frame to input buffer.
        
        Args:
            frame_data: Frame data
            timestamp: Frame timestamp
            frame_number: Frame number
            
        Returns:
            True if frame was buffered, False if dropped
        """
        frame_info = {
            "data": frame_data,
            "timestamp": timestamp,
            "frame_number": frame_number,
            "added_at": time.time()
        }
        
        try:
            self.input_buffer.put_nowait(frame_info)
            return True
        except queue.Full:
            self.buffer_stats["input_overruns"] += 1
            # Drop oldest frame to make room
            try:
                self.input_buffer.get_nowait()
                self.input_buffer.put_nowait(frame_info)
                self.buffer_stats["frames_dropped"] += 1
                return True
            except queue.Empty:
                return False
    
    def get_input_frame(self, timeout: float = 0.1) -> Optional[Dict]:
        """Get frame from input buffer.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Frame info dictionary or None
        """
        try:
            return self.input_buffer.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def add_output_frame(self, frame_data: any, stats: FrameStats) -> bool:
        """Add processed frame to output buffer.
        
        Args:
            frame_data: Processed frame data
            stats: Frame processing statistics
            
        Returns:
            True if frame was buffered
        """
        frame_info = {
            "data": frame_data,
            "stats": stats,
            "output_timestamp": time.time()
        }
        
        try:
            self.output_buffer.put_nowait(frame_info)
            return True
        except queue.Full:
            self.buffer_stats["output_overruns"] += 1
            return False
    
    def get_output_frame(self, timeout: float = 0.1) -> Optional[Dict]:
        """Get processed frame from output buffer.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Frame info dictionary or None
        """
        try:
            return self.output_buffer.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_buffer_status(self) -> Dict[str, int]:
        """Get current buffer status.
        
        Returns:
            Dictionary with buffer statistics
        """
        return {
            "input_size": self.input_buffer.qsize(),
            "output_size": self.output_buffer.qsize(),
            "input_overruns": self.buffer_stats["input_overruns"],
            "output_overruns": self.buffer_stats["output_overruns"],
            "frames_dropped": self.buffer_stats["frames_dropped"]
        }


class RealtimeEngine:
    """Real-time interpolation engine."""
    
    def __init__(
        self,
        preset: InterpolationPreset,
        rife_binary_path: Optional[Path] = None,
        stats_callback: Optional[Callable[[EngineStats], None]] = None
    ):
        """Initialize real-time engine.
        
        Args:
            preset: Interpolation preset
            rife_binary_path: Path to RIFE binary
            stats_callback: Callback for statistics updates
        """
        self.preset = preset
        self.rife_binary_path = rife_binary_path
        self.stats_callback = stats_callback
        
        # Component initialization
        self.quality_manager = AdaptiveQualityManager(preset)
        self.buffer_manager = FrameBufferManager(preset.max_queue_size)
        self.scene_detector = SceneDetector(
            threshold=preset.scene_threshold,
            method=preset.scene_method
        ) if preset.scene_detection else None
        
        # Threading
        self.processing_thread = None
        self.stats_thread = None
        self.is_running = False
        
        # Statistics
        self.stats = EngineStats(target_fps=preset.target_fps)
        self.frame_history = queue.Queue(maxsize=100)
        
        # Performance monitoring
        self.last_frame_time = time.time()
        self.process = psutil.Process()
        
    def start(self) -> None:
        """Start the real-time engine."""
        if self.is_running:
            logger.warning("Engine is already running")
            return
        
        logger.info("Starting FlowForge real-time engine")
        self.is_running = True
        self.stats.start_time = time.time()
        
        # Start processing thread
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            name="FlowForge-Processing",
            daemon=True
        )
        self.processing_thread.start()
        
        # Start statistics thread
        self.stats_thread = threading.Thread(
            target=self._stats_loop,
            name="FlowForge-Stats",
            daemon=True
        )
        self.stats_thread.start()
        
        logger.info("Real-time engine started successfully")
    
    def stop(self) -> None:
        """Stop the real-time engine."""
        if not self.is_running:
            return
        
        logger.info("Stopping FlowForge real-time engine")
        self.is_running = False
        
        # Wait for threads to finish
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        if self.stats_thread:
            self.stats_thread.join(timeout=1.0)
        
        logger.info("Real-time engine stopped")
    
    def process_frame(self, frame_data: any, timestamp: float, frame_number: int) -> bool:
        """Add frame for processing.
        
        Args:
            frame_data: Frame data
            timestamp: Frame timestamp
            frame_number: Frame number
            
        Returns:
            True if frame was accepted for processing
        """
        return self.buffer_manager.add_input_frame(frame_data, timestamp, frame_number)
    
    def get_processed_frame(self, timeout: float = 0.1) -> Optional[Tuple[any, FrameStats]]:
        """Get processed frame.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Tuple of (frame_data, frame_stats) or None
        """
        frame_info = self.buffer_manager.get_output_frame(timeout)
        if frame_info:
            return frame_info["data"], frame_info["stats"]
        return None
    
    def _processing_loop(self) -> None:
        """Main processing loop."""
        logger.info("Processing loop started")
        
        while self.is_running:
            try:
                # Get frame from input buffer
                frame_info = self.buffer_manager.get_input_frame(timeout=0.1)
                if not frame_info:
                    continue
                
                # Create frame stats
                frame_stats = FrameStats(
                    frame_number=frame_info["frame_number"],
                    input_timestamp=frame_info["timestamp"],
                    processing_start=time.time(),
                    processing_end=0.0,
                    output_timestamp=0.0
                )
                
                # Process frame
                processed_data = self._process_single_frame(frame_info, frame_stats)
                
                # Update stats
                frame_stats.processing_end = time.time()
                frame_stats.output_timestamp = time.time()
                
                # Add to output buffer
                self.buffer_manager.add_output_frame(processed_data, frame_stats)
                
                # Update frame history for statistics
                try:
                    self.frame_history.put_nowait(frame_stats)
                except queue.Full:
                    # Remove oldest
                    try:
                        self.frame_history.get_nowait()
                        self.frame_history.put_nowait(frame_stats)
                    except queue.Empty:
                        pass
                
            except Exception as e:
                logger.error(f"Processing loop error: {e}")
                continue
        
        logger.info("Processing loop ended")
    
    def _process_single_frame(self, frame_info: Dict, frame_stats: FrameStats) -> any:
        """Process a single frame.
        
        Args:
            frame_info: Frame information
            frame_stats: Frame statistics (updated in place)
            
        Returns:
            Processed frame data
        """
        frame_data = frame_info["data"]
        
        # Check for scene change (simplified)
        scene_change = False
        if self.scene_detector and hasattr(self, '_last_frame_data'):
            # In real implementation, would convert frame data to numpy arrays
            # and use scene_detector.detect_scene_change()
            pass
        
        frame_stats.scene_change = scene_change
        
        # Decide whether to interpolate
        current_preset = self.quality_manager.get_current_preset()
        should_interpolate = not scene_change or not current_preset.scene_detection
        
        if should_interpolate:
            # In real implementation, this would call RIFE interpolation
            # For now, we simulate processing time
            processing_delay = self._simulate_processing_time(current_preset)
            time.sleep(processing_delay)
            
            frame_stats.interpolated = True
            self.stats.frames_interpolated += 1
        else:
            # Pass through original frame
            frame_stats.interpolated = False
        
        self.stats.frames_processed += 1
        self._last_frame_data = frame_data
        
        return frame_data
    
    def _simulate_processing_time(self, preset: InterpolationPreset) -> float:
        """Simulate RIFE processing time based on preset.
        
        Args:
            preset: Current preset
            
        Returns:
            Simulated processing time in seconds
        """
        # Base processing time varies by quality
        base_times = {
            "emergency": 0.005,
            "fast": 0.010,
            "balanced": 0.020,
            "quality": 0.040
        }
        
        quality_name = preset.quality_profile
        base_time = base_times.get(quality_name, 0.020)
        
        # Add factors for TTA, UHD, tile size
        if preset.tta:
            base_time *= 2.0
        if preset.uhd:
            base_time *= 1.5
        
        # Tile size factor (smaller tiles = more processing)
        tile_factor = 512 / preset.tile_size
        base_time *= tile_factor
        
        return base_time
    
    def _stats_loop(self) -> None:
        """Statistics update loop."""
        logger.info("Statistics loop started")
        
        while self.is_running:
            try:
                time.sleep(1.0)  # Update every second
                self._update_statistics()
                
                # Check for quality adaptation
                adaptation = self.quality_manager.should_adapt(self.stats)
                if adaptation:
                    if self.quality_manager.adapt_quality(adaptation):
                        self.stats.adaptive_changes += 1
                
                # Callback for statistics
                if self.stats_callback:
                    try:
                        self.stats_callback(self.stats)
                    except Exception as e:
                        logger.error(f"Stats callback error: {e}")
                
            except Exception as e:
                logger.error(f"Stats loop error: {e}")
                continue
        
        logger.info("Statistics loop ended")
    
    def _update_statistics(self) -> None:
        """Update engine statistics."""
        current_time = time.time()
        
        # Update buffer statistics
        buffer_status = self.buffer_manager.get_buffer_status()
        self.stats.input_buffer_size = buffer_status["input_size"]
        self.stats.output_buffer_size = buffer_status["output_size"]
        self.stats.buffer_overruns += buffer_status["input_overruns"] + buffer_status["output_overruns"]
        self.stats.frames_dropped = buffer_status["frames_dropped"]
        
        # Calculate FPS from recent frame history
        frame_times = []
        temp_history = []
        
        while not self.frame_history.empty():
            try:
                frame_stats = self.frame_history.get_nowait()
                temp_history.append(frame_stats)
                if current_time - frame_stats.output_timestamp < 5.0:  # Last 5 seconds
                    frame_times.append(frame_stats.processing_time_ms)
            except queue.Empty:
                break
        
        # Put frames back
        for frame_stats in temp_history:
            try:
                self.frame_history.put_nowait(frame_stats)
            except queue.Full:
                break
        
        # Update FPS calculations
        if frame_times:
            recent_frames = len(frame_times)
            self.stats.current_fps = recent_frames / 5.0  # Frames per second over 5 second window
            self.stats.avg_latency_ms = sum(frame_times) / len(frame_times)
            self.stats.max_latency_ms = max(frame_times)
            
            if self.stats.avg_latency_ms > 0:
                self.stats.processing_fps = 1000.0 / self.stats.avg_latency_ms
        
        # Update resource usage
        try:
            self.stats.cpu_usage_percent = self.process.cpu_percent()
            memory_info = self.process.memory_info()
            self.stats.memory_usage_mb = memory_info.rss / (1024 * 1024)
        except:
            pass
        
        # Update GPU usage (if available)
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                self.stats.gpu_usage_percent = gpu.load * 100
                self.stats.gpu_memory_mb = gpu.memoryUsed
        except:
            pass
        
        # Update quality level
        self.stats.quality_level = self.quality_manager.quality_levels[self.quality_manager.current_level]["name"]
        self.stats.last_update = current_time
    
    def get_stats(self) -> EngineStats:
        """Get current engine statistics.
        
        Returns:
            Current EngineStats instance
        """
        return self.stats
    
    def set_preset(self, preset: InterpolationPreset) -> None:
        """Update interpolation preset.
        
        Args:
            preset: New interpolation preset
        """
        logger.info(f"Updating preset to: {preset.name}")
        self.preset = preset
        self.quality_manager = AdaptiveQualityManager(preset)
        self.stats.target_fps = preset.target_fps
        
        # Update scene detector if needed
        if preset.scene_detection:
            self.scene_detector = SceneDetector(
                threshold=preset.scene_threshold,
                method=preset.scene_method
            )
        else:
            self.scene_detector = None
    
    def force_quality_level(self, level: str) -> bool:
        """Force specific quality level.
        
        Args:
            level: Quality level name (emergency, fast, balanced, quality)
            
        Returns:
            True if level was set successfully
        """
        level_names = [ql["name"] for ql in self.quality_manager.quality_levels]
        if level in level_names:
            self.quality_manager.current_level = level_names.index(level)
            self.quality_manager._apply_quality_level()
            logger.info(f"Forced quality level to: {level}")
            return True
        return False