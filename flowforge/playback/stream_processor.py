"""Streaming interpolation processor for real-time RIFE interpolation.

This module provides the core streaming interpolation functionality that powers
FlowForge's real-time video playback. It handles the pipeline:
FFmpeg decode → RIFE interpolate → FFmpeg encode → mpv
"""

import logging
import os
import queue
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import psutil

from .presets import InterpolationPreset

logger = logging.getLogger(__name__)


@dataclass
class StreamStats:
    """Statistics for stream processing."""
    frames_input: int = 0
    frames_output: int = 0
    frames_interpolated: int = 0
    frames_dropped: int = 0
    scene_changes: int = 0
    
    # Performance metrics
    input_fps: float = 0.0
    output_fps: float = 0.0
    processing_fps: float = 0.0
    avg_latency_ms: float = 0.0
    
    # Resource usage
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    gpu_utilization: float = 0.0
    
    # Buffer status
    staging_frames: int = 0
    output_frames: int = 0
    buffer_overruns: int = 0
    
    # Timing
    start_time: float = 0.0
    last_update: float = 0.0
    
    @property
    def uptime_seconds(self) -> float:
        """Stream uptime in seconds."""
        return time.time() - self.start_time
    
    @property
    def interpolation_ratio(self) -> float:
        """Ratio of interpolated to total frames."""
        return self.frames_interpolated / max(1, self.frames_output)


class SlidingWindowBuffer:
    """Manages sliding window buffer for frame processing."""
    
    def __init__(self, window_size: int = 8):
        """Initialize sliding window buffer.
        
        Args:
            window_size: Size of the sliding window
        """
        self.window_size = window_size
        self.frames = {}  # frame_number -> frame_data
        self.timestamps = {}  # frame_number -> timestamp
        self.staging_dir = None
        self.output_dir = None
        self.current_frame = 0
        
    def initialize_dirs(self) -> Tuple[Path, Path]:
        """Initialize staging and output directories.
        
        Returns:
            Tuple of (staging_dir, output_dir)
        """
        self.staging_dir = Path(tempfile.mkdtemp(prefix="flowforge_staging_"))
        self.output_dir = Path(tempfile.mkdtemp(prefix="flowforge_output_"))
        return self.staging_dir, self.output_dir
    
    def add_frame(self, frame_number: int, frame_data: bytes, timestamp: float) -> bool:
        """Add frame to buffer.
        
        Args:
            frame_number: Frame sequence number
            frame_data: Frame image data
            timestamp: Frame timestamp
            
        Returns:
            True if frame was added successfully
        """
        self.frames[frame_number] = frame_data
        self.timestamps[frame_number] = timestamp
        
        # Write frame to staging directory
        if self.staging_dir:
            frame_path = self.staging_dir / f"frame_{frame_number:08d}.png"
            try:
                with open(frame_path, 'wb') as f:
                    f.write(frame_data)
            except Exception as e:
                logger.error(f"Failed to write staging frame {frame_number}: {e}")
                return False
        
        # Cleanup old frames outside window
        cutoff = frame_number - self.window_size
        frames_to_remove = [fn for fn in self.frames.keys() if fn < cutoff]
        for fn in frames_to_remove:
            self._cleanup_frame(fn)
        
        return True
    
    def get_frame_pair(self, frame_number: int) -> Optional[Tuple[int, int]]:
        """Get frame pair for interpolation.
        
        Args:
            frame_number: Current frame number
            
        Returns:
            Tuple of (frame1_number, frame2_number) or None
        """
        if frame_number in self.frames and (frame_number - 1) in self.frames:
            return frame_number - 1, frame_number
        return None
    
    def has_frame(self, frame_number: int) -> bool:
        """Check if frame is in buffer.
        
        Args:
            frame_number: Frame number to check
            
        Returns:
            True if frame exists in buffer
        """
        return frame_number in self.frames
    
    def get_staging_path(self, frame_number: int) -> Optional[Path]:
        """Get staging file path for frame.
        
        Args:
            frame_number: Frame number
            
        Returns:
            Path to staging file or None
        """
        if self.staging_dir and frame_number in self.frames:
            return self.staging_dir / f"frame_{frame_number:08d}.png"
        return None
    
    def get_output_pattern(self, base_frame: int) -> str:
        """Get output filename pattern for interpolated frames.
        
        Args:
            base_frame: Base frame number for interpolation
            
        Returns:
            Output filename pattern
        """
        if self.output_dir:
            return str(self.output_dir / f"interp_{base_frame:08d}_%04d.png")
        return ""
    
    def _cleanup_frame(self, frame_number: int) -> None:
        """Remove frame from buffer and cleanup files.
        
        Args:
            frame_number: Frame number to cleanup
        """
        # Remove from memory
        self.frames.pop(frame_number, None)
        self.timestamps.pop(frame_number, None)
        
        # Remove staging file
        if self.staging_dir:
            staging_file = self.staging_dir / f"frame_{frame_number:08d}.png"
            try:
                if staging_file.exists():
                    staging_file.unlink()
            except Exception:
                pass
    
    def cleanup(self) -> None:
        """Cleanup all temporary files and directories."""
        import shutil
        
        for temp_dir in [self.staging_dir, self.output_dir]:
            if temp_dir and temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    logger.warning(f"Failed to cleanup {temp_dir}: {e}")
        
        self.frames.clear()
        self.timestamps.clear()


class RIFEProcessor:
    """RIFE binary processor for frame interpolation."""
    
    def __init__(self, preset: InterpolationPreset, rife_binary_path: Path):
        """Initialize RIFE processor.
        
        Args:
            preset: Interpolation preset
            rife_binary_path: Path to rife-ncnn-vulkan binary
        """
        self.preset = preset
        self.rife_binary_path = rife_binary_path
        self.process_pool = []  # For concurrent processing
        
        # Determine multiplier
        if preset.multiplier:
            self.multiplier = preset.multiplier
        else:
            # Calculate from target FPS (assume 24fps input for now)
            self.multiplier = preset.target_fps / 24.0
        
        # Performance optimization
        self.thread_count = min(4, psutil.cpu_count())
        
    def interpolate_pair(
        self, 
        staging_dir: Path, 
        output_dir: Path,
        frame1_num: int, 
        frame2_num: int
    ) -> List[Path]:
        """Interpolate between two frames using RIFE binary.
        
        Args:
            staging_dir: Directory containing input frames
            output_dir: Directory for output frames
            frame1_num: First frame number
            frame2_num: Second frame number
            
        Returns:
            List of paths to interpolated frames
        """
        try:
            # Prepare input frames
            frame1_path = staging_dir / f"frame_{frame1_num:08d}.png"
            frame2_path = staging_dir / f"frame_{frame2_num:08d}.png"
            
            if not (frame1_path.exists() and frame2_path.exists()):
                logger.warning(f"Missing input frames: {frame1_num}, {frame2_num}")
                return []
            
            # Create pair directory
            pair_dir = staging_dir / f"pair_{frame1_num}_{frame2_num}"
            pair_dir.mkdir(exist_ok=True)
            
            # Copy frames to pair directory with RIFE naming convention
            import shutil
            shutil.copy2(frame1_path, pair_dir / "00000000.png")
            shutil.copy2(frame2_path, pair_dir / "00000001.png")
            
            # Prepare output directory
            output_pair_dir = output_dir / f"interp_{frame1_num}_{frame2_num}"
            output_pair_dir.mkdir(exist_ok=True)
            
            # Build RIFE command
            cmd = [
                str(self.rife_binary_path),
                "-i", str(pair_dir),
                "-o", str(output_pair_dir),
                "-n", str(int(self.multiplier)),
                "-m", self.preset.model
            ]
            
            # Add optional flags
            if self.preset.tta:
                cmd.append("-x")
            if self.preset.uhd:
                cmd.append("-u")
            
            # GPU settings
            cmd.extend(["-g", "0"])  # Use first GPU
            
            # Threading settings
            cmd.extend(["-j", f"{self.thread_count}:1:1"])
            
            logger.debug(f"Running RIFE: {' '.join(cmd)}")
            
            # Execute RIFE
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10.0  # 10 second timeout per pair
            )
            
            process_time = time.time() - start_time
            
            if result.returncode != 0:
                logger.error(f"RIFE failed: {result.stderr}")
                return []
            
            # Collect output frames
            output_frames = []
            for output_file in sorted(output_pair_dir.glob("*.png")):
                output_frames.append(output_file)
            
            logger.debug(f"RIFE processed {frame1_num}->{frame2_num} in {process_time:.3f}s, "
                        f"generated {len(output_frames)} frames")
            
            # Cleanup pair directory
            shutil.rmtree(pair_dir, ignore_errors=True)
            
            return output_frames
            
        except subprocess.TimeoutExpired:
            logger.error(f"RIFE timeout for frames {frame1_num}-{frame2_num}")
            return []
        except Exception as e:
            logger.error(f"RIFE processing error: {e}")
            return []
    
    def estimate_processing_time(self) -> float:
        """Estimate processing time per frame pair.
        
        Returns:
            Estimated time in seconds
        """
        # Base processing times by quality
        base_times = {
            "rife-v4.6": 0.050,
            "rife-v4.15-lite": 0.025
        }
        
        base_time = base_times.get(self.preset.model, 0.040)
        
        # Factor in options
        if self.preset.tta:
            base_time *= 2.0
        if self.preset.uhd:
            base_time *= 1.5
        
        # Multiplier factor
        base_time *= self.multiplier
        
        return base_time


class StreamProcessor:
    """Main streaming interpolation processor.
    
    This class manages the complete pipeline:
    1. FFmpeg decodes video frames to staging directory
    2. RIFE processes frame pairs from staging to output directory
    3. FFmpeg reads output frames and pipes to mpv
    4. All happens concurrently with sliding window buffer management
    """
    
    def __init__(
        self,
        preset: InterpolationPreset,
        rife_binary_path: Path,
        input_source: Union[str, Path],
        stats_callback: Optional[Callable[[StreamStats], None]] = None
    ):
        """Initialize stream processor.
        
        Args:
            preset: Interpolation preset
            rife_binary_path: Path to RIFE binary
            input_source: Video input source (file path or stream URL)
            stats_callback: Optional callback for statistics updates
        """
        self.preset = preset
        self.rife_binary_path = rife_binary_path
        self.input_source = Path(input_source)
        self.stats_callback = stats_callback
        
        # Components
        self.buffer = SlidingWindowBuffer(preset.buffer_frames)
        self.rife_processor = RIFEProcessor(preset, rife_binary_path)
        
        # Threading
        self.decode_thread = None
        self.interpolation_thread = None
        self.encode_thread = None
        self.stats_thread = None
        
        # State
        self.is_running = False
        self.should_stop = threading.Event()
        self.stats = StreamStats()
        
        # Queues for inter-thread communication
        self.decode_queue = queue.Queue(maxsize=preset.max_queue_size)
        self.interpolation_queue = queue.Queue(maxsize=preset.max_queue_size)
        self.output_queue = queue.Queue(maxsize=preset.max_queue_size * 2)
        
        # Scene detection
        self.last_frame_hash = None
        
        logger.info(f"StreamProcessor initialized: {preset.name} -> {preset.target_fps}fps")
    
    def start(self, output_pipe: Optional[int] = None) -> bool:
        """Start streaming interpolation.
        
        Args:
            output_pipe: Optional output pipe file descriptor
            
        Returns:
            True if started successfully
        """
        if self.is_running:
            logger.warning("Stream processor already running")
            return False
        
        try:
            # Initialize buffer directories
            staging_dir, output_dir = self.buffer.initialize_dirs()
            logger.info(f"Staging: {staging_dir}, Output: {output_dir}")
            
            # Reset state
            self.should_stop.clear()
            self.stats = StreamStats(start_time=time.time())
            
            # Start threads
            self.decode_thread = threading.Thread(
                target=self._decode_loop,
                name="FlowForge-Decode",
                daemon=True
            )
            
            self.interpolation_thread = threading.Thread(
                target=self._interpolation_loop,
                name="FlowForge-Interpolate",
                daemon=True
            )
            
            self.encode_thread = threading.Thread(
                target=self._encode_loop,
                name="FlowForge-Encode",
                args=(output_pipe,),
                daemon=True
            )
            
            self.stats_thread = threading.Thread(
                target=self._stats_loop,
                name="FlowForge-Stats",
                daemon=True
            )
            
            # Start all threads
            self.decode_thread.start()
            self.interpolation_thread.start() 
            self.encode_thread.start()
            self.stats_thread.start()
            
            self.is_running = True
            logger.info("Stream processor started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start stream processor: {e}")
            self.stop()
            return False
    
    def stop(self) -> None:
        """Stop streaming interpolation."""
        if not self.is_running:
            return
        
        logger.info("Stopping stream processor...")
        self.should_stop.set()
        self.is_running = False
        
        # Wait for threads to finish
        threads = [
            self.decode_thread,
            self.interpolation_thread,
            self.encode_thread,
            self.stats_thread
        ]
        
        for thread in threads:
            if thread and thread.is_alive():
                thread.join(timeout=2.0)
                if thread.is_alive():
                    logger.warning(f"Thread {thread.name} did not stop gracefully")
        
        # Cleanup
        self.buffer.cleanup()
        logger.info("Stream processor stopped")
    
    def _decode_loop(self) -> None:
        """Decode input video frames to staging directory."""
        logger.info("Decode loop started")
        
        try:
            # Build FFmpeg decode command
            cmd = [
                "ffmpeg",
                "-i", str(self.input_source),
                "-vf", "fps=24",  # Normalize input FPS for now
                "-f", "image2",
                "-vcodec", "png",
                "-y",  # Overwrite output files
                f"{self.buffer.staging_dir}/frame_%08d.png"
            ]
            
            logger.info(f"Starting FFmpeg decode: {' '.join(cmd[:4])} ...")
            
            # Start FFmpeg process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            frame_number = 1
            
            # Monitor for new frames
            while not self.should_stop.is_set():
                frame_path = self.buffer.staging_dir / f"frame_{frame_number:08d}.png"
                
                if frame_path.exists():
                    # Read frame data
                    try:
                        with open(frame_path, 'rb') as f:
                            frame_data = f.read()
                        
                        # Add to buffer
                        timestamp = time.time()
                        if self.buffer.add_frame(frame_number, frame_data, timestamp):
                            # Queue for interpolation
                            try:
                                self.decode_queue.put((frame_number, timestamp), timeout=0.1)
                                self.stats.frames_input += 1
                            except queue.Full:
                                self.stats.frames_dropped += 1
                        
                        frame_number += 1
                        
                    except Exception as e:
                        logger.error(f"Error reading frame {frame_number}: {e}")
                        break
                else:
                    # Check if FFmpeg is still running
                    if process.poll() is not None:
                        # Process finished
                        break
                    
                    # Wait a bit for next frame
                    time.sleep(0.001)  # 1ms
            
            # Cleanup FFmpeg process
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
            
        except Exception as e:
            logger.error(f"Decode loop error: {e}")
        
        logger.info("Decode loop ended")
    
    def _interpolation_loop(self) -> None:
        """Process frame pairs for interpolation."""
        logger.info("Interpolation loop started")
        
        while not self.should_stop.is_set():
            try:
                # Get frame for interpolation
                frame_info = self.decode_queue.get(timeout=0.5)
                if frame_info is None:
                    continue
                
                frame_number, timestamp = frame_info
                
                # Get frame pair
                pair = self.buffer.get_frame_pair(frame_number)
                if not pair:
                    continue
                
                frame1_num, frame2_num = pair
                
                # Check for scene change (simplified)
                scene_change = self._detect_scene_change(frame1_num, frame2_num)
                if scene_change:
                    self.stats.scene_changes += 1
                
                # Decide whether to interpolate
                should_interpolate = not scene_change or not self.preset.scene_detection
                
                if should_interpolate:
                    # Perform interpolation
                    start_time = time.time()
                    
                    interpolated_frames = self.rife_processor.interpolate_pair(
                        self.buffer.staging_dir,
                        self.buffer.output_dir,
                        frame1_num,
                        frame2_num
                    )
                    
                    process_time = time.time() - start_time
                    
                    if interpolated_frames:
                        # Queue interpolated frames for encoding
                        for i, frame_path in enumerate(interpolated_frames):
                            output_timestamp = timestamp + (i * (1.0 / self.preset.target_fps))
                            self.interpolation_queue.put((frame_path, output_timestamp))
                            self.stats.frames_interpolated += 1
                    else:
                        # Fallback: use original frames
                        original_path = self.buffer.get_staging_path(frame1_num)
                        if original_path:
                            self.interpolation_queue.put((original_path, timestamp))
                        
                else:
                    # Pass through original frame
                    original_path = self.buffer.get_staging_path(frame1_num)
                    if original_path:
                        self.interpolation_queue.put((original_path, timestamp))
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Interpolation loop error: {e}")
                continue
        
        logger.info("Interpolation loop ended")
    
    def _encode_loop(self, output_pipe: Optional[int]) -> None:
        """Encode interpolated frames and output to mpv."""
        logger.info("Encode loop started")
        
        try:
            # Build FFmpeg encode command
            if output_pipe:
                # Pipe to mpv
                cmd = [
                    "ffmpeg",
                    "-f", "image2pipe",
                    "-vcodec", "png",
                    "-r", str(self.preset.target_fps),
                    "-i", "-",  # Read from stdin
                    "-f", "rawvideo",
                    "-pix_fmt", "rgb24",
                    "-"  # Write to stdout
                ]
                
                # Start FFmpeg encoder
                encoder = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=output_pipe,
                    stderr=subprocess.PIPE
                )
            else:
                # For testing, just consume frames
                encoder = None
            
            frame_count = 0
            last_fps_time = time.time()
            
            while not self.should_stop.is_set():
                try:
                    # Get interpolated frame
                    frame_info = self.interpolation_queue.get(timeout=0.5)
                    if frame_info is None:
                        continue
                    
                    frame_path, timestamp = frame_info
                    
                    if encoder:
                        # Send frame to encoder
                        try:
                            with open(frame_path, 'rb') as f:
                                frame_data = f.read()
                            encoder.stdin.write(frame_data)
                            encoder.stdin.flush()
                        except Exception as e:
                            logger.error(f"Error encoding frame: {e}")
                            break
                    
                    self.stats.frames_output += 1
                    frame_count += 1
                    
                    # Update output FPS
                    current_time = time.time()
                    if current_time - last_fps_time >= 1.0:
                        self.stats.output_fps = frame_count / (current_time - last_fps_time)
                        frame_count = 0
                        last_fps_time = current_time
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Encode loop error: {e}")
                    break
            
            # Cleanup encoder
            if encoder:
                encoder.stdin.close()
                encoder.wait(timeout=2.0)
            
        except Exception as e:
            logger.error(f"Encode loop error: {e}")
        
        logger.info("Encode loop ended")
    
    def _stats_loop(self) -> None:
        """Statistics monitoring loop."""
        logger.info("Stats loop started")
        
        while not self.should_stop.is_set():
            try:
                time.sleep(1.0)  # Update every second
                
                # Update system stats
                try:
                    process = psutil.Process()
                    self.stats.cpu_percent = process.cpu_percent()
                    self.stats.memory_mb = process.memory_info().rss / (1024 * 1024)
                except:
                    pass
                
                # Update buffer stats
                self.stats.staging_frames = len(self.buffer.frames)
                self.stats.output_frames = self.interpolation_queue.qsize()
                
                # Calculate processing FPS
                if self.stats.frames_interpolated > 0:
                    uptime = self.stats.uptime_seconds
                    if uptime > 0:
                        self.stats.processing_fps = self.stats.frames_interpolated / uptime
                
                self.stats.last_update = time.time()
                
                # Callback
                if self.stats_callback:
                    try:
                        self.stats_callback(self.stats)
                    except Exception as e:
                        logger.error(f"Stats callback error: {e}")
                
            except Exception as e:
                logger.error(f"Stats loop error: {e}")
                time.sleep(1.0)
        
        logger.info("Stats loop ended")
    
    def _detect_scene_change(self, frame1_num: int, frame2_num: int) -> bool:
        """Simple scene change detection.
        
        Args:
            frame1_num: First frame number
            frame2_num: Second frame number
            
        Returns:
            True if scene change detected
        """
        # Simplified scene detection based on file size difference
        try:
            path1 = self.buffer.get_staging_path(frame1_num)
            path2 = self.buffer.get_staging_path(frame2_num)
            
            if path1 and path2 and path1.exists() and path2.exists():
                size1 = path1.stat().st_size
                size2 = path2.stat().st_size
                
                # Large size difference indicates scene change
                size_diff = abs(size1 - size2) / max(size1, size2)
                return size_diff > self.preset.scene_threshold
        except:
            pass
        
        return False
    
    def get_stats(self) -> StreamStats:
        """Get current streaming statistics.
        
        Returns:
            Current StreamStats instance
        """
        return self.stats
    
    def is_healthy(self) -> bool:
        """Check if stream processor is healthy.
        
        Returns:
            True if processor is running and healthy
        """
        if not self.is_running:
            return False
        
        # Check if threads are alive
        threads = [self.decode_thread, self.interpolation_thread, 
                  self.encode_thread, self.stats_thread]
        
        for thread in threads:
            if thread and not thread.is_alive():
                return False
        
        # Check for reasonable frame rates
        if self.stats.uptime_seconds > 5.0:  # After 5 seconds
            if self.stats.input_fps < 1.0 or self.stats.output_fps < 1.0:
                return False
        
        return True


def create_stream_processor(
    preset: InterpolationPreset,
    input_source: Union[str, Path],
    rife_binary_path: Optional[Path] = None,
    stats_callback: Optional[Callable[[StreamStats], None]] = None
) -> StreamProcessor:
    """Create and configure a stream processor.
    
    Args:
        preset: Interpolation preset
        input_source: Video input source
        rife_binary_path: Path to RIFE binary (auto-detected if None)
        stats_callback: Optional stats callback
        
    Returns:
        Configured StreamProcessor instance
        
    Raises:
        RuntimeError: If RIFE binary not found
    """
    # Auto-detect RIFE binary if not provided
    if not rife_binary_path:
        import platform
        is_wsl = "microsoft" in platform.uname().release.lower()
        
        common_paths = [
            Path("/mnt/c/Users/Kad/Desktop/FlowForge/bin/rife-ncnn-vulkan.exe"),
            Path.home() / ".flowforge" / "bin" / "rife-ncnn-vulkan"
        ]
        
        for path in common_paths:
            if path.exists():
                rife_binary_path = path
                break
        
        if not rife_binary_path:
            raise RuntimeError("RIFE binary not found. Please specify rife_binary_path or run 'flowforge setup'")
    
    return StreamProcessor(preset, rife_binary_path, input_source, stats_callback)