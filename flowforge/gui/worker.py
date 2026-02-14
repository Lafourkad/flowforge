"""
FlowForge Background Worker Threads
Handles video processing without blocking the UI.
"""

import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional, Any

from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker

from .settings import settings


def _find_tool(name: str) -> str:
    """Find ffprobe/ffmpeg/rife binary, checking FlowForge bin dirs first."""
    import sys
    if getattr(sys, 'frozen', False):
        app_dir = Path(sys.executable).parent
    else:
        app_dir = Path(__file__).resolve().parent.parent.parent
    
    candidates = [
        app_dir / "bin" / f"{name}.exe",
        app_dir / "bin" / name,
        Path(rf"C:\Users\Kad\Desktop\FlowForge\bin\{name}.exe"),
        Path.home() / ".flowforge" / "bin" / f"{name}.exe",
    ]
    for c in candidates:
        try:
            if c.exists():
                return str(c)
        except (PermissionError, OSError):
            continue
    return name  # fallback to PATH


class VideoProcessorWorker(QThread):
    """Background worker for RIFE video processing."""
    
    # Signals
    progress_updated = pyqtSignal(int, str, dict)  # percentage, status, extra_info
    processing_finished = pyqtSignal(bool, str)    # success, message
    frame_extracted = pyqtSignal(int)              # frame_count
    rife_progress = pyqtSignal(int, int, float)    # current, total, fps
    encoding_started = pyqtSignal(str)             # encoder_name
    
    def __init__(self):
        super().__init__()
        self._mutex = QMutex()
        self._should_stop = False
        
        # Processing parameters
        self.input_path: Optional[Path] = None
        self.output_path: Optional[Path] = None
        self.fps_multiplier: int = 2
        self.target_fps: Optional[float] = None
        self.gpu_id: int = 0
        self.threads: str = "1:2:2"
        self.scene_detection: bool = True
        self.scene_threshold: float = 0.3
        self.encoding_preset: str = "Balanced"
        self.nvenc: bool = True
        self.crf: int = 18
        
        # Temporary directories
        self.temp_dir: Optional[Path] = None
        self.frames_in: Optional[Path] = None
        self.frames_out: Optional[Path] = None
    
    def setup_processing(self, input_path: Path, output_path: Path, **kwargs) -> None:
        """Setup processing parameters."""
        self.input_path = input_path
        self.output_path = output_path
        
        # Update parameters from kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def stop_processing(self) -> None:
        """Request processing to stop."""
        with QMutexLocker(self._mutex):
            self._should_stop = True
    
    def should_stop(self) -> bool:
        """Check if processing should stop."""
        with QMutexLocker(self._mutex):
            return self._should_stop
    
    def run(self) -> None:
        """Main processing loop."""
        try:
            with QMutexLocker(self._mutex):
                self._should_stop = False
            
            success = self._process_video()
            
            if success:
                self.processing_finished.emit(True, f"Processing completed successfully!")
            else:
                self.processing_finished.emit(False, "Processing was cancelled or failed.")
                
        except Exception as e:
            self.processing_finished.emit(False, f"Error during processing: {str(e)}")
        
        finally:
            self._cleanup()
    
    def _process_video(self) -> bool:
        """Main video processing pipeline."""
        if not self.input_path or not self.output_path:
            return False
        
        # Get video info
        self.progress_updated.emit(5, "Analyzing video...", {})
        video_info = self._get_video_info()
        if not video_info:
            return False
        
        if self.should_stop():
            return False
        
        # Calculate target parameters
        multiplier = self._calculate_multiplier(video_info["fps"])
        target_fps = video_info["fps"] * multiplier
        target_frames = int(video_info["frame_count"] * multiplier)
        
        self.progress_updated.emit(10, f"Target: {target_fps:.2f}fps ({multiplier}x)", {
            "input_fps": video_info["fps"],
            "target_fps": target_fps,
            "multiplier": multiplier
        })
        
        # Setup temp directories
        if not self._setup_temp_dirs():
            return False
        
        if self.should_stop():
            return False
        
        # Extract frames
        self.progress_updated.emit(15, "Extracting frames...", {})
        frame_count = self._extract_frames()
        if frame_count <= 0:
            return False
        
        self.frame_extracted.emit(frame_count)
        
        if self.should_stop():
            return False
        
        # Scene detection (optional)
        if self.scene_detection:
            self.progress_updated.emit(25, "Detecting scene changes...", {})
            scenes = self._detect_scenes()
            self.progress_updated.emit(30, f"Found {len(scenes)} scene changes", {
                "scene_count": len(scenes)
            })
        
        if self.should_stop():
            return False
        
        # RIFE interpolation
        self.progress_updated.emit(35, "Starting RIFE interpolation...", {})
        if not self._run_rife_interpolation(target_frames):
            return False
        
        if self.should_stop():
            return False
        
        # Encode video
        self.progress_updated.emit(85, "Encoding final video...", {})
        if not self._encode_video(target_fps):
            return False
        
        self.progress_updated.emit(100, "Processing complete!", {})
        return True
    
    def _get_video_info(self) -> Optional[Dict[str, Any]]:
        """Get video metadata using ffprobe."""
        try:
            cmd = [
                _find_tool("ffprobe"), "-v", "quiet", "-print_format", "json",
                "-show_streams", "-show_format", str(self.input_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                return None
            
            data = json.loads(result.stdout)
            
            for stream in data["streams"]:
                if stream["codec_type"] == "video":
                    fps_parts = stream["r_frame_rate"].split("/")
                    fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else float(fps_parts[0])
                    
                    duration = float(data["format"].get("duration", 0))
                    frame_count = int(duration * fps) if duration > 0 else 0
                    
                    return {
                        "fps": fps,
                        "width": int(stream["width"]),
                        "height": int(stream["height"]),
                        "duration": duration,
                        "frame_count": frame_count,
                        "codec": stream.get("codec_name", "unknown"),
                        "pix_fmt": stream.get("pix_fmt", "unknown"),
                    }
            
            return None
        
        except (subprocess.SubprocessError, json.JSONDecodeError, KeyError, ValueError):
            return None
    
    def _calculate_multiplier(self, source_fps: float) -> int:
        """Calculate FPS multiplier based on target FPS or multiplier setting."""
        if self.target_fps:
            ratio = self.target_fps / source_fps
            # Round to nearest power of 2 (2, 4, 8, etc.)
            return max(2, 2 ** round(math.log2(ratio)))
        return self.fps_multiplier
    
    def _setup_temp_dirs(self) -> bool:
        """Setup temporary directories for processing."""
        try:
            # Create temp directory
            self.temp_dir = Path(tempfile.mkdtemp(prefix="flowforge_"))
            self.frames_in = self.temp_dir / "input"
            self.frames_out = self.temp_dir / "output"
            
            self.frames_in.mkdir(parents=True, exist_ok=True)
            self.frames_out.mkdir(parents=True, exist_ok=True)
            
            return True
        
        except OSError:
            return False
    
    def _extract_frames(self) -> int:
        """Extract frames from input video."""
        try:
            cmd = [
                _find_tool("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(self.input_path),
                "-vsync", "0", "-q:v", "2",
                f"{self.frames_in}/%08d.png"
            ]
            
            result = subprocess.run(cmd, check=True, timeout=300)
            
            # Count extracted frames
            frame_count = len(list(self.frames_in.glob("*.png")))
            return frame_count
        
        except (subprocess.SubprocessError, subprocess.TimeoutExpired):
            return 0
    
    def _detect_scenes(self) -> list[int]:
        """Detect scene changes using file size heuristic."""
        frames = sorted(self.frames_in.glob("*.png"))
        if len(frames) < 3:
            return []
        
        sizes = [f.stat().st_size for f in frames]
        scenes = []
        
        for i in range(1, len(sizes)):
            if self.should_stop():
                break
            
            ratio = abs(sizes[i] - sizes[i-1]) / max(sizes[i-1], 1)
            if ratio > self.scene_threshold:
                scenes.append(i)
        
        return scenes
    
    def _run_rife_interpolation(self, target_frames: int) -> bool:
        """Run RIFE interpolation with progress monitoring."""
        rife_path = settings.get("rife_binary")
        model_path = settings.get("rife_model_dir")
        
        if not rife_path or not model_path:
            return False
        
        try:
            cmd = [
                settings.to_windows_path(rife_path),
                "-i", settings.to_windows_path(str(self.frames_in)),
                "-o", settings.to_windows_path(str(self.frames_out)),
                "-m", settings.to_windows_path(model_path),
                "-g", str(self.gpu_id),
                "-n", str(target_frames),
                "-j", self.threads,
                "-f", "%08d.png",
            ]
            
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True
            )
            
            done_count = 0
            start_time = time.time()
            last_update = 0
            
            for line in process.stdout:
                if self.should_stop():
                    process.terminate()
                    return False
                
                if "done" in line.lower():
                    done_count += 1
                    current_time = time.time()
                    
                    if current_time - last_update > 1.0:  # Update every second
                        elapsed = current_time - start_time
                        fps = done_count / max(elapsed, 0.1)
                        
                        progress = int(35 + (done_count / target_frames) * 50)  # 35-85%
                        self.progress_updated.emit(
                            progress, 
                            f"RIFE interpolating... ({done_count}/{target_frames})",
                            {"rife_fps": fps}
                        )
                        
                        self.rife_progress.emit(done_count, target_frames, fps)
                        last_update = current_time
            
            process.wait()
            return process.returncode == 0
        
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    def _encode_video(self, fps: float) -> bool:
        """Encode interpolated frames to final video."""
        try:
            # Determine encoding settings
            preset_map = {
                "Quality": ("slow", 16),
                "Balanced": ("medium", 18),
                "Fast": ("veryfast", 20)
            }
            x264_preset, default_crf = preset_map.get(self.encoding_preset, ("medium", 18))
            crf = self.crf if self.crf != 18 else default_crf
            
            cmd = [
                _find_tool("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error",
                "-framerate", str(fps),
                "-i", f"{self.frames_out}/%08d.png",
                "-i", str(self.input_path),  # For audio
                "-map", "0:v", "-map", "1:a",
                "-c:a", "copy", "-shortest"
            ]
            
            if self.nvenc:
                cmd.extend([
                    "-c:v", "h264_nvenc", "-preset", "p4", 
                    "-rc", "vbr", "-cq", str(crf), "-b:v", "0"
                ])
                encoder_name = "NVENC (GPU)"
            else:
                cmd.extend([
                    "-c:v", "libx264", "-preset", x264_preset, "-crf", str(crf)
                ])
                encoder_name = f"x264 ({x264_preset})"
            
            cmd.extend(["-pix_fmt", "yuv420p", str(self.output_path)])
            
            self.encoding_started.emit(encoder_name)
            
            result = subprocess.run(cmd, timeout=1800)  # 30 min timeout
            return result.returncode == 0
        
        except (subprocess.SubprocessError, subprocess.TimeoutExpired):
            return False
    
    def _cleanup(self) -> None:
        """Clean up temporary files."""
        if self.temp_dir and self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir)
            except OSError:
                pass  # Best effort cleanup


class PlaybackWorker(QThread):
    """Worker for launching real-time playback with mpv."""
    
    playback_started = pyqtSignal(bool, str)  # success, message
    
    def __init__(self):
        super().__init__()
        self.video_path: Optional[Path] = None
        self.preset: str = "Film (24→60fps)"
        self.custom_fps: float = 60.0
        self.gpu_threads: int = 2
        self.scene_detection: bool = True
        self.scene_threshold: float = 0.3
    
    def setup_playback(self, video_path: Path, **kwargs) -> None:
        """Setup playback parameters."""
        self.video_path = video_path
        
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def run(self) -> None:
        """Launch mpv with VapourSynth script."""
        try:
            success = self._launch_mpv()
            self.playback_started.emit(success, "Playback started" if success else "Failed to start playback")
        except Exception as e:
            self.playback_started.emit(False, f"Error: {str(e)}")
    
    def _launch_mpv(self) -> bool:
        """Launch mpv with dynamically generated VapourSynth script."""
        mpv_path = settings.get("mpv_path")
        vs_plugin_path = settings.get("vs_plugin_path")
        model_path = settings.get("rife_model_dir")
        
        if not all([mpv_path, vs_plugin_path, model_path]):
            return False
        
        # Generate VapourSynth script
        vs_script = self._generate_vapoursynth_script()
        if not vs_script:
            return False
        
        try:
            # Write temporary VapourSynth script
            temp_vs = Path(tempfile.mktemp(suffix=".vpy"))
            with open(temp_vs, 'w', encoding='utf-8') as f:
                f.write(vs_script)
            
            # Launch mpv
            cmd = [
                mpv_path,
                f"--demuxer-lavf-o=video_path={str(self.video_path)}",
                str(temp_vs)
            ]
            
            subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0)
            
            # Clean up temp file after a delay (mpv should have loaded it)
            self.msleep(2000)
            try:
                temp_vs.unlink()
            except OSError:
                pass
            
            return True
        
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            return False
    
    def _generate_vapoursynth_script(self) -> Optional[str]:
        """Generate VapourSynth script based on current settings."""
        vs_plugin_path = settings.get("vs_plugin_path")
        model_path = settings.get("rife_model_dir")
        
        if not vs_plugin_path or not model_path:
            return None
        
        # Determine target FPS from preset
        fps_map = {
            "Film (24→60fps)": 60,
            "Anime (24→60fps)": 60,
            "Sports (30→60fps)": 60,
            "Smooth (→144fps)": 144,
            "Custom": self.custom_fps
        }
        target_fps = fps_map.get(self.preset, 60)
        
        # Convert paths to Windows format if needed
        plugin_path = settings.to_windows_path(vs_plugin_path).replace('\\', '\\\\')
        model_dir = settings.to_windows_path(model_path).replace('\\', '\\\\')
        
        script = f'''# FlowForge Real-Time RIFE - {self.preset}
import vapoursynth as vs

core = vs.core

# Load the RIFE plugin
core.std.LoadPlugin(r"{plugin_path}")

# Get the video from mpv
clip = video_in

'''
        
        if self.scene_detection:
            script += f'''# Scene detection BEFORE color conversion
clip = core.misc.SCDetect(clip, threshold={self.scene_threshold:.3f})

'''
        
        script += f'''# Convert YUV -> RGBS (32-bit float RGB) as required by RIFE
clip = core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s="709")

# RIFE interpolation to {target_fps}fps
clip = core.rife.RIFE(
    clip,
    model=23,
    fps_num={int(target_fps)},
    fps_den=1,
    model_path=r"{model_dir}",
    gpu_id={self.gpu_id},
    gpu_thread={self.gpu_threads},
    sc={str(self.scene_detection).lower()},
    tta=False,
    uhd=False,
)

# Convert back to YUV for mpv output
clip = core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s="709")

clip.set_output()
'''
        
        return script