"""
RIFE Player Playback Workers
mpv launch and export processing workers.
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional

from PyQt6.QtCore import QObject, QThread, pyqtSignal, QTimer
from PyQt6.QtWidgets import QMessageBox

from .utils import _find_tool, _find_mpv, is_wsl_path, is_windows, probe_video


class PlaybackWorker(QObject):
    """Worker for launching mpv with optional RIFE processing."""
    
    # Signals
    playback_started = pyqtSignal()
    playback_finished = pyqtSignal()
    error_occurred = pyqtSignal(str)
    
    def __init__(self, video_path: Path, rife_config: Dict[str, Any]):
        super().__init__()
        self.video_path = video_path
        self.rife_config = rife_config
        self.process: Optional[subprocess.Popen] = None
        self.gpu_id = rife_config.get("gpu_id", 0)  # Required attribute
    
    def start_playback(self) -> None:
        """Start video playback with optional RIFE processing."""
        try:
            # Validate paths
            if is_wsl_path(self.video_path) and is_windows():
                self.error_occurred.emit(
                    "Cannot play WSL paths on Windows. Please use a Windows path."
                )
                return
            
            if not self.video_path.exists():
                self.error_occurred.emit(f"Video file not found: {self.video_path}")
                return
            
            # Find mpv
            mpv_path = _find_mpv()
            
            if self.rife_config["enabled"]:
                self._start_with_rife(mpv_path)
            else:
                self._start_normal(mpv_path)
                
        except Exception as e:
            self.error_occurred.emit(f"Failed to start playback: {str(e)}")
    
    def _start_normal(self, mpv_path: str) -> None:
        """Start normal playback without RIFE."""
        cmd = [
            mpv_path,
            "--no-config",
            str(self.video_path)
        ]
        
        self.process = subprocess.Popen(
            cmd,
            cwd=str(self.video_path.parent),
            creationflags=subprocess.CREATE_NEW_CONSOLE if is_windows() else 0
        )
        
        self.playback_started.emit()
        
        # Monitor process in background
        QTimer.singleShot(1000, self._check_process)
    
    def _start_with_rife(self, mpv_path: str) -> None:
        """Start playback with RIFE VapourSynth filter."""
        # Generate VapourSynth script
        vs_script = self._generate_vapoursynth_script()
        if not vs_script:
            self.error_occurred.emit("Failed to generate VapourSynth script")
            return
        
        # Write script to temp file in project directory
        # This avoids mpv colon parsing issues with Windows paths
        if is_windows():
            # Use RIFE Player directory to avoid path issues
            flowforge_dir = Path(r"C:\Users\Kad\Desktop\FlowForge")
            if not flowforge_dir.exists():
                # Fallback to video directory
                flowforge_dir = self.video_path.parent
        else:
            flowforge_dir = self.video_path.parent
        
        script_file = flowforge_dir / f"rife_temp_{id(self)}.vpy"
        
        try:
            with open(script_file, 'w', encoding='utf-8') as f:
                f.write(vs_script)
            
            # Launch mpv with VapourSynth filter
            # Use just filename to avoid colon parsing issues
            cmd = [
                mpv_path,
                "--no-config",
                "--hwdec=no",  # Disable hardware decoding for VapourSynth
                f"--vf=vapoursynth:{script_file.name}",
                str(self.video_path)
            ]
            
            self.process = subprocess.Popen(
                cmd,
                cwd=str(flowforge_dir),
                creationflags=subprocess.CREATE_NEW_CONSOLE if is_windows() else 0
            )
            
            self.playback_started.emit()
            
            # Monitor process in background
            QTimer.singleShot(1000, self._check_process)
            
        except Exception as e:
            self.error_occurred.emit(f"Failed to create VapourSynth script: {str(e)}")
            # Clean up temp file
            if script_file.exists():
                try:
                    script_file.unlink()
                except:
                    pass
    
    def _generate_vapoursynth_script(self) -> Optional[str]:
        """Generate VapourSynth script for RIFE processing."""
        try:
            # Get video info for FPS calculation
            video_info = probe_video(self.video_path)
            if not video_info:
                return None
            
            source_fps = video_info.get("fps", 24)
            target_fps = self.rife_config["target_fps"]
            
            # Calculate multiplication factor
            if source_fps and target_fps:
                factor = target_fps / source_fps
                # Round to nearest reasonable factor
                if factor <= 1.5:
                    factor = 1
                elif factor <= 3:
                    factor = 2
                elif factor <= 4.5:
                    factor = 4
                else:
                    factor = int(factor)
            else:
                factor = 2
            
            # VapourSynth script template
            script = f'''import vapoursynth as vs
from vapoursynth import core

# Load RIFE plugin
core.std.LoadPlugin(r"{self.rife_config['vs_plugin_path']}")

# Load video
clip = core.ffms2.Source(r"{self.video_path}")

# Scene detection before color conversion
if {str(self.rife_config['scene_detection']).lower()}:
    clip = core.misc.SCDetect(clip, threshold={self.rife_config['scene_threshold']})

# Convert to RGBS for RIFE (required)
clip = core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s="709")

# Apply RIFE interpolation
clip = core.rife.RIFE(
    clip,
    model_path=r"{self.rife_config['rife_model_path']}",
    factor_num={int(factor)},
    factor_den=1,
    gpu_id={self.rife_config['gpu_id']},
    gpu_thread=2,
    sc_threshold={self.rife_config['scene_threshold'] if self.rife_config['scene_detection'] else 0.0}
)

# Convert back to YUV for output
clip = core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s="709")

clip.set_output()
'''
            return script
            
        except Exception as e:
            print(f"Error generating VapourSynth script: {e}")
            return None
    
    def _check_process(self) -> None:
        """Check if the playback process is still running."""
        if self.process:
            poll_result = self.process.poll()
            if poll_result is not None:
                # Process finished
                self.playback_finished.emit()
                self.process = None
            else:
                # Still running, check again later
                QTimer.singleShot(2000, self._check_process)
    
    def stop_playback(self) -> None:
        """Stop the current playback."""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                try:
                    self.process.kill()
                except ProcessLookupError:
                    pass
            finally:
                self.process = None


class ExportWorker(QObject):
    """Worker for batch export processing."""
    
    # Signals
    progress_updated = pyqtSignal(int)  # Progress percentage (0-100)
    status_updated = pyqtSignal(str)    # Status message
    export_finished = pyqtSignal(bool)  # Success/failure
    error_occurred = pyqtSignal(str)    # Error message
    
    def __init__(self, input_files: list[Path], output_dir: Path, 
                 rife_config: Dict[str, Any], export_settings: Dict[str, Any]):
        super().__init__()
        self.input_files = input_files
        self.output_dir = output_dir
        self.rife_config = rife_config
        self.export_settings = export_settings
        self.cancelled = False
    
    def start_export(self) -> None:
        """Start the batch export process."""
        try:
            self.status_updated.emit("Starting batch export...")
            self.progress_updated.emit(0)
            
            if not self.output_dir.exists():
                self.output_dir.mkdir(parents=True, exist_ok=True)
            
            total_files = len(self.input_files)
            
            for i, input_file in enumerate(self.input_files):
                if self.cancelled:
                    break
                
                self.status_updated.emit(f"Processing {input_file.name}...")
                
                # Process single file
                success = self._process_file(input_file)
                
                if not success and not self.cancelled:
                    self.error_occurred.emit(f"Failed to process {input_file.name}")
                    continue
                
                # Update progress
                progress = int((i + 1) / total_files * 100)
                self.progress_updated.emit(progress)
            
            if not self.cancelled:
                self.status_updated.emit("Export completed!")
                self.export_finished.emit(True)
            else:
                self.status_updated.emit("Export cancelled.")
                self.export_finished.emit(False)
                
        except Exception as e:
            self.error_occurred.emit(f"Export error: {str(e)}")
            self.export_finished.emit(False)
    
    def _process_file(self, input_file: Path) -> bool:
        """Process a single file."""
        try:
            output_file = self.output_dir / f"{input_file.stem}_rife{input_file.suffix}"
            
            # Generate VapourSynth script for this file
            vs_script = self._generate_vapoursynth_script(input_file)
            if not vs_script:
                return False
            
            # Write temporary VapourSynth script
            with tempfile.NamedTemporaryFile(mode='w', suffix='.vpy', 
                                           encoding='utf-8', delete=False) as f:
                f.write(vs_script)
                script_path = f.name
            
            try:
                # Use ffmpeg to process with VapourSynth
                ffmpeg = _find_tool("ffmpeg")
                
                cmd = [
                    ffmpeg,
                    "-f", "vapoursynth",
                    "-i", script_path,
                    "-c:v", "libx264",
                    "-preset", "medium",
                    "-crf", "18",
                    "-pix_fmt", "yuv420p",
                    "-y",  # Overwrite output
                    str(output_file)
                ]
                
                process = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=3600  # 1 hour timeout
                )
                
                return process.returncode == 0
                
            finally:
                # Clean up temp script
                try:
                    Path(script_path).unlink()
                except:
                    pass
                
        except Exception as e:
            print(f"Error processing {input_file}: {e}")
            return False
    
    def _generate_vapoursynth_script(self, input_file: Path) -> Optional[str]:
        """Generate VapourSynth script for a specific file."""
        try:
            # Get video info for FPS calculation
            video_info = probe_video(input_file)
            if not video_info:
                return None
            
            source_fps = video_info.get("fps", 24)
            target_fps = self.rife_config["target_fps"]
            
            # Calculate multiplication factor
            if source_fps and target_fps:
                factor = target_fps / source_fps
                # Round to nearest reasonable factor
                if factor <= 1.5:
                    factor = 1
                elif factor <= 3:
                    factor = 2
                elif factor <= 4.5:
                    factor = 4
                else:
                    factor = int(factor)
            else:
                factor = 2
            
            # VapourSynth script template
            script = f'''import vapoursynth as vs
from vapoursynth import core

# Load RIFE plugin
core.std.LoadPlugin(r"{self.rife_config['vs_plugin_path']}")

# Load video
clip = core.ffms2.Source(r"{input_file}")

# Scene detection before color conversion
if {str(self.rife_config['scene_detection']).lower()}:
    clip = core.misc.SCDetect(clip, threshold={self.rife_config['scene_threshold']})

# Convert to RGBS for RIFE (required)
clip = core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s="709")

# Apply RIFE interpolation
clip = core.rife.RIFE(
    clip,
    model_path=r"{self.rife_config['rife_model_path']}",
    factor_num={int(factor)},
    factor_den=1,
    gpu_id={self.rife_config['gpu_id']},
    gpu_thread=2,
    sc_threshold={self.rife_config['scene_threshold'] if self.rife_config['scene_detection'] else 0.0}
)

# Convert back to YUV for output
clip = core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s="709")

clip.set_output()
'''
            return script
            
        except Exception as e:
            print(f"Error generating VapourSynth script for {input_file}: {e}")
            return None
    
    def cancel_export(self) -> None:
        """Cancel the export process."""
        self.cancelled = True