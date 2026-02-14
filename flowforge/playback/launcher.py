"""mpv launcher for FlowForge real-time interpolation."""

import json
import logging
import os
import platform
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

from .mpv_config import MPVConfigGenerator
from .presets import InterpolationPreset, PresetManager
from .realtime_engine import RealtimeEngine
from .vapoursynth_filter import create_vapoursynth_filter

logger = logging.getLogger(__name__)


class MPVLauncher:
    """Launches mpv with FlowForge real-time interpolation."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize mpv launcher.
        
        Args:
            config_dir: Directory for FlowForge configuration files
        """
        self.config_dir = config_dir or Path.home() / ".flowforge" / "mpv"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.preset_manager = PresetManager()
        self.config_generator = MPVConfigGenerator(self.config_dir)
        
        # Platform detection
        self.is_windows = platform.system() == "Windows"
        self.is_wsl = "microsoft" in platform.uname().release.lower()
        
        # Paths
        self.mpv_path = None
        self.rife_binary_path = None
        
        # Current state
        self.current_preset = None
        self.mpv_process = None
        self.realtime_engine = None
        
        logger.info(f"FlowForge MPV Launcher initialized (config: {self.config_dir})")
    
    def detect_system_components(self) -> Dict[str, any]:
        """Detect system components and capabilities.
        
        Returns:
            Dictionary with detection results
        """
        logger.info("Detecting system components...")
        
        detection_results = {
            "mpv_found": False,
            "mpv_path": None,
            "vapoursynth_found": False,
            "vapoursynth_path": None,
            "vs_rife_plugin": False,
            "rife_binary_found": False,
            "rife_binary_path": None,
            "system_capabilities": {},
            "recommendations": []
        }
        
        # Detect mpv
        mpv_path = self.config_generator.detect_mpv_path()
        if mpv_path:
            detection_results["mpv_found"] = True
            detection_results["mpv_path"] = str(mpv_path)
            self.mpv_path = mpv_path
        
        # Detect VapourSynth
        vs_found, vs_path = self.config_generator.detect_vapoursynth()
        detection_results["vapoursynth_found"] = vs_found
        if vs_path:
            detection_results["vapoursynth_path"] = str(vs_path)
        
        # Check for vs-rife plugin
        if vs_found:
            try:
                import vapoursynth as vs
                core = vs.core
                detection_results["vs_rife_plugin"] = hasattr(core, 'rife')
            except:
                pass
        
        # Detect RIFE binary
        rife_paths = self._find_rife_binary()
        if rife_paths:
            detection_results["rife_binary_found"] = True
            detection_results["rife_binary_path"] = str(rife_paths[0])
            self.rife_binary_path = rife_paths[0]
        
        # Get system capabilities
        detection_results["system_capabilities"] = self.config_generator.get_system_capabilities()
        
        # Generate recommendations
        detection_results["recommendations"] = self._generate_recommendations(detection_results)
        
        logger.info(f"Detection complete: mpv={detection_results['mpv_found']}, "
                   f"VapourSynth={detection_results['vapoursynth_found']}, "
                   f"RIFE={detection_results['rife_binary_found']}")
        
        return detection_results
    
    def _find_rife_binary(self) -> List[Path]:
        """Find RIFE binary in common locations.
        
        Returns:
            List of found RIFE binary paths
        """
        common_paths = [
            # WSL/Windows paths
            Path("/mnt/c/Users/Kad/Desktop/FlowForge/bin/rife-ncnn-vulkan.exe"),
            Path("/mnt/c/Program Files/FlowForge/bin/rife-ncnn-vulkan.exe"),
            
            # FlowForge installation paths
            Path.home() / ".flowforge" / "bin" / "rife-ncnn-vulkan",
            Path.home() / ".flowforge" / "bin" / "rife-ncnn-vulkan.exe",
            
            # System paths
            Path("/usr/local/bin/rife-ncnn-vulkan"),
            Path("/usr/bin/rife-ncnn-vulkan"),
            Path("/opt/rife/bin/rife-ncnn-vulkan")
        ]
        
        # Add Windows-specific paths if applicable
        if self.is_windows or self.is_wsl:
            common_paths.extend([
                Path("C:/Program Files/FlowForge/bin/rife-ncnn-vulkan.exe"),
                Path("C:/FlowForge/bin/rife-ncnn-vulkan.exe"),
                Path("C:/RIFE/rife-ncnn-vulkan.exe")
            ])
        
        found_paths = []
        for path in common_paths:
            if path.exists() and path.is_file():
                found_paths.append(path)
        
        return found_paths
    
    def _generate_recommendations(self, detection_results: Dict) -> List[str]:
        """Generate setup recommendations based on detection results.
        
        Args:
            detection_results: Detection results
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if not detection_results["mpv_found"]:
            if self.is_windows or self.is_wsl:
                recommendations.append(
                    "Install mpv: Download from https://mpv.io or install SVP 4 which includes mpv"
                )
            else:
                recommendations.append(
                    "Install mpv: sudo apt install mpv (Ubuntu/Debian) or equivalent for your distro"
                )
        
        if not detection_results["vapoursynth_found"]:
            recommendations.append(
                "Install VapourSynth: Run the installation script for your platform"
            )
        elif not detection_results["vs_rife_plugin"]:
            recommendations.append(
                "Install vs-rife-ncnn-vulkan plugin for best performance"
            )
        
        if not detection_results["rife_binary_found"]:
            recommendations.append(
                "Download RIFE binary: Run 'flowforge setup' to download dependencies"
            )
        
        capabilities = detection_results["system_capabilities"]
        if not capabilities.get("gpu_available", False):
            recommendations.append(
                "GPU not detected: Real-time interpolation will use CPU (slower performance)"
            )
        elif capabilities.get("gpu_memory_gb", 0) < 4:
            recommendations.append(
                "Low GPU memory detected: Use 'sports' or 'fast' presets for best performance"
            )
        
        return recommendations
    
    def configure_mpv(
        self, 
        preset: InterpolationPreset,
        force_reconfigure: bool = False
    ) -> Dict[str, Path]:
        """Configure mpv for FlowForge interpolation.
        
        Args:
            preset: Interpolation preset to use
            force_reconfigure: Force reconfiguration even if files exist
            
        Returns:
            Dictionary mapping config type to file path
        """
        logger.info(f"Configuring mpv for preset: {preset.name}")
        
        # Check if configuration already exists
        config_files = {
            "mpv_conf": self.config_dir / "mpv.conf",
            "input_conf": self.config_dir / "input.conf",
            "vapoursynth_script": self.config_dir / "flowforge_filter.vpy",
            "preset_info": self.config_dir / "current_preset.json"
        }
        
        if not force_reconfigure and all(f.exists() for f in config_files.values()):
            logger.info("Configuration already exists, skipping (use force_reconfigure=True to override)")
            return config_files
        
        # Create configuration
        try:
            generated_files = self.config_generator.create_configuration(
                preset=preset,
                rife_binary_path=self.rife_binary_path
            )
            
            # Create VapourSynth filter
            if generated_files.get("vapoursynth_script"):
                create_vapoursynth_filter(
                    preset=preset,
                    output_path=generated_files["vapoursynth_script"],
                    rife_binary_path=self.rife_binary_path
                )
            
            self.current_preset = preset
            logger.info(f"mpv configuration created successfully in {self.config_dir}")
            
            return generated_files
            
        except Exception as e:
            logger.error(f"Failed to configure mpv: {e}")
            raise
    
    def launch_mpv(
        self,
        video_file: Union[str, Path],
        preset: Optional[InterpolationPreset] = None,
        additional_args: Optional[List[str]] = None,
        wait_for_exit: bool = False
    ) -> subprocess.Popen:
        """Launch mpv with FlowForge interpolation.
        
        Args:
            video_file: Path to video file to play
            preset: Interpolation preset (uses current if None)
            additional_args: Additional mpv command line arguments
            wait_for_exit: Wait for mpv to exit before returning
            
        Returns:
            mpv subprocess.Popen object
            
        Raises:
            RuntimeError: If mpv is not found or configuration fails
        """
        # Ensure mpv is detected
        if not self.mpv_path:
            detection = self.detect_system_components()
            if not detection["mpv_found"]:
                raise RuntimeError("mpv not found. Please install mpv first.")
        
        # Use provided preset or current preset
        if preset:
            self.configure_mpv(preset)
        elif not self.current_preset:
            # Use default film preset
            self.current_preset = self.preset_manager.get_preset("film")
            self.configure_mpv(self.current_preset)
        
        # Build mpv command
        video_path = Path(video_file)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_file}")
        
        mpv_args = [
            str(self.mpv_path),
            str(video_path),
            f"--config-dir={self.config_dir}",
            f"--input-conf={self.config_dir}/input.conf"
        ]
        
        # Add additional arguments
        if additional_args:
            mpv_args.extend(additional_args)
        
        # Add platform-specific arguments
        if self.is_wsl:
            # For WSL, we might need to handle Windows paths differently
            mpv_args.extend([
                "--terminal=no",  # Disable terminal output in WSL
                "--msg-level=all=warn"  # Reduce log verbosity
            ])
        
        logger.info(f"Launching mpv: {' '.join(mpv_args[:3])} ...")
        
        try:
            # Start mpv process
            if self.is_windows or self.is_wsl:
                # On Windows/WSL, don't attach to console
                self.mpv_process = subprocess.Popen(
                    mpv_args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if self.is_windows else 0
                )
            else:
                # On Linux, allow mpv to use the terminal
                self.mpv_process = subprocess.Popen(mpv_args)
            
            logger.info(f"mpv launched successfully (PID: {self.mpv_process.pid})")
            
            # Wait for mpv to start
            time.sleep(1.0)
            
            if self.mpv_process.poll() is not None:
                # Process exited immediately, check for errors
                stdout, stderr = self.mpv_process.communicate()
                error_msg = f"mpv exited immediately. stderr: {stderr.decode() if stderr else 'None'}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Wait for exit if requested
            if wait_for_exit:
                return_code = self.mpv_process.wait()
                logger.info(f"mpv exited with code: {return_code}")
            
            return self.mpv_process
            
        except Exception as e:
            logger.error(f"Failed to launch mpv: {e}")
            raise
    
    def stop_mpv(self) -> None:
        """Stop currently running mpv process."""
        if self.mpv_process and self.mpv_process.poll() is None:
            logger.info("Stopping mpv process...")
            try:
                self.mpv_process.terminate()
                self.mpv_process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                logger.warning("mpv did not terminate, killing...")
                self.mpv_process.kill()
                self.mpv_process.wait(timeout=2.0)
            except Exception as e:
                logger.error(f"Error stopping mpv: {e}")
            finally:
                self.mpv_process = None
    
    def is_mpv_running(self) -> bool:
        """Check if mpv process is still running.
        
        Returns:
            True if mpv is running
        """
        return self.mpv_process is not None and self.mpv_process.poll() is None
    
    def get_current_preset(self) -> Optional[InterpolationPreset]:
        """Get currently configured preset.
        
        Returns:
            Current InterpolationPreset or None
        """
        return self.current_preset
    
    def switch_preset(self, preset_name: str) -> bool:
        """Switch to different interpolation preset.
        
        Args:
            preset_name: Name of preset to switch to
            
        Returns:
            True if preset was switched successfully
        """
        try:
            new_preset = self.preset_manager.get_preset(preset_name)
            self.configure_mpv(new_preset, force_reconfigure=True)
            
            # If mpv is running, we'd need to restart it or send IPC commands
            # For now, just update the configuration
            logger.info(f"Switched to preset: {preset_name}")
            return True
            
        except KeyError:
            logger.error(f"Preset not found: {preset_name}")
            return False
        except Exception as e:
            logger.error(f"Failed to switch preset: {e}")
            return False
    
    def create_portable_config(self, output_dir: Path) -> Dict[str, Path]:
        """Create portable mpv configuration directory.
        
        Args:
            output_dir: Directory to create portable config in
            
        Returns:
            Dictionary mapping config type to file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.current_preset:
            raise RuntimeError("No preset configured")
        
        logger.info(f"Creating portable config in: {output_dir}")
        
        # Create temporary config generator for output directory
        portable_generator = MPVConfigGenerator(output_dir)
        portable_files = portable_generator.create_configuration(
            preset=self.current_preset,
            rife_binary_path=self.rife_binary_path
        )
        
        # Copy additional files
        readme_content = f"""FlowForge Portable Configuration
====================================

This directory contains mpv configuration files for FlowForge real-time interpolation.

Preset: {self.current_preset.name}
Description: {self.current_preset.description}
Target FPS: {self.current_preset.target_fps}

Usage:
------
1. Ensure mpv, VapourSynth, and RIFE are installed on the target system
2. Launch mpv with: mpv --config-dir="{output_dir}" your_video.mp4

Files:
------
- mpv.conf: Main mpv configuration
- input.conf: Keyboard shortcuts and controls  
- flowforge_filter.vpy: VapourSynth interpolation script
- current_preset.json: Preset information

Controls:
---------
F1: Show help
F2: Toggle interpolation
F3/F4: Cycle presets (if multiple available)
F5/F6: Adjust FPS
F7: Show statistics

Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
FlowForge Version: 2.0.0
"""
        
        readme_path = output_dir / "README.txt"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        portable_files["readme"] = readme_path
        logger.info("Portable configuration created successfully")
        
        return portable_files
    
    def get_system_status(self) -> Dict[str, any]:
        """Get comprehensive system status.
        
        Returns:
            Dictionary with system status information
        """
        detection = self.detect_system_components()
        
        status = {
            "system_info": {
                "platform": platform.system(),
                "is_wsl": self.is_wsl,
                "python_version": platform.python_version()
            },
            "components": detection,
            "current_preset": self.current_preset.to_dict() if self.current_preset else None,
            "mpv_status": {
                "running": self.is_mpv_running(),
                "pid": self.mpv_process.pid if self.is_mpv_running() else None
            },
            "config_directory": str(self.config_dir),
            "last_check": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return status


def quick_play(
    video_file: Union[str, Path],
    preset_name: str = "film",
    config_dir: Optional[Path] = None,
    wait_for_exit: bool = True
) -> subprocess.Popen:
    """Quick play video with FlowForge interpolation.
    
    Args:
        video_file: Path to video file
        preset_name: Interpolation preset to use
        config_dir: Custom config directory
        wait_for_exit: Wait for mpv to exit
        
    Returns:
        mpv subprocess object
    """
    launcher = MPVLauncher(config_dir)
    
    # Detect system
    detection = launcher.detect_system_components()
    if not detection["mpv_found"]:
        raise RuntimeError("mpv not found")
    
    # Get preset
    preset = launcher.preset_manager.get_preset(preset_name)
    
    # Launch
    return launcher.launch_mpv(
        video_file=video_file,
        preset=preset,
        wait_for_exit=wait_for_exit
    )