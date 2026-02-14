"""mpv configuration generator for FlowForge real-time interpolation."""

import logging
import platform
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .presets import InterpolationPreset, PresetManager

logger = logging.getLogger(__name__)


class MPVConfigGenerator:
    """Generates mpv configuration files for real-time interpolation."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize mpv config generator.
        
        Args:
            config_dir: Directory to store mpv configuration files
        """
        self.config_dir = config_dir or Path.home() / ".flowforge" / "mpv"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.preset_manager = PresetManager()
        
        # Platform-specific paths
        self.is_windows = platform.system() == "Windows"
        self.is_wsl = "microsoft" in platform.uname().release.lower()
        
    def detect_mpv_path(self) -> Optional[Path]:
        """Detect mpv installation path.
        
        Returns:
            Path to mpv executable or None if not found
        """
        common_paths = []
        
        if self.is_windows or self.is_wsl:
            common_paths.extend([
                Path("C:/Program Files/SVP 4/mpv64/mpv.exe"),
                Path("C:/Program Files/mpv/mpv.exe"),
                Path("C:/Program Files (x86)/mpv/mpv.exe"),
                Path("C:/ProgramData/chocolatey/bin/mpv.exe"),
                Path("C:/msys64/mingw64/bin/mpv.exe")
            ])
        
        # Linux/Unix paths
        common_paths.extend([
            Path("/usr/bin/mpv"),
            Path("/usr/local/bin/mpv"),
            Path("/opt/mpv/bin/mpv"),
            Path.home() / ".local/bin/mpv"
        ])
        
        # Check PATH first
        mpv_in_path = shutil.which("mpv")
        if mpv_in_path:
            return Path(mpv_in_path)
        
        # Check common installation paths
        for path in common_paths:
            if path.exists():
                return path
        
        return None
    
    def detect_vapoursynth(self) -> Tuple[bool, Optional[Path]]:
        """Detect VapourSynth installation.
        
        Returns:
            Tuple of (is_installed, vapoursynth_path)
        """
        try:
            import vapoursynth as vs
            vs_path = Path(vs.__file__).parent
            return True, vs_path
        except ImportError:
            pass
        
        # Check common VapourSynth paths
        common_vs_paths = []
        
        if self.is_windows or self.is_wsl:
            common_vs_paths.extend([
                Path("C:/Program Files/VapourSynth"),
                Path("C:/Program Files (x86)/VapourSynth"),
                Path("C:/VapourSynth")
            ])
        
        common_vs_paths.extend([
            Path("/usr/local/lib/vapoursynth"),
            Path("/usr/lib/vapoursynth"),
            Path("/opt/vapoursynth")
        ])
        
        for path in common_vs_paths:
            if path.exists():
                return True, path
        
        return False, None
    
    def get_system_capabilities(self) -> Dict[str, any]:
        """Detect system capabilities for performance optimization.
        
        Returns:
            Dictionary with system capability information
        """
        import psutil
        
        capabilities = {
            "cpu_cores": psutil.cpu_count(),
            "ram_gb": psutil.virtual_memory().total / (1024**3),
            "gpu_available": False,
            "gpu_memory_gb": 0,
            "vulkan_available": False,
            "vapoursynth_installed": False,
            "vs_rife_plugin": False,
            "platform": platform.system(),
            "is_wsl": self.is_wsl
        }
        
        # GPU detection
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                capabilities["gpu_available"] = True
                capabilities["gpu_memory_gb"] = max(gpu.memoryTotal / 1024 for gpu in gpus)
        except ImportError:
            pass
        
        # VapourSynth detection
        vs_installed, vs_path = self.detect_vapoursynth()
        capabilities["vapoursynth_installed"] = vs_installed
        capabilities["vapoursynth_path"] = str(vs_path) if vs_path else None
        
        # Check for vs-rife-ncnn-vulkan plugin
        if vs_installed:
            try:
                import vapoursynth as vs
                core = vs.core
                # Try to access rife plugin
                if hasattr(core, 'rife'):
                    capabilities["vs_rife_plugin"] = True
            except:
                pass
        
        # Vulkan detection (approximate)
        vulkan_paths = [
            Path("/usr/lib/x86_64-linux-gnu/libvulkan.so.1"),
            Path("/usr/lib64/libvulkan.so.1"),
            Path("C:/Windows/System32/vulkan-1.dll")
        ]
        capabilities["vulkan_available"] = any(p.exists() for p in vulkan_paths)
        
        return capabilities
    
    def generate_mpv_conf(
        self, 
        preset: InterpolationPreset,
        enable_vapoursynth: bool = True,
        custom_options: Optional[Dict[str, str]] = None
    ) -> str:
        """Generate mpv.conf content.
        
        Args:
            preset: Interpolation preset
            enable_vapoursynth: Enable VapourSynth filter
            custom_options: Additional mpv options
            
        Returns:
            mpv.conf content as string
        """
        capabilities = self.get_system_capabilities()
        
        config_lines = [
            "# FlowForge mpv configuration",
            f"# Preset: {preset.name} - {preset.description}",
            "",
            "# General settings",
            "keep-open=yes",
            "save-position-on-quit=yes",
            "watch-later-directory=~/.flowforge/mpv/watch-later",
            "",
            "# Video output",
            "vo=gpu",
            "hwdec=auto-copy-safe" if capabilities["gpu_available"] else "hwdec=no",
            "gpu-api=vulkan" if capabilities["vulkan_available"] else "gpu-api=opengl",
            "",
            "# Performance optimization",
            f"vd-lavc-threads={min(capabilities['cpu_cores'], 16)}",
            "cache=yes",
            "demuxer-max-bytes=150MiB",
            "demuxer-max-back-bytes=75MiB",
            ""
        ]
        
        # Quality settings based on preset
        if preset.quality_profile == "fast":
            config_lines.extend([
                "# Fast quality profile",
                "scale=bilinear",
                "dscale=bilinear",
                "cscale=bilinear",
                "video-sync=display-resample"
            ])
        elif preset.quality_profile == "quality":
            config_lines.extend([
                "# High quality profile", 
                "scale=ewa_lanczossharp",
                "dscale=ewa_lanczos",
                "cscale=ewa_lanczos",
                "video-sync=display-resample",
                "interpolation=yes",
                "temporal-dither=yes"
            ])
        else:  # balanced
            config_lines.extend([
                "# Balanced quality profile",
                "scale=spline36",
                "dscale=mitchell",
                "cscale=spline36",
                "video-sync=display-resample"
            ])
        
        config_lines.extend(["", "# Audio settings"])
        
        # Audio settings
        config_lines.extend([
            "audio-buffer=1.0",
            "audio-file-auto=fuzzy",
            "volume-max=130",
            ""
        ])
        
        # VapourSynth integration
        if enable_vapoursynth and capabilities["vapoursynth_installed"]:
            vpy_script = self.config_dir / "flowforge_filter.vpy"
            
            config_lines.extend([
                "# FlowForge VapourSynth integration",
                f"vf=vapoursynth={vpy_script}",
                ""
            ])
        
        # OSD and interface
        config_lines.extend([
            "# OSD and interface",
            "osc=yes",
            "osd-bar=yes",
            "osd-duration=2000",
            "osd-status-msg='${time-pos} / ${duration} (${percent-pos}%) | FPS: ${fps} | FlowForge: ${?vf:ON:OFF}'",
            ""
        ])
        
        # Window settings
        config_lines.extend([
            "# Window settings",
            "geometry=50%:50%",
            "autofit-larger=90%x90%",
            "force-window=yes",
            ""
        ])
        
        # Subtitles
        config_lines.extend([
            "# Subtitles",
            "sub-auto=fuzzy",
            "sub-file-paths=ass:srt:sub:subs:subtitles",
            "embeddedfonts=yes",
            "sub-fix-timing=no",
            ""
        ])
        
        # Add custom options
        if custom_options:
            config_lines.extend([
                "# Custom options",
                *[f"{key}={value}" for key, value in custom_options.items()],
                ""
            ])
        
        return "\\n".join(config_lines)
    
    def generate_input_conf(self, preset: InterpolationPreset) -> str:
        """Generate input.conf with FlowForge hotkeys.
        
        Args:
            preset: Current interpolation preset
            
        Returns:
            input.conf content as string
        """
        config_lines = [
            "# FlowForge input configuration",
            "# Real-time interpolation controls",
            "",
            "# FlowForge hotkeys",
            "F1 show-text 'FlowForge Real-time Interpolation Controls:\\nF2: Toggle interpolation\\nF3/F4: Cycle presets\\nF5/F6: FPS +/-\\nF7: Show stats' 3000",
            "",
            "# Toggle interpolation on/off",
            "F2 vf toggle vapoursynth",
            "",
            "# Cycle through presets",
            "F3 script-message flowforge-preset-next",
            "F4 script-message flowforge-preset-prev", 
            "",
            "# FPS adjustment",
            "F5 script-message flowforge-fps-increase",
            "F6 script-message flowforge-fps-decrease",
            "",
            "# Show FlowForge stats",
            "F7 script-message flowforge-show-stats",
            "",
            "# Quality toggles",
            "Ctrl+1 script-message flowforge-quality-fast",
            "Ctrl+2 script-message flowforge-quality-balanced", 
            "Ctrl+3 script-message flowforge-quality-high",
            "",
            "# Scene detection toggle",
            "Ctrl+s script-message flowforge-toggle-scene-detection",
            "",
            "# Buffer controls",
            "Ctrl++ script-message flowforge-buffer-increase",
            "Ctrl+- script-message flowforge-buffer-decrease",
            "",
            "# Standard mpv controls",
            "SPACE cycle pause",
            "RIGHT seek 10",
            "LEFT seek -10",
            "UP seek 60", 
            "DOWN seek -60",
            "PGUP seek 600",
            "PGDWN seek -600",
            "",
            "= add volume 2",
            "- add volume -2",
            "m cycle mute",
            "",
            "f cycle fullscreen",
            "ESC set fullscreen no",
            "",
            "s screenshot",
            "S screenshot video",
            "",
            "j cycle sub",
            "J cycle sub down",
            "",
            "9 add volume -2",
            "0 add volume 2",
            "",
            "q quit",
            "Q quit-watch-later"
        ]
        
        return "\\n".join(config_lines)
    
    def generate_vapoursynth_script(
        self, 
        preset: InterpolationPreset,
        rife_binary_path: Optional[Path] = None
    ) -> str:
        """Generate VapourSynth script for real-time RIFE.
        
        Args:
            preset: Interpolation preset
            rife_binary_path: Path to rife-ncnn-vulkan binary
            
        Returns:
            VapourSynth script content
        """
        capabilities = self.get_system_capabilities()
        use_plugin = capabilities["vs_rife_plugin"]
        
        # Default binary path
        if not rife_binary_path:
            if self.is_wsl or self.is_windows:
                rife_binary_path = Path("/mnt/c/Users/Kad/Desktop/FlowForge/bin/rife-ncnn-vulkan.exe")
            else:
                rife_binary_path = Path.home() / ".flowforge" / "bin" / "rife-ncnn-vulkan"
        
        script_lines = [
            "# FlowForge VapourSynth Filter",
            f"# Preset: {preset.name}",
            f"# Target FPS: {preset.target_fps}",
            "",
            "import vapoursynth as vs",
            "import sys",
            "import os",
            "import subprocess",
            "import threading",
            "import queue",
            "import numpy as np",
            "from typing import Optional",
            "",
            "core = vs.core",
            "",
            "# Configuration",
            f"TARGET_FPS = {preset.target_fps}",
            f"MODEL = '{preset.model}'",
            f"TILE_SIZE = {preset.tile_size}",
            f"TILE_PAD = {preset.tile_pad}",
            f"TTA = {preset.tta}",
            f"UHD = {preset.uhd}",
            f"SCENE_DETECTION = {preset.scene_detection}",
            f"SCENE_THRESHOLD = {preset.scene_threshold}",
            f"BUFFER_FRAMES = {preset.buffer_frames}",
            f"USE_PLUGIN = {use_plugin}",
            f"RIFE_BINARY = r'{rife_binary_path}'",
            ""
        ]
        
        if use_plugin:
            # Use vs-rife-ncnn-vulkan plugin
            script_lines.extend([
                "# Using vs-rife-ncnn-vulkan plugin (preferred)",
                "def flowforge_interpolate(clip):",
                "    # Get input clip properties",
                "    input_fps = clip.fps_num / clip.fps_den",
                "    multiplier = TARGET_FPS / input_fps",
                "    ",
                "    if multiplier <= 1.0:",
                "        return clip  # No interpolation needed",
                "    ",
                "    # Apply RIFE interpolation",
                "    if SCENE_DETECTION:",
                "        # Scene change detection",
                "        sc = core.misc.SCDetect(clip, threshold=SCENE_THRESHOLD)",
                "        interpolated = core.rife.RIFE(",
                "            clip,",
                "            model=MODEL,",
                "            factor_num=int(multiplier * 1000),",
                "            factor_den=1000,",
                "            gpu_id=0,",
                "            tta=TTA,",
                "            uhd=UHD,",
                "            sc=sc",
                "        )",
                "    else:",
                "        interpolated = core.rife.RIFE(",
                "            clip,", 
                "            model=MODEL,",
                "            factor_num=int(multiplier * 1000),",
                "            factor_den=1000,",
                "            gpu_id=0,",
                "            tta=TTA,",
                "            uhd=UHD",
                "        )",
                "    ",
                "    # Adjust frame rate",
                "    interpolated = core.std.AssumeFPS(interpolated, fpsnum=TARGET_FPS, fpsden=1)",
                "    return interpolated",
                ""
            ])
        else:
            # Fallback using binary subprocess
            script_lines.extend([
                "# Using rife-ncnn-vulkan binary (fallback)",
                "class RIFEProcessor:",
                "    def __init__(self):",
                "        self.frame_queue = queue.Queue(maxsize=BUFFER_FRAMES * 2)",
                "        self.output_queue = queue.Queue(maxsize=BUFFER_FRAMES * 4)",
                "        self.processing_thread = None",
                "        self.input_fps = 24.0",
                "        self.frame_count = 0",
                "        ",
                "    def start_processing(self, clip):",
                "        self.input_fps = clip.fps_num / clip.fps_den",
                "        self.processing_thread = threading.Thread(",
                "            target=self._process_frames,",
                "            daemon=True",
                "        )",
                "        self.processing_thread.start()",
                "        ",
                "    def _process_frames(self):",
                "        while True:",
                "            try:",
                "                frame_pair = self.frame_queue.get(timeout=1.0)",
                "                if frame_pair is None:",
                "                    break",
                "                    ",
                "                frame1, frame2 = frame_pair",
                "                interpolated = self._interpolate_pair(frame1, frame2)",
                "                ",
                "                for frame in interpolated:",
                "                    self.output_queue.put(frame)",
                "                    ",
                "            except queue.Empty:",
                "                continue",
                "            except Exception as e:",
                "                print(f'RIFE processing error: {e}', file=sys.stderr)",
                "                continue",
                "        ",
                "    def _interpolate_pair(self, frame1, frame2):",
                "        # Save frames to temp files",
                "        import tempfile",
                "        with tempfile.TemporaryDirectory() as tmpdir:",
                "            frame1_path = os.path.join(tmpdir, 'frame1.png')",
                "            frame2_path = os.path.join(tmpdir, 'frame2.png')",
                "            output_dir = os.path.join(tmpdir, 'output')",
                "            os.makedirs(output_dir)",
                "            ",
                "            # Convert frames to numpy and save as PNG",
                "            # (Implementation details omitted for brevity)",
                "            ",
                "            # Call RIFE binary",
                "            multiplier = TARGET_FPS / self.input_fps",
                "            cmd = [",
                "                str(RIFE_BINARY),",
                "                '-i', tmpdir,",
                "                '-o', output_dir,",
                "                '-n', str(int(multiplier)),",
                "                '-m', MODEL,",
                "                '-g', '0',",
                "                '-j', '1:1:1'  # Threading",
                "            ]",
                "            ",
                "            if TTA:",
                "                cmd.append('-x')",
                "            if UHD:",
                "                cmd.append('-u')",
                "                ",
                "            try:",
                "                subprocess.run(cmd, check=True, capture_output=True)",
                "                # Load interpolated frames",
                "                # (Implementation details omitted)",
                "                return []  # Return list of interpolated frames",
                "            except subprocess.CalledProcessError:",
                "                return [frame1, frame2]  # Fallback",
                "",
                "processor = RIFEProcessor()",
                "",
                "def flowforge_interpolate(clip):",
                "    processor.start_processing(clip)",
                "    ",
                "    def get_frame(n, f):",
                "        # Frame processing logic",
                "        # (Simplified - full implementation would handle frame pairs)",
                "        return f",
                "    ",
                "    return core.std.ModifyFrame(clip, clip, get_frame)",
                ""
            ])
        
        # Main filter function
        script_lines.extend([
            "# Main filter entry point",
            "def main(video_in):",
            "    # Apply FlowForge interpolation",
            "    output = flowforge_interpolate(video_in)",
            "    return output",
            "",
            "# Export for mpv",
            "video_in = video_in",
            "video_out = main(video_in)"
        ])
        
        return "\\n".join(script_lines)
    
    def create_configuration(
        self, 
        preset: InterpolationPreset,
        rife_binary_path: Optional[Path] = None,
        custom_mpv_options: Optional[Dict[str, str]] = None
    ) -> Dict[str, Path]:
        """Create complete mpv configuration for FlowForge.
        
        Args:
            preset: Interpolation preset to use
            rife_binary_path: Path to RIFE binary
            custom_mpv_options: Custom mpv options
            
        Returns:
            Dictionary mapping config type to file path
        """
        config_files = {}
        
        # Generate mpv.conf
        mpv_conf_content = self.generate_mpv_conf(preset, True, custom_mpv_options)
        mpv_conf_path = self.config_dir / "mpv.conf"
        with open(mpv_conf_path, 'w') as f:
            f.write(mpv_conf_content)
        config_files["mpv_conf"] = mpv_conf_path
        
        # Generate input.conf
        input_conf_content = self.generate_input_conf(preset)
        input_conf_path = self.config_dir / "input.conf" 
        with open(input_conf_path, 'w') as f:
            f.write(input_conf_content)
        config_files["input_conf"] = input_conf_path
        
        # Generate VapourSynth script
        capabilities = self.get_system_capabilities()
        if capabilities["vapoursynth_installed"]:
            vpy_content = self.generate_vapoursynth_script(preset, rife_binary_path)
            vpy_path = self.config_dir / "flowforge_filter.vpy"
            with open(vpy_path, 'w') as f:
                f.write(vpy_content)
            config_files["vapoursynth_script"] = vpy_path
        
        # Create preset info file
        preset_info_path = self.config_dir / "current_preset.json"
        with open(preset_info_path, 'w') as f:
            import json
            json.dump(preset.to_dict(), f, indent=2)
        config_files["preset_info"] = preset_info_path
        
        logger.info(f"Created FlowForge configuration in {self.config_dir}")
        return config_files