"""FlowForge command-line interface."""

import logging
import sys
from pathlib import Path
from typing import Optional

import click
from tqdm import tqdm

from . import __version__
from .core.interpolator import VideoInterpolator
from .playback.launcher import MPVLauncher, quick_play
from .playback.presets import PresetManager, get_system_recommendations
from .utils.download import ModelDownloader


# Configure logging
def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    """Setup logging configuration."""
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stderr)
        ]
    )


# Global progress bar for CLI feedback
_current_progress_bar = None


def progress_callback(stage: str, progress: float) -> None:
    """Progress callback for CLI operations."""
    global _current_progress_bar
    
    # Create new progress bar for new stage
    if _current_progress_bar is None or _current_progress_bar.desc != stage:
        if _current_progress_bar:
            _current_progress_bar.close()
        
        _current_progress_bar = tqdm(
            total=100,
            desc=stage,
            unit="%",
            bar_format="{l_bar}{bar}| {n:.1f}/{total:.1f}% [{elapsed}<{remaining}]"
        )
    
    # Update progress
    new_progress = int(progress * 100)
    if new_progress > _current_progress_bar.n:
        _current_progress_bar.update(new_progress - _current_progress_bar.n)
    
    # Close progress bar when complete
    if progress >= 1.0:
        _current_progress_bar.close()
        _current_progress_bar = None


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--quiet', '-q', is_flag=True, help='Quiet output (warnings only)')
@click.version_option(version=__version__, prog_name='FlowForge')
@click.pass_context
def cli(ctx, verbose: bool, quiet: bool):
    """FlowForge - High-quality video frame interpolation using RIFE.
    
    An open-source alternative to SVP 4 Pro for smooth video playback.
    """
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['quiet'] = quiet
    setup_logging(verbose, quiet)


@cli.command()
@click.argument('input_video', type=click.Path(exists=True, path_type=Path))
@click.option('-o', '--output', 'output_video', required=True,
              type=click.Path(path_type=Path), help='Output video file')
@click.option('--fps', type=float, help='Target frame rate (alternative to multiplier)')
@click.option('--multiplier', '-m', type=click.Choice(['2', '4', '8']),
              help='Interpolation multiplier (2x, 4x, 8x)')
@click.option('--model', type=click.Choice(['rife-v4.6', 'rife-v4.15-lite']),
              default='rife-v4.6', help='RIFE model to use')
@click.option('--gpu', type=int, default=0, help='GPU device ID (-1 for CPU)')
@click.option('--threads', type=int, default=1, help='Number of threads')
@click.option('--start-time', type=float, help='Start time in seconds')
@click.option('--end-time', type=float, help='End time in seconds')
@click.option('--scene-threshold', type=float, default=0.3,
              help='Scene change detection threshold (0.0-1.0)')
@click.option('--no-audio', is_flag=True, help='Remove audio from output')
@click.option('--no-subtitles', is_flag=True, help='Remove subtitles from output')
@click.option('--codec', default='libx264', help='Output video codec')
@click.option('--preset', default='medium', help='Encoding preset')
@click.option('--crf', type=int, default=18, help='Constant Rate Factor (lower = better quality)')
@click.option('--nvenc', is_flag=True, help='Use NVIDIA hardware encoding')
@click.option('--temp-dir', type=click.Path(path_type=Path),
              help='Custom temporary directory')
@click.option('--keep-temp', is_flag=True, help='Keep temporary files after processing')
@click.pass_context
def interpolate(ctx, **kwargs):
    """Interpolate video frames to increase frame rate.
    
    Examples:
    
        # Double frame rate (30fps -> 60fps)
        flowforge interpolate video.mp4 -o video_60fps.mp4 --multiplier 2
        
        # Target specific frame rate
        flowforge interpolate video.mp4 -o smooth.mp4 --fps 120
        
        # Use different model and GPU
        flowforge interpolate video.mp4 -o output.mp4 --model rife-v4.15-lite --gpu 1
        
        # Process only part of video
        flowforge interpolate long.mp4 -o clip.mp4 --start-time 60 --end-time 120
    """
    try:
        # Extract parameters
        input_video = kwargs['input_video']
        output_video = kwargs['output_video']
        target_fps = kwargs['fps']
        multiplier = int(kwargs['multiplier']) if kwargs['multiplier'] else None
        
        # Validate parameters
        if target_fps is None and multiplier is None:
            click.echo("Either --fps or --multiplier must be specified", err=True)
            return
        
        if target_fps is not None and multiplier is not None:
            click.echo("Cannot specify both --fps and --multiplier", err=True)
            return
        
        # Create interpolator
        interpolator = VideoInterpolator(
            model_name=kwargs['model'],
            gpu_id=kwargs['gpu'],
            num_threads=kwargs['threads'],
            scene_threshold=kwargs['scene_threshold'],
            temp_dir=kwargs['temp_dir'],
            cleanup_temp=not kwargs['keep_temp']
        )
        
        click.echo(f"🎬 FlowForge v{__version__} - Video Frame Interpolation")
        click.echo(f"📁 Input:  {input_video}")
        click.echo(f"📁 Output: {output_video}")
        click.echo(f"🔧 Model:  {kwargs['model']} (GPU {kwargs['gpu']})")
        
        if target_fps:
            click.echo(f"🎯 Target: {target_fps} FPS")
        else:
            click.echo(f"🎯 Multiplier: {multiplier}x")
        
        # Estimate processing time
        if not ctx.obj['quiet']:
            click.echo("\n⏱️  Estimating processing time...")
            estimate = interpolator.estimate_processing_time(
                input_video, 
                multiplier or 2
            )
            click.echo(f"   Estimated time: {estimate['estimated_total_formatted']}")
            click.echo(f"   Input frames: {estimate['input_frames']:,}")
            click.echo(f"   Output frames: {estimate['output_frames']:,}")
        
        click.echo("\n🚀 Starting interpolation...\n")
        
        # Perform interpolation
        results = interpolator.interpolate_video(
            input_video,
            output_video,
            target_fps=target_fps,
            multiplier=multiplier,
            start_time=kwargs['start_time'],
            end_time=kwargs['end_time'],
            preserve_audio=not kwargs['no_audio'],
            preserve_subtitles=not kwargs['no_subtitles'],
            output_codec=kwargs['codec'],
            output_preset=kwargs['preset'],
            output_crf=kwargs['crf'],
            use_nvenc=kwargs['nvenc'],
            progress_callback=progress_callback if not ctx.obj['quiet'] else None
        )
        
        # Display results
        click.echo("\n✅ Interpolation completed successfully!")
        click.echo(f"⏱️  Processing time: {results['processing']['time_seconds']:.1f}s")
        click.echo(f"📊 Input:  {results['input_info']['fps']:.2f} FPS, {results['input_info']['frame_count']:,} frames")
        click.echo(f"📊 Output: {results['output_info']['fps']:.2f} FPS, {results['output_info']['frame_count']:,} frames")
        click.echo(f"🔄 Interpolated: {results['interpolation']['frames_interpolated']:,} frames")
        click.echo(f"🎬 Scenes: {results['interpolation']['scene_segments']} segments, {results['interpolation']['scene_changes']} changes")
        click.echo(f"📁 Output file: {output_video}")
        
        if ctx.obj['verbose']:
            click.echo(f"🗂️  Temp directory: {results['processing']['temp_dir']}")
    
    except KeyboardInterrupt:
        click.echo("\n🛑 Interpolation cancelled by user", err=True)
        return
    except Exception as e:
        click.echo(f"\n❌ Error: {e}", err=True)
        if ctx.obj['verbose']:
            import traceback
            traceback.print_exc()
        return


@cli.command()
@click.argument('video_file', type=click.Path(exists=True, path_type=Path))
@click.option('--estimate', is_flag=True, help='Include processing time estimates')
@click.pass_context
def info(ctx, video_file: Path, estimate: bool):
    """Show detailed information about a video file.
    
    Examples:
    
        flowforge info video.mp4
        flowforge info video.mp4 --estimate
    """
    try:
        # Create minimal interpolator for info
        interpolator = VideoInterpolator()
        
        click.echo(f"📹 Video Information: {video_file}")
        click.echo("=" * 50)
        
        # Get video information
        info_data = interpolator.get_video_info(video_file)
        
        # Basic information
        click.echo(f"📁 File size:    {info_data['file_size'] / (1024*1024):.1f} MB")
        click.echo(f"⏱️  Duration:     {info_data['duration']:.2f} seconds")
        click.echo(f"🎞️  Frame rate:   {info_data['fps']:.2f} FPS")
        click.echo(f"📊 Frames:       {info_data['frame_count']:,}")
        click.echo(f"📐 Resolution:   {info_data['resolution']['width']}x{info_data['resolution']['height']}")
        click.echo(f"🎥 Video codec:  {info_data['codec']} ({info_data['pixel_format']})")
        
        if info_data['bitrate'] > 0:
            click.echo(f"📈 Bitrate:      {info_data['bitrate'] / 1000:.0f} Kbps")
        
        if info_data['audio_streams'] > 0:
            click.echo(f"🔊 Audio:        {info_data['audio_streams']} stream(s)")
        
        if info_data['subtitle_streams'] > 0:
            click.echo(f"💬 Subtitles:    {info_data['subtitle_streams']} stream(s)")
        
        # Processing estimates
        if estimate:
            click.echo("\n⏱️  Processing Time Estimates")
            click.echo("-" * 30)
            
            for multiplier in [2, 4, 8]:
                est = interpolator.estimate_processing_time(video_file, multiplier)
                output_fps = info_data['fps'] * multiplier
                click.echo(f"{multiplier}x ({output_fps:.1f} FPS): {est['estimated_total_formatted']}")
        
    except Exception as e:
        click.echo(f"❌ Error reading video file: {e}", err=True)
        if ctx.obj['verbose']:
            import traceback
            traceback.print_exc()


@cli.command()
@click.option('--models', multiple=True, type=click.Choice(['rife-v4.6', 'rife-v4.15-lite']),
              help='Models to download (default: rife-v4.6)')
@click.option('--install-dir', type=click.Path(path_type=Path),
              help='Custom installation directory')
@click.option('--force', is_flag=True, help='Force re-download even if files exist')
@click.pass_context
def setup(ctx, models, install_dir: Optional[Path], force: bool):
    """Download and setup FlowForge dependencies.
    
    This downloads:
    - RIFE-NCNN-Vulkan binary for your platform
    - RIFE model files
    
    Examples:
    
        flowforge setup
        flowforge setup --models rife-v4.15-lite
        flowforge setup --install-dir ~/my-flowforge
    """
    try:
        # Default models
        if not models:
            models = ['rife-v4.6']
        
        downloader = ModelDownloader(install_dir)
        
        click.echo(f"🔧 FlowForge Setup v{__version__}")
        click.echo(f"📁 Installation directory: {downloader.install_dir}")
        click.echo(f"💻 Platform: {downloader.platform}")
        click.echo(f"📦 Models to download: {', '.join(models)}")
        click.echo()
        
        # Check current status
        click.echo("📋 Checking current installation...")
        binary_installed = downloader.is_rife_installed()
        click.echo(f"   RIFE binary: {'✅ Installed' if binary_installed else '❌ Missing'}")
        
        for model in models:
            model_installed = downloader.is_model_installed(model)
            click.echo(f"   Model {model}: {'✅ Installed' if model_installed else '❌ Missing'}")
        
        click.echo()
        
        # Download RIFE binary
        if not binary_installed or force:
            click.echo("⬇️  Downloading RIFE binary...")
            try:
                binary_path = downloader.download_rife_binary(force=force)
                click.echo(f"✅ RIFE binary installed: {binary_path}")
            except Exception as e:
                click.echo(f"❌ Failed to download RIFE binary: {e}", err=True)
                return
        else:
            click.echo("✅ RIFE binary already installed")
        
        # Download models
        for model in models:
            if not downloader.is_model_installed(model) or force:
                click.echo(f"⬇️  Downloading model: {model}...")
                try:
                    model_path = downloader.download_model(model, force=force)
                    click.echo(f"✅ Model {model} installed: {model_path}")
                except Exception as e:
                    click.echo(f"❌ Failed to download model {model}: {e}", err=True)
                    continue
            else:
                click.echo(f"✅ Model {model} already installed")
        
        # Test installation
        click.echo("\n🧪 Testing installation...")
        try:
            interpolator = VideoInterpolator(model_name=models[0])
            test_results = interpolator.test_setup()
            
            if test_results['overall_status']:
                click.echo("✅ All tests passed! FlowForge is ready to use.")
            else:
                click.echo("⚠️  Some tests failed:")
                for error in test_results['errors']:
                    click.echo(f"   ❌ {error}")
                
                click.echo("\n💡 Try running setup again or check the installation directory.")
        
        except Exception as e:
            click.echo(f"⚠️  Test failed: {e}")
            if ctx.obj['verbose']:
                import traceback
                traceback.print_exc()
        
        click.echo(f"\n🎉 Setup complete! Installation directory: {downloader.install_dir}")
        
    except Exception as e:
        click.echo(f"❌ Setup failed: {e}", err=True)
        if ctx.obj['verbose']:
            import traceback
            traceback.print_exc()


@cli.command()
@click.pass_context
def test(ctx):
    """Test FlowForge installation and performance.
    
    Runs various tests to ensure everything is working correctly.
    """
    try:
        click.echo("🧪 FlowForge Installation Test")
        click.echo("=" * 35)
        
        # Test with default model
        interpolator = VideoInterpolator()
        test_results = interpolator.test_setup()
        
        # Display test results
        tests = [
            ("FFmpeg", test_results['ffmpeg']),
            ("RIFE Binary", test_results['rife_binary']),
            ("RIFE Model", test_results['rife_model']),
            ("RIFE Interpolation", test_results['rife_interpolation']),
            ("GPU Available", test_results['gpu_available'])
        ]
        
        for test_name, passed in tests:
            status = "✅ PASS" if passed else "❌ FAIL"
            click.echo(f"{test_name:<20} {status}")
        
        # Display errors
        if test_results['errors']:
            click.echo("\n❌ Errors:")
            for error in test_results['errors']:
                click.echo(f"   • {error}")
        
        # Overall status
        click.echo("\n" + "=" * 35)
        if test_results['overall_status']:
            click.echo("✅ All critical tests passed!")
            click.echo("🚀 FlowForge is ready to interpolate videos.")
        else:
            click.echo("❌ Some critical tests failed.")
            click.echo("🔧 Run 'flowforge setup' to install dependencies.")
        
        # GPU information
        try:
            gpu_info = interpolator.rife.get_gpu_info()
            if 'gpu_count' in gpu_info:
                click.echo(f"\n💻 GPU Information:")
                click.echo(f"   Available GPUs: {gpu_info.get('gpu_count', 0)}")
                click.echo(f"   Current GPU: {gpu_info.get('current_gpu', 0)}")
                click.echo(f"   Vulkan Available: {gpu_info.get('vulkan_available', False)}")
        except:
            pass
        
    except Exception as e:
        click.echo(f"❌ Test failed: {e}", err=True)
        if ctx.obj['verbose']:
            import traceback
            traceback.print_exc()


# Phase 2: Real-time playback commands

@cli.command()
@click.argument('video_file', type=click.Path(exists=True, path_type=Path))
@click.option('--preset', default='film', 
              help='Interpolation preset (film, anime, sports, smooth, custom)')
@click.option('--fps', type=float, 
              help='Target frame rate (overrides preset)')
@click.option('--config-dir', type=click.Path(path_type=Path),
              help='Custom mpv config directory')
@click.option('--no-wait', is_flag=True,
              help='Don\'t wait for mpv to exit')
@click.option('--mpv-args', 
              help='Additional mpv arguments (space-separated)')
@click.pass_context
def play(ctx, video_file: Path, preset: str, fps: Optional[float], 
         config_dir: Optional[Path], no_wait: bool, mpv_args: Optional[str]):
    """Play video with real-time FlowForge interpolation.
    
    This launches mpv with VapourSynth integration for real-time RIFE interpolation.
    
    Examples:
    
        # Play with film preset (24→60fps)
        flowforge play movie.mp4 --preset film
        
        # Play with custom target FPS
        flowforge play video.mp4 --preset sports --fps 120
        
        # Play with additional mpv options
        flowforge play video.mp4 --mpv-args "--fullscreen --volume=50"
    """
    try:
        click.echo(f"🎬 FlowForge v{__version__} - Real-time Playback")
        click.echo(f"📁 Video: {video_file}")
        click.echo(f"🎯 Preset: {preset}")
        
        # Create launcher
        launcher = MPVLauncher(config_dir)
        
        # Detect system components
        if not ctx.obj['quiet']:
            click.echo("\n🔍 Detecting system components...")
            detection = launcher.detect_system_components()
            
            # Show detection results
            status_icons = {"✅": True, "❌": False}
            click.echo(f"   mpv: {status_icons[detection['mpv_found']]} {'Found' if detection['mpv_found'] else 'Not found'}")
            click.echo(f"   VapourSynth: {status_icons[detection['vapoursynth_found']]} {'Found' if detection['vapoursynth_found'] else 'Not found'}")
            click.echo(f"   RIFE binary: {status_icons[detection['rife_binary_found']]} {'Found' if detection['rife_binary_found'] else 'Not found'}")
            
            if detection['recommendations']:
                click.echo("\n💡 Recommendations:")
                for rec in detection['recommendations']:
                    click.echo(f"   • {rec}")
        
        # Get preset
        preset_manager = PresetManager()
        try:
            interpolation_preset = preset_manager.get_preset(preset)
            
            # Override FPS if specified
            if fps:
                interpolation_preset.target_fps = fps
                interpolation_preset.multiplier = None  # Let it calculate
            
            click.echo(f"🎯 Target FPS: {interpolation_preset.target_fps}")
            
        except KeyError:
            click.echo(f"❌ Preset '{preset}' not found", err=True)
            click.echo("Available presets:", err=True)
            for p in preset_manager.list_presets():
                click.echo(f"  • {p}", err=True)
            return
        
        # Parse additional mpv arguments
        additional_args = []
        if mpv_args:
            additional_args = mpv_args.split()
        
        click.echo("\n🚀 Launching mpv with FlowForge interpolation...")
        
        # Launch mpv
        mpv_process = launcher.launch_mpv(
            video_file=video_file,
            preset=interpolation_preset,
            additional_args=additional_args,
            wait_for_exit=not no_wait
        )
        
        if no_wait:
            click.echo(f"✅ mpv launched successfully (PID: {mpv_process.pid})")
            click.echo("Use 'flowforge stop' to stop playback")
        else:
            click.echo("✅ Playback completed")
        
    except FileNotFoundError as e:
        click.echo(f"❌ File not found: {e}", err=True)
    except RuntimeError as e:
        click.echo(f"❌ Runtime error: {e}", err=True)
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        if ctx.obj['verbose']:
            import traceback
            traceback.print_exc()


@cli.command('configure-mpv')
@click.option('--preset', default='film',
              help='Preset to configure for')
@click.option('--config-dir', type=click.Path(path_type=Path),
              help='Custom mpv config directory')
@click.option('--force', is_flag=True,
              help='Force reconfiguration even if files exist')
@click.option('--portable', type=click.Path(path_type=Path),
              help='Create portable configuration in specified directory')
@click.pass_context
def configure_mpv(ctx, preset: str, config_dir: Optional[Path], 
                  force: bool, portable: Optional[Path]):
    """Configure mpv for FlowForge real-time interpolation.
    
    This creates mpv configuration files with VapourSynth integration.
    
    Examples:
    
        # Configure mpv with film preset
        flowforge configure-mpv --preset film
        
        # Force reconfiguration  
        flowforge configure-mpv --preset anime --force
        
        # Create portable config directory
        flowforge configure-mpv --portable ./my-config
    """
    try:
        click.echo(f"🔧 FlowForge mpv Configuration")
        
        # Create launcher
        launcher = MPVLauncher(config_dir)
        
        # System detection
        click.echo("🔍 Detecting system components...")
        detection = launcher.detect_system_components()
        
        if not detection['mpv_found']:
            click.echo("❌ mpv not found. Please install mpv first.", err=True)
            return
        
        # Get preset
        preset_manager = PresetManager()
        try:
            interpolation_preset = preset_manager.get_preset(preset)
            click.echo(f"📋 Using preset: {preset} ({interpolation_preset.description})")
        except KeyError:
            click.echo(f"❌ Preset '{preset}' not found", err=True)
            return
        
        # Configure
        if portable:
            click.echo(f"📦 Creating portable configuration in: {portable}")
            config_files = launcher.create_portable_config(portable)
        else:
            click.echo(f"⚙️ Configuring mpv...")
            config_files = launcher.configure_mpv(interpolation_preset, force)
        
        # Show created files
        click.echo("\n✅ Configuration files created:")
        for config_type, file_path in config_files.items():
            click.echo(f"   {config_type}: {file_path}")
        
        # Show usage instructions
        config_dir_path = portable or launcher.config_dir
        click.echo(f"\n📖 Usage:")
        click.echo(f"   mpv --config-dir=\"{config_dir_path}\" your_video.mp4")
        click.echo(f"   Or use: flowforge play your_video.mp4 --preset {preset}")
        
    except Exception as e:
        click.echo(f"❌ Configuration failed: {e}", err=True)
        if ctx.obj['verbose']:
            import traceback
            traceback.print_exc()


@cli.group('presets')
def presets():
    """Manage interpolation presets."""
    pass


@presets.command('list')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed preset information')
def list_presets(verbose: bool):
    """List available interpolation presets.
    
    Examples:
    
        flowforge presets list
        flowforge presets list --verbose
    """
    try:
        preset_manager = PresetManager()
        preset_names = preset_manager.list_presets()
        
        if not preset_names:
            click.echo("No presets found")
            return
        
        click.echo("📋 Available Interpolation Presets:")
        click.echo("=" * 40)
        
        for name in preset_names:
            try:
                preset = preset_manager.get_preset(name)
                info = preset_manager.get_preset_info(name)
                
                status = "built-in" if info["is_builtin"] else "custom"
                
                if verbose:
                    click.echo(f"\n🎯 {name} ({status})")
                    click.echo(f"   Description: {preset.description}")
                    click.echo(f"   Target FPS: {preset.target_fps}")
                    click.echo(f"   Model: {preset.model}")
                    click.echo(f"   Quality: {preset.quality_profile}")
                    click.echo(f"   Scene detection: {preset.scene_detection}")
                    if preset.multiplier:
                        click.echo(f"   Multiplier: {preset.multiplier}x")
                else:
                    click.echo(f"  • {name:<12} {preset.target_fps:>6.0f} FPS  ({status}) - {preset.description}")
                    
            except Exception as e:
                click.echo(f"  • {name:<12} ERROR: {e}")
        
        # Show system recommendations
        if verbose:
            click.echo("\n💻 System Recommendations:")
            recommendations = get_system_recommendations()
            for preset_name in recommendations['recommended_presets']:
                click.echo(f"   ✅ {preset_name}")
            
            if recommendations['performance_tips']:
                click.echo("\n💡 Performance Tips:")
                for tip in recommendations['performance_tips']:
                    click.echo(f"   • {tip}")
        
    except Exception as e:
        click.echo(f"❌ Error listing presets: {e}", err=True)


@presets.command('show')
@click.argument('preset_name')
def show_preset(preset_name: str):
    """Show detailed information about a preset.
    
    Examples:
    
        flowforge presets show film
        flowforge presets show anime
    """
    try:
        preset_manager = PresetManager()
        preset = preset_manager.get_preset(preset_name)
        info = preset_manager.get_preset_info(preset_name)
        
        click.echo(f"🎯 Preset: {preset.name}")
        click.echo("=" * 50)
        click.echo(f"Description:      {preset.description}")
        click.echo(f"Type:             {'Built-in' if info['is_builtin'] else 'Custom'}")
        click.echo(f"Target FPS:       {preset.target_fps}")
        click.echo(f"Multiplier:       {preset.multiplier or 'Auto'}")
        click.echo(f"Model:            {preset.model}")
        click.echo(f"Quality Profile:  {preset.quality_profile}")
        click.echo(f"TTA:              {preset.tta}")
        click.echo(f"UHD Mode:         {preset.uhd}")
        click.echo(f"Tile Size:        {preset.tile_size}x{preset.tile_size}")
        click.echo(f"Scene Detection:  {preset.scene_detection}")
        if preset.scene_detection:
            click.echo(f"Scene Threshold:  {preset.scene_threshold}")
            click.echo(f"Scene Method:     {preset.scene_method}")
        click.echo(f"Buffer Frames:    {preset.buffer_frames}")
        click.echo(f"Max Queue Size:   {preset.max_queue_size}")
        click.echo(f"Drop Threshold:   {preset.drop_threshold}")
        
        if not info['is_builtin'] and info['file_path']:
            click.echo(f"File Path:        {info['file_path']}")
        
    except KeyError:
        click.echo(f"❌ Preset '{preset_name}' not found", err=True)
        available = PresetManager().list_presets()
        click.echo(f"Available presets: {', '.join(available)}", err=True)
    except Exception as e:
        click.echo(f"❌ Error showing preset: {e}", err=True)


@presets.command('create')
@click.argument('preset_name')
@click.option('--base-preset', default='film', 
              help='Base preset to copy from')
@click.option('--fps', type=float,
              help='Target frame rate')
@click.option('--description',
              help='Preset description')
@click.option('--model', 
              type=click.Choice(['rife-v4.6', 'rife-v4.15-lite']),
              help='RIFE model to use')
@click.option('--quality', 
              type=click.Choice(['fast', 'balanced', 'quality']),
              help='Quality profile')
@click.option('--no-scene-detection', is_flag=True,
              help='Disable scene detection')
@click.option('--overwrite', is_flag=True,
              help='Overwrite existing preset')
def create_preset(preset_name: str, base_preset: str, fps: Optional[float],
                  description: Optional[str], model: Optional[str], 
                  quality: Optional[str], no_scene_detection: bool, overwrite: bool):
    """Create a custom interpolation preset.
    
    Examples:
    
        # Create preset based on film
        flowforge presets create my-preset --base-preset film --fps 75
        
        # Create fast gaming preset
        flowforge presets create gaming --fps 144 --quality fast --no-scene-detection
    """
    try:
        preset_manager = PresetManager()
        
        # Get base preset
        base = preset_manager.get_preset(base_preset)
        
        # Create new preset with modifications
        new_preset = InterpolationPreset(
            name=preset_name,
            description=description or f"Custom preset based on {base_preset}",
            target_fps=fps or base.target_fps,
            multiplier=base.multiplier,
            model=model or base.model,
            tta=base.tta,
            uhd=base.uhd,
            quality_profile=quality or base.quality_profile,
            scene_detection=not no_scene_detection and base.scene_detection,
            scene_threshold=base.scene_threshold,
            scene_method=base.scene_method,
            buffer_frames=base.buffer_frames,
            max_queue_size=base.max_queue_size,
            drop_threshold=base.drop_threshold,
            tile_size=base.tile_size,
            tile_pad=base.tile_pad
        )
        
        # Save preset
        preset_manager.save_preset(new_preset, overwrite=overwrite)
        
        click.echo(f"✅ Created preset: {preset_name}")
        click.echo(f"   Target FPS: {new_preset.target_fps}")
        click.echo(f"   Model: {new_preset.model}")
        click.echo(f"   Quality: {new_preset.quality_profile}")
        
    except KeyError:
        click.echo(f"❌ Base preset '{base_preset}' not found", err=True)
    except FileExistsError:
        click.echo(f"❌ Preset '{preset_name}' already exists. Use --overwrite to replace.", err=True)
    except Exception as e:
        click.echo(f"❌ Error creating preset: {e}", err=True)


@presets.command('delete')
@click.argument('preset_name')
@click.option('--yes', is_flag=True, help='Skip confirmation')
def delete_preset(preset_name: str, yes: bool):
    """Delete a custom preset.
    
    Examples:
    
        flowforge presets delete my-preset
        flowforge presets delete old-preset --yes
    """
    try:
        preset_manager = PresetManager()
        
        # Check if preset exists and is custom
        info = preset_manager.get_preset_info(preset_name)
        if info['is_builtin']:
            click.echo(f"❌ Cannot delete built-in preset '{preset_name}'", err=True)
            return
        
        # Confirm deletion
        if not yes:
            if not click.confirm(f"Delete preset '{preset_name}'?"):
                click.echo("❌ Cancelled")
                return
        
        # Delete preset
        preset_manager.delete_preset(preset_name)
        click.echo(f"✅ Deleted preset: {preset_name}")
        
    except KeyError:
        click.echo(f"❌ Preset '{preset_name}' not found", err=True)
    except ValueError as e:
        click.echo(f"❌ {e}", err=True)
    except Exception as e:
        click.echo(f"❌ Error deleting preset: {e}", err=True)


@cli.command('system-status')
@click.option('--json-output', is_flag=True, help='Output in JSON format')
def system_status(json_output: bool):
    """Show FlowForge system status and recommendations.
    
    Examples:
    
        flowforge system-status
        flowforge system-status --json-output
    """
    try:
        launcher = MPVLauncher()
        status = launcher.get_system_status()
        
        if json_output:
            import json
            click.echo(json.dumps(status, indent=2))
            return
        
        click.echo("🖥️  FlowForge System Status")
        click.echo("=" * 50)
        
        # System info
        sys_info = status['system_info']
        click.echo(f"Platform:         {sys_info['platform']}")
        if sys_info.get('is_wsl'):
            click.echo("Environment:      WSL2")
        click.echo(f"Python:           {sys_info['python_version']}")
        
        # Component status
        click.echo("\n🔧 Components:")
        components = status['components']
        status_icon = lambda x: "✅" if x else "❌"
        
        click.echo(f"  mpv:              {status_icon(components['mpv_found'])} {components.get('mpv_path', 'Not found')}")
        click.echo(f"  VapourSynth:      {status_icon(components['vapoursynth_found'])} {components.get('vapoursynth_path', 'Not found')}")
        click.echo(f"  vs-rife plugin:   {status_icon(components['vs_rife_plugin'])}")
        click.echo(f"  RIFE binary:      {status_icon(components['rife_binary_found'])} {components.get('rife_binary_path', 'Not found')}")
        
        # System capabilities
        caps = components['system_capabilities']
        click.echo(f"\n💻 Hardware:")
        click.echo(f"  CPU cores:        {caps.get('cpu_cores', 'Unknown')}")
        click.echo(f"  RAM:              {caps.get('ram_gb', 0):.1f} GB")
        click.echo(f"  GPU available:    {status_icon(caps.get('gpu_available', False))}")
        if caps.get('gpu_available'):
            click.echo(f"  GPU memory:       {caps.get('gpu_memory_gb', 0):.1f} GB")
        click.echo(f"  Vulkan:           {status_icon(caps.get('vulkan_available', False))}")
        
        # Current preset
        if status['current_preset']:
            preset = status['current_preset']
            click.echo(f"\n🎯 Active Preset:")
            click.echo(f"  Name:             {preset['name']}")
            click.echo(f"  Target FPS:       {preset['target_fps']}")
            click.echo(f"  Model:            {preset['model']}")
        
        # mpv status
        mpv_status = status['mpv_status']
        click.echo(f"\n📺 mpv Status:")
        click.echo(f"  Running:          {status_icon(mpv_status['running'])}")
        if mpv_status['pid']:
            click.echo(f"  PID:              {mpv_status['pid']}")
        
        # Recommendations
        if components['recommendations']:
            click.echo(f"\n💡 Recommendations:")
            for rec in components['recommendations']:
                click.echo(f"  • {rec}")
        
        click.echo(f"\nConfig directory:   {status['config_directory']}")
        click.echo(f"Last check:         {status['last_check']}")
        
    except Exception as e:
        click.echo(f"❌ Error getting system status: {e}", err=True)


def main():
    """Main entry point for the CLI."""
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\n🛑 Interrupted", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()