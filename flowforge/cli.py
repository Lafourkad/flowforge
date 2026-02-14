"""FlowForge command-line interface."""

import logging
import sys
from pathlib import Path
from typing import Optional

import click
from tqdm import tqdm

from . import __version__
from .core.interpolator import VideoInterpolator
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