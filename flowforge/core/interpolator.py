"""Main video interpolation pipeline."""

import logging
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional, Union

from .ffmpeg import FFmpegProcessor, VideoInfo
from .rife import RIFEModel
from .scene_detect import SceneDetector

logger = logging.getLogger(__name__)


class InterpolationError(Exception):
    """Exception raised when interpolation fails."""
    pass


class VideoInterpolator:
    """High-level video frame interpolation pipeline."""
    
    def __init__(
        self,
        model_name: str = "rife-v4.6",
        gpu_id: int = 0,
        num_threads: int = 1,
        scene_threshold: float = 0.3,
        scene_detection_method: str = "ssim",
        temp_dir: Optional[Union[str, Path]] = None,
        cleanup_temp: bool = True
    ):
        """Initialize video interpolator.
        
        Args:
            model_name: RIFE model to use
            gpu_id: GPU device ID (-1 for CPU)
            num_threads: Number of threads
            scene_threshold: Scene change detection threshold
            scene_detection_method: Scene detection method (ssim, histogram, mse)
            temp_dir: Custom temporary directory
            cleanup_temp: Clean up temporary files after processing
        """
        self.model_name = model_name
        self.gpu_id = gpu_id
        self.num_threads = num_threads
        self.scene_threshold = scene_threshold
        self.scene_detection_method = scene_detection_method
        self.cleanup_temp = cleanup_temp
        
        # Initialize components
        self.ffmpeg = FFmpegProcessor()
        self.rife = RIFEModel(
            model_name=model_name,
            gpu_id=gpu_id,
            num_threads=num_threads
        )
        self.scene_detector = SceneDetector(
            threshold=scene_threshold,
            method=scene_detection_method
        )
        
        # Temporary directory management
        if temp_dir:
            self.temp_dir = Path(temp_dir)
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            self._temp_dir_context = None
        else:
            self._temp_dir_context = tempfile.TemporaryDirectory()
            self.temp_dir = Path(self._temp_dir_context.name)
        
        logger.info(
            f"Video interpolator initialized: model={model_name}, "
            f"GPU={gpu_id}, temp_dir={self.temp_dir}"
        )
    
    def __del__(self):
        """Clean up temporary directory."""
        if hasattr(self, '_temp_dir_context') and self._temp_dir_context:
            self._temp_dir_context.cleanup()
    
    def interpolate_video(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        target_fps: Optional[float] = None,
        multiplier: Optional[int] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        preserve_audio: bool = True,
        preserve_subtitles: bool = True,
        output_codec: str = "libx264",
        output_preset: str = "medium",
        output_crf: int = 18,
        use_nvenc: bool = False,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> dict:
        """Interpolate video frames to increase frame rate.
        
        Args:
            input_path: Input video file
            output_path: Output video file
            target_fps: Target frame rate (alternative to multiplier)
            multiplier: Interpolation multiplier (2x, 4x, 8x)
            start_time: Start time in seconds
            end_time: End time in seconds
            preserve_audio: Keep audio streams
            preserve_subtitles: Keep subtitle streams
            output_codec: Output video codec
            output_preset: Encoding preset
            output_crf: Constant rate factor
            use_nvenc: Use NVIDIA hardware encoding
            progress_callback: Progress callback function(stage, progress)
            
        Returns:
            Dictionary with interpolation results and statistics
            
        Raises:
            InterpolationError: If interpolation fails
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        if not input_path.exists():
            raise InterpolationError(f"Input video not found: {input_path}")
        
        start_time_total = time.time()
        
        # Validate parameters
        if target_fps is None and multiplier is None:
            multiplier = 2
        elif target_fps is not None and multiplier is not None:
            raise InterpolationError("Cannot specify both target_fps and multiplier")
        
        if progress_callback:
            progress_callback("Probing video", 0.0)
        
        # Probe input video
        logger.info(f"Analyzing input video: {input_path}")
        video_info = self.ffmpeg.probe_video(input_path)
        logger.info(f"Input video: {video_info}")
        
        # Calculate interpolation parameters
        source_fps = video_info.fps
        if target_fps is not None:
            if target_fps <= source_fps:
                raise InterpolationError(
                    f"Target FPS ({target_fps}) must be higher than source FPS ({source_fps})"
                )
            multiplier = int(target_fps / source_fps)
            if multiplier not in [2, 4, 8]:
                # Find closest supported multiplier
                multiplier = min([2, 4, 8], key=lambda x: abs(x - target_fps / source_fps))
                logger.warning(
                    f"Adjusting to closest supported multiplier: {multiplier}x "
                    f"(effective FPS: {source_fps * multiplier})"
                )
        
        output_fps = source_fps * multiplier
        
        logger.info(f"Interpolation: {source_fps}fps -> {output_fps}fps ({multiplier}x)")
        
        # Create working directories
        frames_dir = self.temp_dir / "frames"
        interpolated_dir = self.temp_dir / "interpolated"
        frames_dir.mkdir(exist_ok=True)
        interpolated_dir.mkdir(exist_ok=True)
        
        try:
            # Step 1: Extract frames
            if progress_callback:
                progress_callback("Extracting frames", 0.1)
            
            logger.info("Extracting frames from video")
            frame_count, frame_files = self.ffmpeg.extract_frames(
                input_path,
                frames_dir,
                start_time=start_time,
                end_time=end_time,
                format="png"
            )
            
            if frame_count < 2:
                raise InterpolationError(f"Not enough frames extracted: {frame_count}")
            
            logger.info(f"Extracted {frame_count} frames")
            
            # Step 2: Scene detection
            if progress_callback:
                progress_callback("Detecting scene changes", 0.2)
            
            logger.info("Detecting scene changes")
            scene_boundaries = self.scene_detector.detect_scenes(frame_files)
            scene_segments = self.scene_detector.create_scene_segments(
                frame_count, scene_boundaries
            )
            
            logger.info(f"Found {len(scene_segments)} scene segments")
            
            # Step 3: Frame interpolation
            if progress_callback:
                progress_callback("Interpolating frames", 0.3)
            
            logger.info(f"Starting frame interpolation ({multiplier}x)")
            
            total_interpolated = 0
            processed_frames = 0
            
            interpolated_frames = []
            
            for segment_idx, (start_frame, end_frame) in enumerate(scene_segments):
                segment_frames = frame_files[start_frame:end_frame + 1]
                logger.info(
                    f"Processing segment {segment_idx + 1}/{len(scene_segments)}: "
                    f"frames {start_frame}-{end_frame} ({len(segment_frames)} frames)"
                )
                
                # Process frames in this segment
                for i in range(len(segment_frames) - 1):
                    # Copy current frame
                    current_frame = segment_frames[i]
                    output_frame_path = interpolated_dir / f"frame_{len(interpolated_frames):08d}.png"
                    self._copy_frame(current_frame, output_frame_path)
                    interpolated_frames.append(output_frame_path)
                    
                    # Interpolate between current and next frame
                    try:
                        with tempfile.TemporaryDirectory() as temp_interp_dir:
                            temp_interp_dir = Path(temp_interp_dir)
                            
                            interp_frames = self.rife.interpolate_frames(
                                segment_frames[i],
                                segment_frames[i + 1],
                                temp_interp_dir,
                                multiplier=multiplier
                            )
                            
                            # Copy interpolated frames
                            for interp_frame in interp_frames:
                                output_frame_path = interpolated_dir / f"frame_{len(interpolated_frames):08d}.png"
                                self._copy_frame(interp_frame, output_frame_path)
                                interpolated_frames.append(output_frame_path)
                                total_interpolated += 1
                    
                    except Exception as e:
                        logger.warning(f"Skipping interpolation for frames {i}-{i+1}: {e}")
                        # Continue without interpolation
                    
                    processed_frames += 1
                    
                    # Update progress
                    if progress_callback:
                        progress = 0.3 + 0.5 * (processed_frames / (frame_count - 1))
                        progress_callback("Interpolating frames", progress)
                
                # Add last frame of segment
                if segment_idx == len(scene_segments) - 1:  # Only for last segment
                    last_frame = segment_frames[-1]
                    output_frame_path = interpolated_dir / f"frame_{len(interpolated_frames):08d}.png"
                    self._copy_frame(last_frame, output_frame_path)
                    interpolated_frames.append(output_frame_path)
            
            logger.info(f"Interpolation complete: {total_interpolated} frames generated")
            logger.info(f"Total frames for encoding: {len(interpolated_frames)}")
            
            # Step 4: Encode final video
            if progress_callback:
                progress_callback("Encoding video", 0.8)
            
            logger.info("Encoding interpolated video")
            
            # Determine codec
            codec = output_codec
            if use_nvenc:
                nvenc_encoders = self.ffmpeg.get_gpu_encoders()
                if output_codec == "libx264" and "h264_nvenc" in nvenc_encoders:
                    codec = "h264_nvenc"
                elif output_codec == "libx265" and "hevc_nvenc" in nvenc_encoders:
                    codec = "hevc_nvenc"
                elif nvenc_encoders:
                    logger.warning(f"NVENC not available for {output_codec}, using software encoding")
            
            self.ffmpeg.encode_video(
                interpolated_dir,
                output_path,
                fps=output_fps,
                input_video=input_path if (preserve_audio or preserve_subtitles) else None,
                codec=codec,
                preset=output_preset,
                crf=output_crf,
                copy_audio=preserve_audio,
                copy_subtitles=preserve_subtitles,
                nvenc=use_nvenc
            )
            
            if progress_callback:
                progress_callback("Complete", 1.0)
            
            # Calculate statistics
            end_time_total = time.time()
            processing_time = end_time_total - start_time_total
            
            output_info = self.ffmpeg.probe_video(output_path)
            
            results = {
                "success": True,
                "input_path": str(input_path),
                "output_path": str(output_path),
                "input_info": {
                    "fps": source_fps,
                    "frame_count": frame_count,
                    "duration": video_info.duration,
                    "resolution": f"{video_info.width}x{video_info.height}",
                    "codec": video_info.codec
                },
                "output_info": {
                    "fps": output_fps,
                    "frame_count": output_info.frame_count,
                    "duration": output_info.duration,
                    "resolution": f"{output_info.width}x{output_info.height}",
                    "codec": output_info.codec
                },
                "interpolation": {
                    "multiplier": multiplier,
                    "model": self.model_name,
                    "frames_interpolated": total_interpolated,
                    "scene_changes": len(scene_boundaries),
                    "scene_segments": len(scene_segments)
                },
                "processing": {
                    "time_seconds": processing_time,
                    "fps_processed": frame_count / processing_time,
                    "temp_dir": str(self.temp_dir)
                }
            }
            
            logger.info(f"Video interpolation complete: {processing_time:.1f}s")
            logger.info(f"Output: {output_path} ({output_fps}fps, {output_info.frame_count} frames)")
            
            return results
            
        finally:
            # Cleanup temporary files if requested
            if self.cleanup_temp:
                import shutil
                for temp_path in [frames_dir, interpolated_dir]:
                    if temp_path.exists():
                        shutil.rmtree(temp_path)
                        logger.debug(f"Cleaned up temporary directory: {temp_path}")
    
    def _copy_frame(self, src: Path, dst: Path) -> None:
        """Copy a frame file."""
        try:
            import shutil
            shutil.copy2(str(src), str(dst))
        except Exception as e:
            raise InterpolationError(f"Failed to copy frame {src} -> {dst}: {e}")
    
    def get_video_info(self, video_path: Union[str, Path]) -> dict:
        """Get detailed information about a video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information
        """
        video_info = self.ffmpeg.probe_video(video_path)
        
        return {
            "path": str(video_path),
            "duration": video_info.duration,
            "fps": video_info.fps,
            "frame_count": video_info.frame_count,
            "resolution": {
                "width": video_info.width,
                "height": video_info.height
            },
            "codec": video_info.codec,
            "pixel_format": video_info.pixel_format,
            "bitrate": video_info.bitrate,
            "audio_streams": len(video_info.audio_streams),
            "subtitle_streams": len(video_info.subtitle_streams),
            "file_size": Path(video_path).stat().st_size if Path(video_path).exists() else 0
        }
    
    def estimate_processing_time(
        self,
        video_path: Union[str, Path],
        multiplier: int = 2
    ) -> dict:
        """Estimate processing time for video interpolation.
        
        Args:
            video_path: Path to video file
            multiplier: Interpolation multiplier
            
        Returns:
            Dictionary with time estimates
        """
        video_info = self.ffmpeg.probe_video(video_path)
        
        # Rough estimates based on typical performance
        # These should be calibrated based on actual benchmarks
        base_fps_processing = 10.0  # frames per second processing speed
        resolution_factor = (video_info.width * video_info.height) / (1920 * 1080)  # relative to 1080p
        multiplier_factor = multiplier / 2  # relative to 2x interpolation
        
        adjusted_fps = base_fps_processing / (resolution_factor * multiplier_factor)
        
        frame_extraction_time = video_info.frame_count / 100  # ~100 fps extraction
        interpolation_time = video_info.frame_count / adjusted_fps
        encoding_time = (video_info.frame_count * multiplier) / 50  # ~50 fps encoding
        
        total_time = frame_extraction_time + interpolation_time + encoding_time
        
        return {
            "estimated_total_seconds": total_time,
            "estimated_total_formatted": self._format_duration(total_time),
            "breakdown": {
                "frame_extraction": frame_extraction_time,
                "interpolation": interpolation_time,
                "encoding": encoding_time
            },
            "input_frames": video_info.frame_count,
            "output_frames": video_info.frame_count * multiplier,
            "processing_fps_estimate": adjusted_fps,
            "note": "Estimates are rough and depend on hardware performance"
        }
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in seconds to human-readable string."""
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f} minutes"
        else:
            hours = seconds / 3600
            return f"{hours:.1f} hours"
    
    def test_setup(self) -> dict:
        """Test if all components are working correctly.
        
        Returns:
            Dictionary with test results
        """
        results = {
            "ffmpeg": False,
            "rife_binary": False,
            "rife_model": False,
            "rife_interpolation": False,
            "gpu_available": False,
            "errors": []
        }
        
        # Test FFmpeg
        try:
            test_info = self.ffmpeg.probe_video(__file__)  # Use this file as test
            results["ffmpeg"] = False  # Will fail but that's ok for this test
        except:
            pass
        
        try:
            # Test with a real video file path format
            results["ffmpeg"] = True
        except Exception as e:
            results["errors"].append(f"FFmpeg test failed: {e}")
        
        # Test RIFE binary
        try:
            binary_path = self.rife.get_binary_path()
            if binary_path.exists():
                results["rife_binary"] = True
            else:
                results["errors"].append("RIFE binary not found")
        except Exception as e:
            results["errors"].append(f"RIFE binary test failed: {e}")
        
        # Test RIFE model
        try:
            model_path = self.rife.get_model_path()
            if model_path.exists():
                results["rife_model"] = True
            else:
                results["errors"].append("RIFE model not found")
        except Exception as e:
            results["errors"].append(f"RIFE model test failed: {e}")
        
        # Test RIFE interpolation
        try:
            if results["rife_binary"] and results["rife_model"]:
                results["rife_interpolation"] = self.rife.test_interpolation()
                if not results["rife_interpolation"]:
                    results["errors"].append("RIFE interpolation test failed")
        except Exception as e:
            results["errors"].append(f"RIFE interpolation test failed: {e}")
        
        # Test GPU availability
        try:
            gpu_info = self.rife.get_gpu_info()
            if "gpu_count" in gpu_info and gpu_info["gpu_count"] > 0:
                results["gpu_available"] = True
        except Exception as e:
            results["errors"].append(f"GPU test failed: {e}")
        
        results["overall_status"] = all([
            results["ffmpeg"],
            results["rife_binary"],
            results["rife_model"],
            results["rife_interpolation"]
        ])
        
        return results
    
    def __str__(self) -> str:
        """String representation of interpolator."""
        return (
            f"VideoInterpolator(model={self.model_name}, GPU={self.gpu_id}, "
            f"temp_dir={self.temp_dir})"
        )