"""FFmpeg utilities for video processing."""

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class FFmpegError(Exception):
    """Exception raised when FFmpeg operations fail."""
    pass


class VideoInfo:
    """Container for video information."""
    
    def __init__(self, data: Dict):
        """Initialize video info from ffprobe output."""
        self.data = data
        self._video_stream = None
        self._audio_streams = None
        self._subtitle_streams = None
        
        for stream in data.get('streams', []):
            if stream.get('codec_type') == 'video' and self._video_stream is None:
                self._video_stream = stream
    
    @property
    def width(self) -> int:
        """Video width in pixels."""
        return int(self._video_stream.get('width', 0)) if self._video_stream else 0
    
    @property
    def height(self) -> int:
        """Video height in pixels."""
        return int(self._video_stream.get('height', 0)) if self._video_stream else 0
    
    @property
    def fps(self) -> float:
        """Video frame rate."""
        if not self._video_stream:
            return 0.0
        
        fps_str = self._video_stream.get('avg_frame_rate', '0/1')
        try:
            num, den = map(int, fps_str.split('/'))
            return num / den if den != 0 else 0.0
        except (ValueError, ZeroDivisionError):
            return 0.0
    
    @property
    def frame_count(self) -> int:
        """Total number of frames."""
        if not self._video_stream:
            return 0
        
        nb_frames = self._video_stream.get('nb_frames')
        if nb_frames:
            return int(nb_frames)
        
        # Fallback: calculate from duration and fps
        duration = self.duration
        if duration > 0 and self.fps > 0:
            return int(duration * self.fps)
        
        return 0
    
    @property
    def duration(self) -> float:
        """Video duration in seconds."""
        if not self._video_stream:
            return 0.0
        
        duration = self._video_stream.get('duration')
        if duration:
            return float(duration)
        
        # Try format duration
        format_info = self.data.get('format', {})
        duration = format_info.get('duration')
        return float(duration) if duration else 0.0
    
    @property
    def codec(self) -> str:
        """Video codec name."""
        return self._video_stream.get('codec_name', 'unknown') if self._video_stream else 'unknown'
    
    @property
    def pixel_format(self) -> str:
        """Video pixel format."""
        return self._video_stream.get('pix_fmt', 'unknown') if self._video_stream else 'unknown'
    
    @property
    def bitrate(self) -> int:
        """Video bitrate in bits/second."""
        if not self._video_stream:
            return 0
        
        bitrate = self._video_stream.get('bit_rate')
        if bitrate:
            return int(bitrate)
        
        # Try format bitrate
        format_info = self.data.get('format', {})
        bitrate = format_info.get('bit_rate')
        return int(bitrate) if bitrate else 0
    
    @property
    def audio_streams(self) -> List[Dict]:
        """List of audio stream information."""
        if self._audio_streams is None:
            self._audio_streams = [
                stream for stream in self.data.get('streams', [])
                if stream.get('codec_type') == 'audio'
            ]
        return self._audio_streams
    
    @property
    def subtitle_streams(self) -> List[Dict]:
        """List of subtitle stream information."""
        if self._subtitle_streams is None:
            self._subtitle_streams = [
                stream for stream in self.data.get('streams', [])
                if stream.get('codec_type') == 'subtitle'
            ]
        return self._subtitle_streams
    
    def __str__(self) -> str:
        """String representation of video info."""
        return (
            f"VideoInfo({self.width}x{self.height}, {self.fps:.2f}fps, "
            f"{self.duration:.1f}s, {self.codec})"
        )


class FFmpegProcessor:
    """FFmpeg wrapper for video processing operations."""
    
    def __init__(self, ffmpeg_path: str = "ffmpeg", ffprobe_path: str = "ffprobe"):
        """Initialize FFmpeg processor.
        
        Args:
            ffmpeg_path: Path to ffmpeg binary
            ffprobe_path: Path to ffprobe binary
        """
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path
        self._check_dependencies()
    
    def _check_dependencies(self) -> None:
        """Check if FFmpeg binaries are available."""
        if not shutil.which(self.ffmpeg_path):
            raise FFmpegError(f"FFmpeg binary not found: {self.ffmpeg_path}")
        
        if not shutil.which(self.ffprobe_path):
            raise FFmpegError(f"FFprobe binary not found: {self.ffprobe_path}")
        
        logger.info(f"FFmpeg found at: {shutil.which(self.ffmpeg_path)}")
        logger.info(f"FFprobe found at: {shutil.which(self.ffprobe_path)}")
    
    def probe_video(self, video_path: Union[str, Path]) -> VideoInfo:
        """Get video information using ffprobe.
        
        Args:
            video_path: Path to video file
            
        Returns:
            VideoInfo object with video metadata
            
        Raises:
            FFmpegError: If probing fails
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FFmpegError(f"Video file not found: {video_path}")
        
        cmd = [
            self.ffprobe_path,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(video_path)
        ]
        
        logger.debug(f"Running command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=30
            )
            
            data = json.loads(result.stdout)
            return VideoInfo(data)
            
        except subprocess.CalledProcessError as e:
            logger.error(f"FFprobe failed: {e.stderr}")
            raise FFmpegError(f"Failed to probe video: {e.stderr}")
        except subprocess.TimeoutExpired:
            raise FFmpegError("FFprobe timed out")
        except json.JSONDecodeError as e:
            raise FFmpegError(f"Invalid JSON from ffprobe: {e}")
    
    def extract_frames(
        self,
        video_path: Union[str, Path],
        output_dir: Union[str, Path],
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        fps: Optional[float] = None,
        format: str = "png"
    ) -> Tuple[int, List[Path]]:
        """Extract frames from video.
        
        Args:
            video_path: Input video file
            output_dir: Directory to save frames
            start_time: Start time in seconds (optional)
            end_time: End time in seconds (optional)
            fps: Target FPS for extraction (optional)
            format: Output format (png, jpg)
            
        Returns:
            Tuple of (frame_count, list_of_frame_paths)
            
        Raises:
            FFmpegError: If extraction fails
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build output pattern
        output_pattern = output_dir / f"frame_%08d.{format}"
        
        cmd = [self.ffmpeg_path, "-y", "-i", str(video_path)]
        
        # Add time range if specified
        if start_time is not None:
            cmd.extend(["-ss", str(start_time)])
        if end_time is not None:
            cmd.extend(["-to", str(end_time)])
        
        # Add FPS filter if specified
        if fps is not None:
            cmd.extend(["-r", str(fps)])
        
        # Output options
        cmd.extend([
            "-q:v", "1",  # Best quality for PNG/JPG
            str(output_pattern)
        ])
        
        logger.info(f"Extracting frames to {output_dir}")
        logger.debug(f"Running command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Count extracted frames
            frame_files = sorted(output_dir.glob(f"frame_*.{format}"))
            frame_count = len(frame_files)
            
            logger.info(f"Extracted {frame_count} frames")
            return frame_count, frame_files
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Frame extraction failed: {e.stderr}")
            raise FFmpegError(f"Failed to extract frames: {e.stderr}")
    
    def encode_video(
        self,
        frame_dir: Union[str, Path],
        output_path: Union[str, Path],
        fps: float,
        input_video: Optional[Union[str, Path]] = None,
        codec: str = "libx264",
        preset: str = "medium",
        crf: int = 18,
        pixel_format: str = "yuv420p",
        copy_audio: bool = True,
        copy_subtitles: bool = True,
        nvenc: bool = False
    ) -> None:
        """Encode frames back to video.
        
        Args:
            frame_dir: Directory containing frames
            output_path: Output video path
            fps: Output frame rate
            input_video: Original video for audio/subtitle copying
            codec: Video codec (libx264, libx265)
            preset: Encoding preset
            crf: Constant Rate Factor (lower = better quality)
            pixel_format: Output pixel format
            copy_audio: Copy audio streams from input
            copy_subtitles: Copy subtitle streams from input
            nvenc: Use NVIDIA hardware encoding
            
        Raises:
            FFmpegError: If encoding fails
        """
        frame_dir = Path(frame_dir)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Find frame files
        frame_files = list(frame_dir.glob("frame_*.png"))
        if not frame_files:
            frame_files = list(frame_dir.glob("frame_*.jpg"))
        
        if not frame_files:
            raise FFmpegError(f"No frame files found in {frame_dir}")
        
        # Determine input pattern
        first_frame = sorted(frame_files)[0]
        if first_frame.suffix == ".png":
            input_pattern = str(frame_dir / "frame_%08d.png")
        else:
            input_pattern = str(frame_dir / "frame_%08d.jpg")
        
        # Use hardware encoding if requested and available
        if nvenc:
            if codec == "libx264":
                codec = "h264_nvenc"
            elif codec == "libx265":
                codec = "hevc_nvenc"
        
        cmd = [
            self.ffmpeg_path, "-y",
            "-r", str(fps),
            "-i", input_pattern,
        ]
        
        # Add input video for audio/subtitle streams
        if input_video and copy_audio or copy_subtitles:
            cmd.extend(["-i", str(input_video)])
        
        # Video encoding options
        cmd.extend([
            "-c:v", codec,
            "-preset", preset,
            "-crf", str(crf),
            "-pix_fmt", pixel_format,
        ])
        
        # Audio handling
        if input_video and copy_audio:
            cmd.extend(["-c:a", "copy", "-map", "1:a?"])
        else:
            cmd.extend(["-an"])  # No audio
        
        # Subtitle handling
        if input_video and copy_subtitles:
            cmd.extend(["-c:s", "copy", "-map", "1:s?"])
        
        # Map video from frames
        cmd.extend(["-map", "0:v:0"])
        
        cmd.append(str(output_path))
        
        logger.info(f"Encoding video to {output_path}")
        logger.debug(f"Running command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info(f"Successfully encoded video: {output_path}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Video encoding failed: {e.stderr}")
            raise FFmpegError(f"Failed to encode video: {e.stderr}")
    
    def get_gpu_encoders(self) -> List[str]:
        """Get available hardware encoders.
        
        Returns:
            List of available hardware encoder names
        """
        try:
            result = subprocess.run(
                [self.ffmpeg_path, "-hide_banner", "-encoders"],
                capture_output=True,
                text=True,
                check=True
            )
            
            encoders = []
            for line in result.stdout.split('\n'):
                if 'nvenc' in line.lower():
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        encoders.append(parts[1])
            
            return encoders
            
        except subprocess.CalledProcessError:
            logger.warning("Could not query available encoders")
            return []