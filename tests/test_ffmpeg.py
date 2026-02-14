"""Tests for FFmpeg utilities."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from flowforge.core.ffmpeg import FFmpegError, FFmpegProcessor, VideoInfo


class TestVideoInfo(unittest.TestCase):
    """Test VideoInfo class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_data = {
            'streams': [
                {
                    'codec_type': 'video',
                    'width': 1920,
                    'height': 1080,
                    'avg_frame_rate': '30/1',
                    'nb_frames': '1500',
                    'duration': '50.0',
                    'codec_name': 'h264',
                    'pix_fmt': 'yuv420p',
                    'bit_rate': '5000000'
                },
                {
                    'codec_type': 'audio',
                    'codec_name': 'aac',
                    'channels': 2
                },
                {
                    'codec_type': 'subtitle',
                    'codec_name': 'subrip'
                }
            ],
            'format': {
                'duration': '50.0',
                'bit_rate': '5500000'
            }
        }
    
    def test_video_properties(self):
        """Test basic video properties."""
        info = VideoInfo(self.sample_data)
        
        self.assertEqual(info.width, 1920)
        self.assertEqual(info.height, 1080)
        self.assertEqual(info.fps, 30.0)
        self.assertEqual(info.frame_count, 1500)
        self.assertEqual(info.duration, 50.0)
        self.assertEqual(info.codec, 'h264')
        self.assertEqual(info.pixel_format, 'yuv420p')
        self.assertEqual(info.bitrate, 5000000)
    
    def test_stream_counts(self):
        """Test stream counting."""
        info = VideoInfo(self.sample_data)
        
        self.assertEqual(len(info.audio_streams), 1)
        self.assertEqual(len(info.subtitle_streams), 1)
    
    def test_empty_data(self):
        """Test handling of empty data."""
        info = VideoInfo({'streams': []})
        
        self.assertEqual(info.width, 0)
        self.assertEqual(info.height, 0)
        self.assertEqual(info.fps, 0.0)
        self.assertEqual(info.frame_count, 0)
        self.assertEqual(info.duration, 0.0)
        self.assertEqual(info.codec, 'unknown')
    
    def test_fps_calculation(self):
        """Test FPS calculation from fraction."""
        data = {
            'streams': [
                {
                    'codec_type': 'video',
                    'avg_frame_rate': '24000/1001'  # 23.976 fps
                }
            ]
        }
        info = VideoInfo(data)
        self.assertAlmostEqual(info.fps, 23.976, places=2)
    
    def test_fallback_duration(self):
        """Test duration fallback to format info."""
        data = {
            'streams': [
                {
                    'codec_type': 'video',
                    # No duration in stream
                }
            ],
            'format': {
                'duration': '100.5'
            }
        }
        info = VideoInfo(data)
        self.assertEqual(info.duration, 100.5)
    
    def test_string_representation(self):
        """Test string representation."""
        info = VideoInfo(self.sample_data)
        str_repr = str(info)
        
        self.assertIn('1920x1080', str_repr)
        self.assertIn('30.00fps', str_repr)
        self.assertIn('h264', str_repr)


class TestFFmpegProcessor(unittest.TestCase):
    """Test FFmpegProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = None
    
    def tearDown(self):
        """Clean up after tests."""
        if self.temp_dir:
            self.temp_dir.cleanup()
    
    @patch('shutil.which')
    def test_initialization(self, mock_which):
        """Test processor initialization."""
        mock_which.side_effect = lambda x: f'/usr/bin/{x}' if x in ['ffmpeg', 'ffprobe'] else None
        
        processor = FFmpegProcessor()
        self.assertEqual(processor.ffmpeg_path, 'ffmpeg')
        self.assertEqual(processor.ffprobe_path, 'ffprobe')
    
    @patch('shutil.which')
    def test_missing_ffmpeg(self, mock_which):
        """Test error when FFmpeg is missing."""
        mock_which.return_value = None
        
        with self.assertRaises(FFmpegError):
            FFmpegProcessor()
    
    @patch('subprocess.run')
    @patch('shutil.which')
    def test_probe_video_success(self, mock_which, mock_run):
        """Test successful video probing."""
        mock_which.side_effect = lambda x: f'/usr/bin/{x}'
        
        # Mock successful ffprobe output
        mock_run.return_value.stdout = '''
        {
            "streams": [
                {
                    "codec_type": "video",
                    "width": 1920,
                    "height": 1080,
                    "avg_frame_rate": "30/1"
                }
            ],
            "format": {
                "duration": "60.0"
            }
        }
        '''
        mock_run.return_value.returncode = 0
        
        # Create temporary test file
        self.temp_dir = tempfile.TemporaryDirectory()
        test_file = Path(self.temp_dir.name) / 'test.mp4'
        test_file.touch()
        
        processor = FFmpegProcessor()
        info = processor.probe_video(test_file)
        
        self.assertEqual(info.width, 1920)
        self.assertEqual(info.height, 1080)
        self.assertEqual(info.fps, 30.0)
    
    @patch('shutil.which')
    def test_probe_nonexistent_file(self, mock_which):
        """Test probing non-existent file."""
        mock_which.side_effect = lambda x: f'/usr/bin/{x}'
        
        processor = FFmpegProcessor()
        
        with self.assertRaises(FFmpegError):
            processor.probe_video('nonexistent.mp4')
    
    @patch('subprocess.run')
    @patch('shutil.which')
    def test_probe_video_failure(self, mock_which, mock_run):
        """Test handling of ffprobe failure."""
        mock_which.side_effect = lambda x: f'/usr/bin/{x}'
        
        # Mock failed ffprobe
        from subprocess import CalledProcessError
        mock_run.side_effect = CalledProcessError(1, 'ffprobe', stderr='Invalid file')
        
        # Create temporary test file
        self.temp_dir = tempfile.TemporaryDirectory()
        test_file = Path(self.temp_dir.name) / 'test.mp4'
        test_file.touch()
        
        processor = FFmpegProcessor()
        
        with self.assertRaises(FFmpegError):
            processor.probe_video(test_file)
    
    @patch('subprocess.run')
    @patch('shutil.which')
    def test_extract_frames_success(self, mock_which, mock_run):
        """Test successful frame extraction."""
        mock_which.side_effect = lambda x: f'/usr/bin/{x}'
        mock_run.return_value.returncode = 0
        
        # Create temporary directories
        self.temp_dir = tempfile.TemporaryDirectory()
        temp_path = Path(self.temp_dir.name)
        
        video_file = temp_path / 'test.mp4'
        video_file.touch()
        
        output_dir = temp_path / 'frames'
        output_dir.mkdir()
        
        # Create mock extracted frames
        for i in range(5):
            (output_dir / f'frame_{i+1:08d}.png').touch()
        
        processor = FFmpegProcessor()
        frame_count, frame_files = processor.extract_frames(video_file, output_dir)
        
        self.assertEqual(frame_count, 5)
        self.assertEqual(len(frame_files), 5)
        mock_run.assert_called_once()
    
    @patch('subprocess.run')
    @patch('shutil.which')
    def test_encode_video_success(self, mock_which, mock_run):
        """Test successful video encoding."""
        mock_which.side_effect = lambda x: f'/usr/bin/{x}'
        mock_run.return_value.returncode = 0
        
        # Create temporary directories
        self.temp_dir = tempfile.TemporaryDirectory()
        temp_path = Path(self.temp_dir.name)
        
        frame_dir = temp_path / 'frames'
        frame_dir.mkdir()
        
        # Create mock frame files
        for i in range(5):
            (frame_dir / f'frame_{i+1:08d}.png').touch()
        
        output_file = temp_path / 'output.mp4'
        
        processor = FFmpegProcessor()
        processor.encode_video(frame_dir, output_file, fps=30.0)
        
        mock_run.assert_called_once()
        # Check that command includes expected parameters
        call_args = mock_run.call_args[0][0]
        self.assertIn('ffmpeg', call_args[0])
        self.assertIn('-r', call_args)
        self.assertIn('30.0', call_args)
    
    @patch('subprocess.run')
    @patch('shutil.which')
    def test_get_gpu_encoders(self, mock_which, mock_run):
        """Test GPU encoder detection."""
        mock_which.side_effect = lambda x: f'/usr/bin/{x}'
        
        # Mock ffmpeg encoder list output
        mock_run.return_value.stdout = '''
        V..... h264_nvenc            NVIDIA NVENC H.264 encoder
        V..... hevc_nvenc            NVIDIA NVENC H.265/HEVC encoder
        V..... libx264               libx264 H.264 / AVC / MPEG-4 AVC
        '''
        mock_run.return_value.returncode = 0
        
        processor = FFmpegProcessor()
        encoders = processor.get_gpu_encoders()
        
        self.assertIn('h264_nvenc', encoders)
        self.assertIn('hevc_nvenc', encoders)
    
    @patch('subprocess.run')
    @patch('shutil.which')
    def test_encode_with_nvenc(self, mock_which, mock_run):
        """Test encoding with NVENC."""
        mock_which.side_effect = lambda x: f'/usr/bin/{x}'
        mock_run.return_value.returncode = 0
        
        # Create temporary directories
        self.temp_dir = tempfile.TemporaryDirectory()
        temp_path = Path(self.temp_dir.name)
        
        frame_dir = temp_path / 'frames'
        frame_dir.mkdir()
        
        # Create mock frame files
        (frame_dir / 'frame_00000001.png').touch()
        
        output_file = temp_path / 'output.mp4'
        
        processor = FFmpegProcessor()
        processor.encode_video(
            frame_dir, 
            output_file, 
            fps=30.0,
            codec='libx264',
            nvenc=True
        )
        
        # Check that h264_nvenc was used instead of libx264
        call_args = mock_run.call_args[0][0]
        self.assertIn('h264_nvenc', call_args)
        self.assertNotIn('libx264', call_args)


class TestIntegration(unittest.TestCase):
    """Integration tests for FFmpeg functionality."""
    
    @patch('subprocess.run')
    @patch('shutil.which')
    def test_full_workflow(self, mock_which, mock_run):
        """Test complete extract->encode workflow."""
        mock_which.side_effect = lambda x: f'/usr/bin/{x}'
        
        # Mock successful operations
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = '''
        {
            "streams": [{"codec_type": "video", "width": 640, "height": 480, "avg_frame_rate": "24/1"}],
            "format": {"duration": "10.0"}
        }
        '''
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test video file
            input_video = temp_path / 'input.mp4'
            input_video.touch()
            
            # Create frame directory and mock frames
            frames_dir = temp_path / 'frames'
            frames_dir.mkdir()
            for i in range(3):
                (frames_dir / f'frame_{i+1:08d}.png').touch()
            
            output_video = temp_path / 'output.mp4'
            
            processor = FFmpegProcessor()
            
            # Test video probing
            info = processor.probe_video(input_video)
            self.assertEqual(info.width, 640)
            
            # Test frame extraction
            frame_count, frame_files = processor.extract_frames(input_video, frames_dir)
            self.assertEqual(frame_count, 3)
            
            # Test video encoding
            processor.encode_video(frames_dir, output_video, fps=24.0)
            
            # Verify ffmpeg was called for encoding
            encoding_call_found = False
            for call in mock_run.call_args_list:
                if 'ffmpeg' in str(call) and str(output_video) in str(call):
                    encoding_call_found = True
                    break
            
            self.assertTrue(encoding_call_found)


if __name__ == '__main__':
    unittest.main()