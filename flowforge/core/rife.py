"""RIFE model wrapper for frame interpolation."""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Union

from ..utils.download import ModelDownloader

logger = logging.getLogger(__name__)


class RIFEError(Exception):
    """Exception raised when RIFE operations fail."""
    pass


class RIFEModel:
    """Wrapper for RIFE-NCNN-Vulkan frame interpolation."""
    
    SUPPORTED_MODELS = [
        "rife-v4.6",
        "rife-v4.15-lite"
    ]
    
    def __init__(
        self,
        model_name: str = "rife-v4.6",
        gpu_id: int = 0,
        num_threads: int = 1,
        tta_mode: bool = False,
        tta_temporal_mode: bool = False,
        uhd_mode: bool = False,
        install_dir: Optional[Path] = None
    ):
        """Initialize RIFE model wrapper.
        
        Args:
            model_name: RIFE model to use
            gpu_id: GPU device ID (-1 for CPU)
            num_threads: Number of threads for processing
            tta_mode: Test-time augmentation mode
            tta_temporal_mode: Temporal test-time augmentation
            uhd_mode: Ultra HD mode for high resolution videos
            install_dir: Custom installation directory
            
        Raises:
            RIFEError: If model is not supported or RIFE is not installed
        """
        if model_name not in self.SUPPORTED_MODELS:
            raise RIFEError(f"Unsupported model: {model_name}. Supported: {self.SUPPORTED_MODELS}")
        
        self.model_name = model_name
        self.gpu_id = gpu_id
        self.num_threads = num_threads
        self.tta_mode = tta_mode
        self.tta_temporal_mode = tta_temporal_mode
        self.uhd_mode = uhd_mode
        
        # Initialize downloader and check installation
        self.downloader = ModelDownloader(install_dir)
        self._check_installation()
        
        logger.info(
            f"RIFE model initialized: {model_name}, GPU: {gpu_id}, "
            f"threads: {num_threads}, TTA: {tta_mode}"
        )
    
    def _check_installation(self) -> None:
        """Check if RIFE binary and model are installed."""
        # Check binary
        if not self.downloader.is_rife_installed():
            raise RIFEError(
                "RIFE binary not found. Run 'flowforge setup' to install dependencies."
            )
        
        # Check model
        if not self.downloader.is_model_installed(self.model_name):
            raise RIFEError(
                f"RIFE model '{self.model_name}' not found. "
                "Run 'flowforge setup' to download models."
            )
        
        # Test binary execution
        try:
            binary_path = self.downloader.get_rife_binary_path()
            result = subprocess.run(
                [str(binary_path), "-h"],
                capture_output=True,
                text=True,
                timeout=10
            )
            # RIFE prints usage to stderr and may return non-zero for -h, that's OK
            output = result.stdout + result.stderr
            if "rife-ncnn-vulkan" not in output and "Usage" not in output:
                raise RIFEError(f"RIFE binary test failed: unexpected output")
                
        except subprocess.TimeoutExpired:
            raise RIFEError("RIFE binary test timed out")
        except FileNotFoundError:
            raise RIFEError("RIFE binary not executable")
    
    def get_binary_path(self) -> Path:
        """Get path to RIFE binary."""
        return self.downloader.get_rife_binary_path()
    
    def get_model_path(self) -> Path:
        """Get path to model directory."""
        return self.downloader.get_model_path(self.model_name)
    
    def interpolate_frames(
        self,
        input_frame1: Union[str, Path],
        input_frame2: Union[str, Path],
        output_dir: Union[str, Path],
        multiplier: int = 2,
        timestep: Optional[float] = None
    ) -> List[Path]:
        """Interpolate frames between two input frames.
        
        Args:
            input_frame1: Path to first input frame
            input_frame2: Path to second input frame
            output_dir: Directory to save interpolated frames
            multiplier: Interpolation multiplier (2, 4, 8)
            timestep: Custom timestep (0.0-1.0) for single interpolation
            
        Returns:
            List of paths to generated frames
            
        Raises:
            RIFEError: If interpolation fails
        """
        input_frame1 = Path(input_frame1)
        input_frame2 = Path(input_frame2)
        output_dir = Path(output_dir)
        
        # Validate inputs
        if not input_frame1.exists():
            raise RIFEError(f"Input frame not found: {input_frame1}")
        if not input_frame2.exists():
            raise RIFEError(f"Input frame not found: {input_frame2}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use timestep mode for single interpolation, multiplier for multiple
        if timestep is not None:
            return self._interpolate_timestep(input_frame1, input_frame2, output_dir, timestep)
        else:
            return self._interpolate_multiplier(input_frame1, input_frame2, output_dir, multiplier)
    
    def _interpolate_timestep(
        self,
        frame1: Path,
        frame2: Path,
        output_dir: Path,
        timestep: float
    ) -> List[Path]:
        """Interpolate single frame at specific timestep.
        
        Args:
            frame1: First input frame
            frame2: Second input frame
            output_dir: Output directory
            timestep: Timestep value (0.0-1.0)
            
        Returns:
            List containing path to generated frame
        """
        if not 0.0 <= timestep <= 1.0:
            raise RIFEError(f"Timestep must be between 0.0 and 1.0, got: {timestep}")
        
        binary_path = self.get_binary_path()
        model_path = self.get_model_path()
        
        output_filename = f"interpolated_{timestep:.3f}.png"
        output_path = output_dir / output_filename
        
        cmd = [
            str(binary_path),
            "-i", str(frame1),
            "-i", str(frame2),
            "-o", str(output_path),
            "-m", str(model_path),
            "-g", str(self.gpu_id),
            "-j", str(self.num_threads),
            "-t", str(timestep)
        ]
        
        # Add optional flags
        if self.tta_mode:
            cmd.append("-x")
        if self.tta_temporal_mode:
            cmd.append("-z")
        if self.uhd_mode:
            cmd.append("-u")
        
        logger.debug(f"Running RIFE command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=300  # 5 minute timeout
            )
            
            if not output_path.exists():
                raise RIFEError("RIFE did not generate expected output file")
            
            logger.debug(f"Generated interpolated frame: {output_path}")
            return [output_path]
            
        except subprocess.CalledProcessError as e:
            logger.error(f"RIFE interpolation failed: {e.stderr}")
            raise RIFEError(f"RIFE interpolation failed: {e.stderr}")
        except subprocess.TimeoutExpired:
            raise RIFEError("RIFE interpolation timed out")
    
    def _interpolate_multiplier(
        self,
        frame1: Path,
        frame2: Path,
        output_dir: Path,
        multiplier: int
    ) -> List[Path]:
        """Interpolate frames with specified multiplier.
        
        Args:
            frame1: First input frame
            frame2: Second input frame
            output_dir: Output directory
            multiplier: Interpolation multiplier
            
        Returns:
            List of paths to generated frames
        """
        if multiplier not in [2, 4, 8]:
            raise RIFEError(f"Unsupported multiplier: {multiplier}. Use 2, 4, or 8.")
        
        binary_path = self.get_binary_path()
        model_path = self.get_model_path()
        
        cmd = [
            str(binary_path),
            "-i", str(frame1),
            "-i", str(frame2),
            "-o", str(output_dir),
            "-m", str(model_path),
            "-g", str(self.gpu_id),
            "-j", str(self.num_threads),
            "-n", str(multiplier - 1)  # RIFE uses n+1 output frames
        ]
        
        # Add optional flags
        if self.tta_mode:
            cmd.append("-x")
        if self.tta_temporal_mode:
            cmd.append("-z")
        if self.uhd_mode:
            cmd.append("-u")
        
        logger.debug(f"Running RIFE command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=300  # 5 minute timeout
            )
            
            # Find generated frames
            generated_frames = sorted(output_dir.glob("*.png"))
            if not generated_frames:
                generated_frames = sorted(output_dir.glob("*.jpg"))
            
            if len(generated_frames) != multiplier - 1:
                raise RIFEError(
                    f"Expected {multiplier - 1} frames, got {len(generated_frames)}"
                )
            
            logger.debug(f"Generated {len(generated_frames)} interpolated frames")
            return generated_frames
            
        except subprocess.CalledProcessError as e:
            logger.error(f"RIFE interpolation failed: {e.stderr}")
            raise RIFEError(f"RIFE interpolation failed: {e.stderr}")
        except subprocess.TimeoutExpired:
            raise RIFEError("RIFE interpolation timed out")
    
    def interpolate_sequence(
        self,
        input_frames: List[Union[str, Path]],
        output_dir: Union[str, Path],
        multiplier: int = 2,
        skip_scene_changes: bool = True,
        scene_threshold: float = 0.3
    ) -> List[Path]:
        """Interpolate an entire sequence of frames.
        
        Args:
            input_frames: List of input frame paths
            output_dir: Output directory for interpolated frames
            multiplier: Interpolation multiplier
            skip_scene_changes: Skip interpolation across scene changes
            scene_threshold: Scene change detection threshold
            
        Returns:
            List of all output frame paths (original + interpolated)
            
        Raises:
            RIFEError: If interpolation fails
        """
        if len(input_frames) < 2:
            raise RIFEError("Need at least 2 frames for interpolation")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Detect scene changes if requested
        scene_changes = set()
        if skip_scene_changes:
            from .scene_detect import SceneDetector
            scene_detector = SceneDetector(threshold=scene_threshold)
            scene_changes = set(scene_detector.detect_scenes(input_frames))
            logger.info(f"Scene changes detected at frames: {sorted(scene_changes)}")
        
        output_frames = []
        frame_counter = 0
        
        for i in range(len(input_frames) - 1):
            # Copy current frame
            current_frame = Path(input_frames[i])
            output_frame = output_dir / f"frame_{frame_counter:08d}.png"
            
            # Copy frame (convert if needed)
            self._copy_frame(current_frame, output_frame)
            output_frames.append(output_frame)
            frame_counter += 1
            
            # Skip interpolation if scene change detected
            if (i + 1) in scene_changes:
                logger.debug(f"Skipping interpolation at scene change: frame {i + 1}")
                continue
            
            # Interpolate between current and next frame
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_dir = Path(temp_dir)
                    
                    interpolated = self.interpolate_frames(
                        input_frames[i],
                        input_frames[i + 1],
                        temp_dir,
                        multiplier=multiplier
                    )
                    
                    # Copy interpolated frames to output
                    for interp_frame in interpolated:
                        output_frame = output_dir / f"frame_{frame_counter:08d}.png"
                        self._copy_frame(interp_frame, output_frame)
                        output_frames.append(output_frame)
                        frame_counter += 1
                        
            except RIFEError as e:
                logger.warning(f"Skipping interpolation between frames {i}-{i+1}: {e}")
                continue
        
        # Copy last frame
        last_frame = Path(input_frames[-1])
        output_frame = output_dir / f"frame_{frame_counter:08d}.png"
        self._copy_frame(last_frame, output_frame)
        output_frames.append(output_frame)
        
        logger.info(f"Sequence interpolation complete: {len(output_frames)} total frames")
        return output_frames
    
    def _copy_frame(self, src: Path, dst: Path) -> None:
        """Copy and potentially convert a frame file.
        
        Args:
            src: Source frame path
            dst: Destination frame path
        """
        try:
            # If formats match, simple copy
            if src.suffix.lower() == dst.suffix.lower():
                import shutil
                shutil.copy2(str(src), str(dst))
            else:
                # Convert using PIL
                from PIL import Image
                with Image.open(src) as img:
                    img.save(dst)
                    
        except Exception as e:
            raise RIFEError(f"Failed to copy frame {src} -> {dst}: {e}")
    
    def get_gpu_info(self) -> dict:
        """Get GPU information from RIFE binary.
        
        Returns:
            Dictionary with GPU information
        """
        binary_path = self.get_binary_path()
        
        try:
            result = subprocess.run(
                [str(binary_path), "-v"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # Parse output for GPU information
            gpu_info = {
                "gpu_count": 0,
                "gpus": [],
                "current_gpu": self.gpu_id,
                "vulkan_available": "vulkan" in result.stdout.lower()
            }
            
            # Simple parsing - actual format depends on RIFE output
            lines = result.stdout.split('\n')
            for line in lines:
                if 'gpu' in line.lower() and ':' in line:
                    gpu_info["gpu_count"] += 1
                    gpu_info["gpus"].append(line.strip())
            
            return gpu_info
            
        except Exception as e:
            logger.warning(f"Could not get GPU info: {e}")
            return {"error": str(e)}
    
    def test_interpolation(self) -> bool:
        """Test RIFE interpolation with dummy frames.
        
        Returns:
            True if test succeeds
        """
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir = Path(temp_dir)
                
                # Create test frames
                from PIL import Image
                import numpy as np
                
                # Create simple test images
                img1 = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
                img2 = Image.fromarray(np.ones((64, 64, 3), dtype=np.uint8) * 255)
                
                frame1_path = temp_dir / "frame1.png"
                frame2_path = temp_dir / "frame2.png"
                
                img1.save(frame1_path)
                img2.save(frame2_path)
                
                # Test interpolation
                output_frames = self.interpolate_frames(
                    frame1_path,
                    frame2_path,
                    temp_dir,
                    multiplier=2
                )
                
                return len(output_frames) > 0
                
        except Exception as e:
            logger.error(f"RIFE test failed: {e}")
            return False
    
    def __str__(self) -> str:
        """String representation of RIFE model."""
        return (
            f"RIFEModel(model={self.model_name}, gpu={self.gpu_id}, "
            f"threads={self.num_threads})"
        )