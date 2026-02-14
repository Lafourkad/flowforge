"""Scene change detection for video frame interpolation."""

import logging
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

logger = logging.getLogger(__name__)


class SceneDetector:
    """Detect scene changes in video frames to avoid interpolation across cuts."""
    
    def __init__(
        self,
        threshold: float = 0.3,
        method: str = "ssim",
        min_scene_length: int = 10
    ):
        """Initialize scene detector.
        
        Args:
            threshold: Scene change detection threshold (0.0-1.0)
                      Lower values = more sensitive to changes
            method: Detection method ("ssim", "histogram", "mse")
            min_scene_length: Minimum frames between scene changes
        """
        self.threshold = threshold
        self.method = method
        self.min_scene_length = min_scene_length
        
        if method not in ["ssim", "histogram", "mse"]:
            raise ValueError(f"Invalid method: {method}. Use 'ssim', 'histogram', or 'mse'")
        
        logger.info(
            f"Scene detector initialized: method={method}, "
            f"threshold={threshold}, min_scene_length={min_scene_length}"
        )
    
    def detect_scenes(self, frame_paths: List[Union[str, Path]]) -> List[int]:
        """Detect scene changes in a sequence of frames.
        
        Args:
            frame_paths: List of paths to frame images
            
        Returns:
            List of frame indices where scene changes occur
        """
        if len(frame_paths) < 2:
            return []
        
        logger.info(f"Analyzing {len(frame_paths)} frames for scene changes")
        scene_boundaries = []
        prev_frame = None
        
        for i, frame_path in enumerate(frame_paths):
            frame = self._load_frame(frame_path)
            if frame is None:
                logger.warning(f"Could not load frame: {frame_path}")
                continue
            
            if prev_frame is not None:
                similarity = self._calculate_similarity(prev_frame, frame)
                
                # Scene change detected if similarity is below threshold
                if similarity < self.threshold:
                    # Enforce minimum scene length
                    if not scene_boundaries or (i - scene_boundaries[-1]) >= self.min_scene_length:
                        scene_boundaries.append(i)
                        logger.debug(f"Scene change detected at frame {i} (similarity: {similarity:.3f})")
            
            prev_frame = frame
        
        logger.info(f"Detected {len(scene_boundaries)} scene changes")
        return scene_boundaries
    
    def detect_scenes_with_scores(
        self, 
        frame_paths: List[Union[str, Path]]
    ) -> Tuple[List[int], List[float]]:
        """Detect scene changes and return similarity scores.
        
        Args:
            frame_paths: List of paths to frame images
            
        Returns:
            Tuple of (scene_boundaries, similarity_scores)
        """
        if len(frame_paths) < 2:
            return [], []
        
        logger.info(f"Analyzing {len(frame_paths)} frames for scene changes (with scores)")
        scene_boundaries = []
        similarity_scores = []
        prev_frame = None
        
        for i, frame_path in enumerate(frame_paths):
            frame = self._load_frame(frame_path)
            if frame is None:
                logger.warning(f"Could not load frame: {frame_path}")
                continue
            
            if prev_frame is not None:
                similarity = self._calculate_similarity(prev_frame, frame)
                similarity_scores.append(similarity)
                
                # Scene change detected if similarity is below threshold
                if similarity < self.threshold:
                    # Enforce minimum scene length
                    if not scene_boundaries or (i - scene_boundaries[-1]) >= self.min_scene_length:
                        scene_boundaries.append(i)
                        logger.debug(f"Scene change detected at frame {i} (similarity: {similarity:.3f})")
            
            prev_frame = frame
        
        logger.info(f"Detected {len(scene_boundaries)} scene changes")
        return scene_boundaries, similarity_scores
    
    def _load_frame(self, frame_path: Union[str, Path]) -> np.ndarray:
        """Load and preprocess a frame for comparison.
        
        Args:
            frame_path: Path to frame image
            
        Returns:
            Preprocessed frame array or None if loading fails
        """
        try:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                return None
            
            # Convert to grayscale for comparison
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Resize for faster processing (maintain aspect ratio)
            height, width = frame.shape
            if width > 640:
                scale = 640 / width
                new_width = 640
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            return frame
            
        except Exception as e:
            logger.error(f"Failed to load frame {frame_path}: {e}")
            return None
    
    def _calculate_similarity(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate similarity between two frames.
        
        Args:
            frame1: First frame
            frame2: Second frame
            
        Returns:
            Similarity score (0.0 = completely different, 1.0 = identical)
        """
        if frame1.shape != frame2.shape:
            # Resize frame2 to match frame1
            frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
        
        if self.method == "ssim":
            return self._ssim_similarity(frame1, frame2)
        elif self.method == "histogram":
            return self._histogram_similarity(frame1, frame2)
        elif self.method == "mse":
            return self._mse_similarity(frame1, frame2)
        else:
            raise ValueError(f"Unknown similarity method: {self.method}")
    
    def _ssim_similarity(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate Structural Similarity Index (SSIM).
        
        Args:
            frame1: First frame
            frame2: Second frame
            
        Returns:
            SSIM value between -1 and 1 (mapped to 0-1 range)
        """
        try:
            # SSIM returns value between -1 and 1
            ssim_value = ssim(frame1, frame2)
            # Map to 0-1 range
            return (ssim_value + 1) / 2
        except Exception as e:
            logger.warning(f"SSIM calculation failed, falling back to MSE: {e}")
            return self._mse_similarity(frame1, frame2)
    
    def _histogram_similarity(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate histogram-based similarity.
        
        Args:
            frame1: First frame
            frame2: Second frame
            
        Returns:
            Correlation coefficient between histograms (0-1)
        """
        # Calculate histograms
        hist1 = cv2.calcHist([frame1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([frame2], [0], None, [256], [0, 256])
        
        # Normalize histograms
        cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        # Calculate correlation
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        # Ensure result is in 0-1 range
        return max(0.0, correlation)
    
    def _mse_similarity(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate Mean Squared Error based similarity.
        
        Args:
            frame1: First frame
            frame2: Second frame
            
        Returns:
            Similarity based on inverse normalized MSE (0-1)
        """
        # Calculate MSE
        mse = np.mean((frame1.astype(np.float32) - frame2.astype(np.float32)) ** 2)
        
        # Normalize MSE and convert to similarity (inverse relationship)
        # Max possible MSE for 8-bit images is 255^2
        max_mse = 255.0 ** 2
        normalized_mse = min(mse / max_mse, 1.0)
        
        # Convert to similarity (1 - normalized_mse)
        return 1.0 - normalized_mse
    
    def create_scene_segments(
        self, 
        frame_count: int, 
        scene_boundaries: List[int]
    ) -> List[Tuple[int, int]]:
        """Create scene segments from boundary list.
        
        Args:
            frame_count: Total number of frames
            scene_boundaries: List of frame indices where scenes change
            
        Returns:
            List of (start_frame, end_frame) tuples for each scene
        """
        if not scene_boundaries:
            return [(0, frame_count - 1)]
        
        segments = []
        start_frame = 0
        
        for boundary in scene_boundaries:
            if boundary > start_frame:
                segments.append((start_frame, boundary - 1))
                start_frame = boundary
        
        # Add final segment
        if start_frame < frame_count:
            segments.append((start_frame, frame_count - 1))
        
        logger.info(f"Created {len(segments)} scene segments")
        return segments
    
    def should_interpolate_between(self, frame1_path: Union[str, Path], frame2_path: Union[str, Path]) -> bool:
        """Check if interpolation should be performed between two frames.
        
        Args:
            frame1_path: Path to first frame
            frame2_path: Path to second frame
            
        Returns:
            True if frames are similar enough for interpolation
        """
        frame1 = self._load_frame(frame1_path)
        frame2 = self._load_frame(frame2_path)
        
        if frame1 is None or frame2 is None:
            logger.warning("Could not load frames for similarity check")
            return False
        
        similarity = self._calculate_similarity(frame1, frame2)
        should_interpolate = similarity >= self.threshold
        
        logger.debug(
            f"Frame similarity: {similarity:.3f}, "
            f"interpolate: {should_interpolate}"
        )
        
        return should_interpolate