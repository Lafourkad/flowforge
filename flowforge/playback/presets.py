"""Interpolation presets for real-time playback."""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class InterpolationPreset:
    """RIFE interpolation preset configuration."""
    
    name: str
    description: str
    target_fps: float
    multiplier: Optional[float] = None
    
    # RIFE parameters
    model: str = "rife-v4.6"
    tta: bool = False  # Test Time Augmentation
    uhd: bool = False  # Ultra HD mode
    skip: bool = True  # Skip duplicate frames
    
    # Quality/performance tradeoffs
    tile_size: int = 512  # Tile size for processing large frames
    tile_pad: int = 32   # Tile padding to reduce artifacts
    prepad: int = 0      # Temporal prepadding
    quality_profile: str = "balanced"  # fast, balanced, quality
    
    # Scene detection
    scene_detection: bool = True
    scene_threshold: float = 0.3
    scene_method: str = "ssim"  # ssim, histogram, mse
    
    # Buffer management
    buffer_frames: int = 8  # Read-ahead buffer size
    max_queue_size: int = 16  # Maximum frame queue
    drop_threshold: float = 0.8  # Drop frames when queue > this ratio
    
    # GPU/memory management
    gpu_memory_fraction: float = 0.8  # Max VRAM usage
    fallback_cpu: bool = True  # Fall back to CPU if GPU overloaded
    
    def to_dict(self) -> Dict:
        """Convert preset to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'InterpolationPreset':
        """Create preset from dictionary."""
        return cls(**data)
    
    def validate(self) -> None:
        """Validate preset parameters."""
        if self.target_fps <= 0:
            raise ValueError("target_fps must be positive")
        if self.multiplier is not None and self.multiplier <= 1.0:
            raise ValueError("multiplier must be > 1.0")
        if not 0.0 <= self.scene_threshold <= 1.0:
            raise ValueError("scene_threshold must be 0.0-1.0")
        if self.buffer_frames < 1:
            raise ValueError("buffer_frames must be >= 1")
        if self.max_queue_size < self.buffer_frames:
            raise ValueError("max_queue_size must be >= buffer_frames")


class PresetManager:
    """Manager for interpolation presets."""
    
    def __init__(self, presets_dir: Optional[Path] = None):
        """Initialize preset manager.
        
        Args:
            presets_dir: Directory to store custom presets
        """
        self.presets_dir = presets_dir or Path.home() / ".flowforge" / "presets"
        self.presets_dir.mkdir(parents=True, exist_ok=True)
        
        # Built-in presets
        self._builtin_presets = {
            "film": InterpolationPreset(
                name="film",
                description="24→60fps for films, moderate artifact suppression",
                target_fps=60.0,
                multiplier=2.5,  # 24*2.5=60
                model="rife-v4.6",
                tta=False,
                uhd=False,
                quality_profile="balanced",
                scene_detection=True,
                scene_threshold=0.25,  # More sensitive for film cuts
                buffer_frames=6,
                tile_size=512
            ),
            
            "anime": InterpolationPreset(
                name="anime",
                description="24→60fps for anime, strong artifact suppression",
                target_fps=60.0,
                multiplier=2.5,
                model="rife-v4.6",
                tta=True,  # Better for hard edges
                uhd=False,
                quality_profile="quality",
                scene_detection=True,
                scene_threshold=0.35,  # Less sensitive, anime has fewer cuts
                buffer_frames=8,
                tile_size=400,  # Smaller tiles for better quality
                tile_pad=40
            ),
            
            "sports": InterpolationPreset(
                name="sports",
                description="30→60fps for sports, fast processing",
                target_fps=60.0,
                multiplier=2.0,
                model="rife-v4.15-lite",  # Faster model
                tta=False,
                uhd=False,
                quality_profile="fast",
                scene_detection=False,  # Sports have continuous motion
                buffer_frames=4,  # Smaller buffer for lower latency
                tile_size=640,
                drop_threshold=0.6  # More aggressive frame dropping
            ),
            
            "smooth": InterpolationPreset(
                name="smooth",
                description="Maximum smoothness (any→144fps)",
                target_fps=144.0,
                multiplier=6.0,  # 24*6=144
                model="rife-v4.6",
                tta=False,
                uhd=True,  # High quality for high fps
                quality_profile="balanced",
                scene_detection=True,
                scene_threshold=0.2,
                buffer_frames=12,
                max_queue_size=24,
                tile_size=384  # Smaller for more precision
            ),
            
            "custom": InterpolationPreset(
                name="custom",
                description="User-defined settings",
                target_fps=60.0,
                multiplier=2.0,
                model="rife-v4.6",
                quality_profile="balanced",
                buffer_frames=8
            )
        }
        
    def get_builtin_presets(self) -> Dict[str, InterpolationPreset]:
        """Get all built-in presets."""
        return self._builtin_presets.copy()
    
    def get_preset(self, name: str) -> InterpolationPreset:
        """Get preset by name.
        
        Args:
            name: Preset name
            
        Returns:
            InterpolationPreset instance
            
        Raises:
            KeyError: If preset not found
        """
        # Check built-in presets first
        if name in self._builtin_presets:
            return self._builtin_presets[name]
        
        # Check custom presets
        preset_file = self.presets_dir / f"{name}.json"
        if preset_file.exists():
            with open(preset_file) as f:
                data = json.load(f)
            return InterpolationPreset.from_dict(data)
        
        raise KeyError(f"Preset '{name}' not found")
    
    def list_presets(self) -> List[str]:
        """List all available presets."""
        presets = list(self._builtin_presets.keys())
        
        # Add custom presets
        for preset_file in self.presets_dir.glob("*.json"):
            name = preset_file.stem
            if name not in presets:
                presets.append(name)
        
        return sorted(presets)
    
    def save_preset(self, preset: InterpolationPreset, overwrite: bool = False) -> None:
        """Save custom preset.
        
        Args:
            preset: Preset to save
            overwrite: Allow overwriting existing preset
            
        Raises:
            FileExistsError: If preset exists and overwrite=False
            ValueError: If trying to overwrite built-in preset
        """
        if preset.name in self._builtin_presets:
            raise ValueError(f"Cannot overwrite built-in preset '{preset.name}'")
        
        preset_file = self.presets_dir / f"{preset.name}.json"
        
        if preset_file.exists() and not overwrite:
            raise FileExistsError(f"Preset '{preset.name}' already exists")
        
        preset.validate()
        
        with open(preset_file, 'w') as f:
            json.dump(preset.to_dict(), f, indent=2)
        
        logger.info(f"Saved preset '{preset.name}' to {preset_file}")
    
    def delete_preset(self, name: str) -> None:
        """Delete custom preset.
        
        Args:
            name: Preset name to delete
            
        Raises:
            ValueError: If trying to delete built-in preset
            FileNotFoundError: If preset doesn't exist
        """
        if name in self._builtin_presets:
            raise ValueError(f"Cannot delete built-in preset '{name}'")
        
        preset_file = self.presets_dir / f"{name}.json"
        
        if not preset_file.exists():
            raise FileNotFoundError(f"Preset '{name}' not found")
        
        preset_file.unlink()
        logger.info(f"Deleted preset '{name}'")
    
    def get_preset_info(self, name: str) -> Dict:
        """Get detailed preset information.
        
        Args:
            name: Preset name
            
        Returns:
            Dictionary with preset info and metadata
        """
        preset = self.get_preset(name)
        is_builtin = name in self._builtin_presets
        
        return {
            "preset": preset.to_dict(),
            "is_builtin": is_builtin,
            "can_modify": not is_builtin,
            "file_path": str(self.presets_dir / f"{name}.json") if not is_builtin else None
        }
    
    def create_adaptive_preset(
        self, 
        input_fps: float, 
        target_fps: float,
        content_type: str = "auto"
    ) -> InterpolationPreset:
        """Create adaptive preset based on input video.
        
        Args:
            input_fps: Input video frame rate
            target_fps: Desired output frame rate
            content_type: Content type hint (film, anime, sports, auto)
            
        Returns:
            Adaptive InterpolationPreset
        """
        multiplier = target_fps / input_fps
        
        # Auto-detect content type based on fps
        if content_type == "auto":
            if 23 <= input_fps <= 25:
                content_type = "film"
            elif 29 <= input_fps <= 31:
                content_type = "sports"
            else:
                content_type = "film"  # Default
        
        # Start with base preset
        base_preset = self._builtin_presets.get(content_type, self._builtin_presets["film"])
        
        # Create adaptive preset
        adaptive = InterpolationPreset(
            name=f"adaptive_{content_type}_{target_fps}fps",
            description=f"Adaptive {content_type} preset: {input_fps:.1f}→{target_fps:.1f}fps",
            target_fps=target_fps,
            multiplier=multiplier,
            model=base_preset.model,
            tta=base_preset.tta,
            uhd=multiplier >= 4.0,  # Enable UHD for high multipliers
            quality_profile="fast" if multiplier >= 6.0 else base_preset.quality_profile,
            scene_detection=base_preset.scene_detection,
            scene_threshold=base_preset.scene_threshold,
            buffer_frames=max(4, int(multiplier * 2)),
            max_queue_size=max(8, int(multiplier * 4)),
            tile_size=max(256, min(640, int(512 / (multiplier ** 0.5)))),
            drop_threshold=0.7 if multiplier >= 4.0 else 0.8
        )
        
        adaptive.validate()
        return adaptive


def get_system_recommendations() -> Dict[str, any]:
    """Get system-specific preset recommendations.
    
    Returns:
        Dictionary with recommended settings based on hardware
    """
    import platform
    import psutil
    
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        gpu_memory_gb = max(gpu.memoryTotal / 1024 for gpu in gpus) if gpus else 0
        gpu_available = len(gpus) > 0
    except (ImportError, Exception):
        gpu_memory_gb = 0
        gpu_available = False
    
    ram_gb = psutil.virtual_memory().total / (1024**3)
    cpu_cores = psutil.cpu_count()
    
    recommendations = {
        "platform": platform.system(),
        "ram_gb": ram_gb,
        "cpu_cores": cpu_cores,
        "gpu_available": gpu_available,
        "gpu_memory_gb": gpu_memory_gb,
        "recommended_presets": [],
        "performance_tips": []
    }
    
    # Preset recommendations based on hardware
    if gpu_memory_gb >= 8:
        recommendations["recommended_presets"].extend(["smooth", "anime", "film"])
        recommendations["performance_tips"].append("High-end GPU detected: all presets supported")
    elif gpu_memory_gb >= 4:
        recommendations["recommended_presets"].extend(["anime", "film", "sports"])
        recommendations["performance_tips"].append("Mid-range GPU: avoid 'smooth' preset for best performance")
    elif gpu_available:
        recommendations["recommended_presets"].extend(["sports", "film"])
        recommendations["performance_tips"].append("Low GPU memory: use 'sports' or reduced tile sizes")
    else:
        recommendations["recommended_presets"].append("sports")
        recommendations["performance_tips"].append("No GPU detected: CPU-only mode, expect slower performance")
    
    # Memory recommendations
    if ram_gb < 8:
        recommendations["performance_tips"].append("Low RAM: reduce buffer_frames to 4-6")
    elif ram_gb >= 32:
        recommendations["performance_tips"].append("High RAM: increase buffer_frames to 12-16 for smoother playback")
    
    return recommendations