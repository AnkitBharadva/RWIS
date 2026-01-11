"""
Pipeline configuration module for Railway Wagon Inspection.

Provides PipelineConfig dataclass and load_config() function for
loading all configurable parameters from a configuration file.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json
import os


@dataclass
class PipelineConfig:
    """Configuration for the railway wagon inspection pipeline."""
    
    # Video input
    video_source: str = ""
    
    # Blur thresholds (single threshold for simplified gating)
    blur_threshold: float = 100.0  # Below this: apply deblur; Above: skip deblur
    # Legacy thresholds for backward compatibility
    blur_threshold_t1: float = 100.0  # Below this: skip deblur
    blur_threshold_t2: float = 300.0  # Above this: no deblur (too blurry)
    
    # Model paths
    # NOTE: The model files are named opposite to their function:
    # - damage_detector.pt actually contains wagon_body/wheel classes
    # - wagon_detector.pt actually contains damage classes (Bamboo Door, Breakage, etc.)
    wagon_model_path: str = "models/damage_detector.pt"  # Contains: wagon_body, wheel
    damage_model_path: str = "models/wagon_detector.pt"  # Contains: Bamboo Door, Breakage, Close Door, Damage Door, Dent, Open Door, Wagon
    mprnet_model_path: str = "MPRNet/Deblurring/pretrained_models/model_deblurring.pth"
    # Legacy NAFNet path for backward compatibility
    nafnet_model_path: str = "models/nafnet_deblur.pth"
    
    # ROI resizing settings
    max_roi_width: int = 256  # Maximum ROI width for deblurring
    
    # N-th frame execution settings
    deblur_frame_interval: int = 3  # Run MPRNet every N frames (default 3-5)
    
    # Detection settings
    wagon_confidence_threshold: float = 0.25  # Lowered for better detection
    damage_confidence_threshold: float = 0.5
    
    # Counting line position (as fraction of frame dimension)
    counting_line_position: float = 0.5
    # Counting line orientation: "horizontal" (wagons move up/down) or "vertical" (wagons move left/right)
    counting_line_orientation: str = "vertical"  # Default to vertical for horizontal wagon movement
    
    # OCR settings
    ocr_gpu_enabled: bool = True
    ocr_language: str = "en"
    
    # Performance settings - FP16 inference
    use_fp16: bool = True  # Use FP16 (half precision) for MPRNet inference
    fp32_fallback: bool = True  # Fall back to FP32 on numerical instability
    max_batch_size: int = 1  # Batch size for MPRNet (always 1)
    enable_threading: bool = True
    
    # Output settings
    output_dir: str = "outputs"
    enable_debug_frames: bool = False
    log_format: List[str] = field(default_factory=lambda: ["csv", "json"])
    
    def validate(self) -> None:
        """Validate configuration values."""
        # Validate legacy blur thresholds if used
        if self.blur_threshold_t1 >= self.blur_threshold_t2:
            raise ValueError(
                f"blur_threshold_t1 ({self.blur_threshold_t1}) must be less than "
                f"blur_threshold_t2 ({self.blur_threshold_t2})"
            )
        
        # Validate single blur threshold
        if self.blur_threshold < 0:
            raise ValueError(
                f"blur_threshold must be non-negative, got {self.blur_threshold}"
            )
        
        if not 0.0 <= self.counting_line_position <= 1.0:
            raise ValueError(
                f"counting_line_position must be between 0.0 and 1.0, "
                f"got {self.counting_line_position}"
            )
        
        if self.counting_line_orientation not in ("horizontal", "vertical"):
            raise ValueError(
                f"counting_line_orientation must be 'horizontal' or 'vertical', "
                f"got {self.counting_line_orientation}"
            )
        
        if not 0.0 <= self.wagon_confidence_threshold <= 1.0:
            raise ValueError(
                f"wagon_confidence_threshold must be between 0.0 and 1.0, "
                f"got {self.wagon_confidence_threshold}"
            )
        
        if not 0.0 <= self.damage_confidence_threshold <= 1.0:
            raise ValueError(
                f"damage_confidence_threshold must be between 0.0 and 1.0, "
                f"got {self.damage_confidence_threshold}"
            )
        
        if self.max_batch_size < 1:
            raise ValueError(
                f"max_batch_size must be at least 1, got {self.max_batch_size}"
            )
        
        # Validate N-th frame interval
        if self.deblur_frame_interval < 1:
            raise ValueError(
                f"deblur_frame_interval must be at least 1, got {self.deblur_frame_interval}"
            )
        
        # Validate max ROI width
        if self.max_roi_width < 1:
            raise ValueError(
                f"max_roi_width must be at least 1, got {self.max_roi_width}"
            )
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def load_config(config_path: Optional[str] = None) -> PipelineConfig:
    """
    Load pipeline configuration from a JSON file.
    
    Args:
        config_path: Path to configuration JSON file. If None, returns default config.
    
    Returns:
        PipelineConfig instance with loaded values.
    
    Raises:
        FileNotFoundError: If config_path is specified but file doesn't exist.
        ValueError: If configuration values are invalid.
    """
    if config_path is None:
        config = PipelineConfig()
        config.validate()
        return config
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    config = PipelineConfig(**config_dict)
    config.validate()
    return config
