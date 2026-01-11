"""Calibration Manager for auto-calibrating blur and illumination thresholds.

This module provides automatic threshold calibration based on statistical
analysis of initial video frames. It computes optimal blur and illumination
thresholds using percentile-based calculations.

Feature: ocr-enhancement-improvements
Validates: Requirements 3.1, 3.2, 3.3, 3.4, 4.1, 4.2, 4.3, 4.4, 4.5
"""

import logging
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Result of auto-calibration.
    
    Attributes:
        blur_threshold: Computed blur threshold based on percentile
        low_light_threshold: Computed low-light threshold based on percentile
        gamma_value: Computed gamma value based on mean luminance
        sample_count: Number of samples used for calibration
        blur_scores: List of blur scores from samples
        luminance_values: List of mean luminance values from samples
    """
    blur_threshold: float
    low_light_threshold: int
    gamma_value: float
    sample_count: int
    blur_scores: List[float] = field(default_factory=list)
    luminance_values: List[float] = field(default_factory=list)


class CalibrationManager:
    """Manages auto-calibration for blur and illumination thresholds.
    
    The calibration process:
    1. Collect sample frames during initial video processing
    2. Compute blur scores and luminance values for each sample
    3. Use percentile-based analysis to determine optimal thresholds
    4. Provide calibration results to blur detector and illumination controller
    
    Attributes:
        sample_size: Number of frames to sample for calibration
        blur_percentile: Percentile for blur threshold (0-100)
        luminance_percentile: Percentile for low-light threshold (0-100)
    """
    
    DEFAULT_SAMPLE_SIZE = 30
    DEFAULT_BLUR_PERCENTILE = 50  # Median
    DEFAULT_LUMINANCE_PERCENTILE = 25  # Lower quartile
    
    # Gamma computation constants
    TARGET_LUMINANCE = 128  # Target mean luminance for gamma correction
    MIN_GAMMA = 0.5
    MAX_GAMMA = 2.5
    
    def __init__(
        self,
        sample_size: int = DEFAULT_SAMPLE_SIZE,
        blur_percentile: float = DEFAULT_BLUR_PERCENTILE,
        luminance_percentile: float = DEFAULT_LUMINANCE_PERCENTILE
    ):
        """Initialize calibration manager.
        
        Args:
            sample_size: Number of frames to sample for calibration
            blur_percentile: Percentile for blur threshold (0-100)
            luminance_percentile: Percentile for low-light threshold (0-100)
            
        Raises:
            ValueError: If percentile values are out of range [0, 100]
            ValueError: If sample_size is less than 1
        """
        if not 0 <= blur_percentile <= 100:
            raise ValueError(f"blur_percentile must be in [0, 100], got {blur_percentile}")
        if not 0 <= luminance_percentile <= 100:
            raise ValueError(f"luminance_percentile must be in [0, 100], got {luminance_percentile}")
        if sample_size < 1:
            raise ValueError(f"sample_size must be at least 1, got {sample_size}")
        
        self.sample_size = sample_size
        self.blur_percentile = blur_percentile
        self.luminance_percentile = luminance_percentile
        
        # Internal state
        self._blur_scores: List[float] = []
        self._luminance_values: List[float] = []
        self._calibration_complete = False
    
    def _compute_blur_score(self, frame: np.ndarray) -> float:
        """Compute blur score using Laplacian variance.
        
        Args:
            frame: Input frame as BGR numpy array (H, W, 3) or grayscale (H, W)
            
        Returns:
            Blur score as float. Higher values indicate sharper images.
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Compute Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())
    
    def _compute_mean_luminance(self, frame: np.ndarray) -> float:
        """Compute mean luminance of a frame.
        
        Args:
            frame: Input frame as BGR numpy array (H, W, 3) or grayscale (H, W)
            
        Returns:
            Mean luminance value (0-255)
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        return float(np.mean(gray))
    
    def add_sample(self, frame: np.ndarray) -> bool:
        """Add a frame sample for calibration.
        
        Computes blur score and luminance for the frame and adds them
        to the sample collections.
        
        Args:
            frame: Input frame as BGR numpy array
            
        Returns:
            True if calibration is complete (enough samples collected)
            
        Raises:
            ValueError: If frame is None or empty
        """
        if frame is None or frame.size == 0:
            logger.warning("Empty frame provided to calibration, skipping")
            return self._calibration_complete
        
        # Don't add more samples if calibration is already complete
        if self._calibration_complete:
            return True
        
        # Compute and store blur score
        blur_score = self._compute_blur_score(frame)
        self._blur_scores.append(blur_score)
        
        # Compute and store luminance
        luminance = self._compute_mean_luminance(frame)
        self._luminance_values.append(luminance)
        
        # Check if calibration is complete
        if len(self._blur_scores) >= self.sample_size:
            self._calibration_complete = True
            logger.info(f"Calibration complete with {len(self._blur_scores)} samples")
        
        return self._calibration_complete
    
    def is_calibration_complete(self) -> bool:
        """Check if calibration has enough samples.
        
        Returns:
            True if sample_count >= sample_size
        """
        return self._calibration_complete
    
    def get_calibration_progress(self) -> float:
        """Get calibration progress as a fraction.
        
        Returns:
            Progress value from 0.0 to 1.0
        """
        if self.sample_size == 0:
            return 1.0
        return min(1.0, len(self._blur_scores) / self.sample_size)
    
    def get_sample_count(self) -> int:
        """Get current number of samples collected.
        
        Returns:
            Number of samples collected
        """
        return len(self._blur_scores)
    
    def _compute_gamma_from_luminance(self, mean_luminance: float) -> float:
        """Compute optimal gamma value based on mean luminance.
        
        The gamma value is computed to bring the mean luminance closer
        to the target luminance (128). Lower luminance results in lower
        gamma (brightening), higher luminance results in higher gamma
        (darkening).
        
        Args:
            mean_luminance: Mean luminance of the samples
            
        Returns:
            Computed gamma value, clamped to [MIN_GAMMA, MAX_GAMMA]
        """
        if mean_luminance <= 0:
            return self.MIN_GAMMA
        
        # Compute gamma to achieve target luminance
        # Using the relationship: output = input^gamma
        # We want: target = mean^gamma
        # So: gamma = log(target) / log(mean)
        # But we invert it since gamma < 1 brightens
        
        # Simple linear mapping: low luminance -> low gamma (brighten)
        # high luminance -> high gamma (darken)
        # Normalize luminance to [0, 1] range
        normalized = mean_luminance / 255.0
        
        # Map to gamma range: low luminance (0) -> MIN_GAMMA, high luminance (1) -> MAX_GAMMA
        gamma = self.MIN_GAMMA + normalized * (self.MAX_GAMMA - self.MIN_GAMMA)
        
        return max(self.MIN_GAMMA, min(self.MAX_GAMMA, gamma))
    
    def compute_calibration(self) -> Optional[CalibrationResult]:
        """Compute calibrated thresholds from samples.
        
        Uses percentile-based analysis to determine optimal thresholds:
        - blur_threshold: Set at configured percentile of blur scores
        - low_light_threshold: Set at configured percentile of luminance values
        - gamma_value: Computed based on mean luminance
        
        Returns:
            CalibrationResult with computed thresholds, or None if
            insufficient samples
        """
        if len(self._blur_scores) == 0:
            logger.warning("No samples collected for calibration")
            return None
        
        # Compute blur threshold at configured percentile
        blur_threshold = float(np.percentile(self._blur_scores, self.blur_percentile))
        
        # Compute low-light threshold at configured percentile
        low_light_threshold = int(np.percentile(self._luminance_values, self.luminance_percentile))
        
        # Compute gamma based on mean luminance
        mean_luminance = float(np.mean(self._luminance_values))
        gamma_value = self._compute_gamma_from_luminance(mean_luminance)
        
        logger.info(
            f"Calibration computed: blur_threshold={blur_threshold:.2f}, "
            f"low_light_threshold={low_light_threshold}, gamma={gamma_value:.2f}"
        )
        
        return CalibrationResult(
            blur_threshold=blur_threshold,
            low_light_threshold=low_light_threshold,
            gamma_value=gamma_value,
            sample_count=len(self._blur_scores),
            blur_scores=self._blur_scores.copy(),
            luminance_values=self._luminance_values.copy()
        )
    
    def reset(self) -> None:
        """Reset calibration state for recalibration.
        
        Clears all collected samples and resets the calibration complete flag.
        """
        self._blur_scores.clear()
        self._luminance_values.clear()
        self._calibration_complete = False
        logger.info("Calibration state reset")
