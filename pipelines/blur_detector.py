"""Blur detection module for the Railway Wagon Inspection Pipeline.

This module provides blur score computation using Laplacian variance
and blur decision logic based on configurable thresholds.

IMPORTANT: Laplacian variance interpretation:
- HIGH variance = SHARP image (many edges detected)
- LOW variance = BLURRY image (few edges detected)

So the threshold logic is:
- blur_score >= threshold: Image is SHARP enough, skip deblurring
- blur_score < threshold: Image is BLURRY, apply deblurring

Feature: ocr-enhancement-improvements
Validates: Requirements 3.5, 3.6, 6.4
"""

import logging
import cv2
import numpy as np
from typing import TYPE_CHECKING

from utils.data_models import BlurDecision, BlurSettings

if TYPE_CHECKING:
    from pipelines.calibration_manager import CalibrationResult

logger = logging.getLogger(__name__)


class BlurDetector:
    """Detects blur in video frames using Laplacian variance.
    
    CRITICAL: Understanding Laplacian variance:
    - HIGH score = SHARP image (many edges, high variance)
    - LOW score = BLURRY image (few edges, low variance)
    
    The blur decision is based on a single threshold (simplified model):
    - blur_score >= threshold: SKIP_DEBLUR (image is sharp enough)
    - blur_score < threshold: ROI_DEBLUR (image is blurry, apply deblur)
    
    For backward compatibility, T1 and T2 thresholds are still supported:
    - blur_score >= T2: SKIP_DEBLUR (very sharp, definitely skip)
    - T1 <= blur_score < T2: ROI_DEBLUR (moderate blur, apply deblur)
    - blur_score < T1: NO_DEBLUR (extremely blurry, deblur won't help)
    
    Auto-calibration support:
    - When auto_mode is True, thresholds can be updated from CalibrationResult
    - When auto_mode is False, user-specified thresholds are used
    - Switching modes preserves current threshold values
    
    Attributes:
        t1: Lower blur threshold (below this, image is too blurry to recover)
        t2: Upper blur threshold (above this, image is sharp enough)
        auto_mode: Whether auto-calibration mode is enabled
    """
    
    def __init__(self, t1: float, t2: float, auto_mode: bool = True):
        """Initialize BlurDetector with configurable thresholds.
        
        The thresholds define blur score ranges:
        - blur_score >= t2: Sharp image, skip deblurring
        - t1 <= blur_score < t2: Moderate blur, apply deblurring
        - blur_score < t1: Severe blur, deblurring won't help
        
        Args:
            t1: Lower blur threshold. Frames with blur_score < t1 are
                considered too blurry to recover (NO_DEBLUR).
            t2: Upper blur threshold. Frames with blur_score >= t2 are
                considered sharp enough (SKIP_DEBLUR).
            auto_mode: Whether to use auto-calibrated values (default: True)
                
        Raises:
            ValueError: If t1 >= t2 (thresholds must satisfy t1 < t2)
        """
        if t1 >= t2:
            raise ValueError(f"T1 must be less than T2. Got T1={t1}, T2={t2}")
        
        self.t1 = t1
        self.t2 = t2
        self._auto_mode = auto_mode
        
        logger.debug(f"BlurDetector initialized: t1={t1}, t2={t2}, auto_mode={auto_mode}")
    
    def compute_blur_score(self, frame: np.ndarray) -> float:
        """Compute blur score using Laplacian variance.
        
        The Laplacian operator highlights regions of rapid intensity change.
        The variance of the Laplacian response indicates the amount of edges
        in the image:
        - HIGH variance = MORE edges = SHARPER image
        - LOW variance = FEWER edges = BLURRIER image
        
        Args:
            frame: Input frame as BGR numpy array (H, W, 3) or grayscale (H, W)
            
        Returns:
            Blur score as float. HIGHER values indicate SHARPER images.
            Typical ranges:
            - < 50: Very blurry
            - 50-100: Moderately blurry
            - 100-300: Acceptable sharpness
            - > 300: Sharp
            
        Raises:
            ValueError: If frame is None or empty
        """
        if frame is None or frame.size == 0:
            raise ValueError("Frame cannot be None or empty")
        
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Compute Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        # Return variance as blur score
        return float(laplacian.var())
    
    def is_blurry(self, blur_score: float) -> bool:
        """Simple check if image is blurry based on score.
        
        Args:
            blur_score: Computed blur score from compute_blur_score()
            
        Returns:
            True if image is blurry (score < t2), False if sharp
        """
        return blur_score < self.t2
    
    def needs_deblur(self, blur_score: float) -> bool:
        """Check if image needs deblurring (and can benefit from it).
        
        Returns True only if:
        - Image is blurry enough to need deblurring (score < t2)
        - Image is not so blurry that deblurring won't help (score >= t1)
        
        Args:
            blur_score: Computed blur score from compute_blur_score()
            
        Returns:
            True if deblurring should be applied
        """
        return self.t1 <= blur_score < self.t2
    
    def get_blur_decision(self, blur_score: float) -> BlurDecision:
        """Determine blur decision based on score and thresholds.
        
        Decision logic (remember: HIGH score = SHARP, LOW score = BLURRY):
        - blur_score >= T2: SKIP_DEBLUR (image is sharp enough)
        - T1 <= blur_score < T2: ROI_DEBLUR (moderate blur, apply deblur)
        - blur_score < T1: NO_DEBLUR (too blurry, deblur won't help)
        
        Args:
            blur_score: Computed blur score from compute_blur_score()
            
        Returns:
            BlurDecision enum value indicating the processing decision
        """
        if blur_score >= self.t2:
            # Sharp image - no need for deblurring
            return BlurDecision.SKIP_DEBLUR
        elif blur_score >= self.t1:
            # Moderate blur - deblurring can help
            return BlurDecision.ROI_DEBLUR
        else:
            # Severe blur - deblurring won't help much
            return BlurDecision.NO_DEBLUR
    
    def get_blur_level(self, blur_score: float) -> str:
        """Get human-readable blur level description.
        
        Args:
            blur_score: Computed blur score
            
        Returns:
            String description of blur level
        """
        if blur_score >= self.t2:
            return "sharp"
        elif blur_score >= self.t1:
            return "moderate_blur"
        else:
            return "severe_blur"
    
    # Auto/Manual mode methods
    
    def set_threshold(self, threshold: float) -> None:
        """Set the blur threshold (t2 value).
        
        This sets the upper threshold that determines when an image
        is considered sharp enough to skip deblurring.
        
        Args:
            threshold: New threshold value. Must be greater than t1.
            
        Raises:
            ValueError: If threshold <= t1
        """
        if threshold <= self.t1:
            raise ValueError(f"Threshold must be greater than t1={self.t1}. Got {threshold}")
        
        old_threshold = self.t2
        self.t2 = threshold
        logger.debug(f"Blur threshold updated: {old_threshold} -> {threshold}")
    
    def get_threshold(self) -> float:
        """Get the current blur threshold (t2 value).
        
        Returns:
            Current blur threshold value
        """
        return self.t2
    
    def set_auto_mode(self, enabled: bool) -> None:
        """Enable or disable auto-calibration mode.
        
        When switching modes, the current threshold value is preserved.
        
        Args:
            enabled: True to enable auto mode, False for manual mode
        """
        old_mode = self._auto_mode
        self._auto_mode = enabled
        logger.debug(f"Auto mode changed: {old_mode} -> {enabled}")
    
    def is_auto_mode(self) -> bool:
        """Check if auto-calibration mode is enabled.
        
        Returns:
            True if auto mode is enabled, False otherwise
        """
        return self._auto_mode
    
    def update_from_calibration(self, result: "CalibrationResult") -> None:
        """Update threshold from calibration result.
        
        Only updates if auto_mode is enabled. The calibration result's
        blur_threshold is used as the new t2 value.
        
        Args:
            result: CalibrationResult containing computed blur_threshold
        """
        if not self._auto_mode:
            logger.debug("Auto mode disabled, ignoring calibration update")
            return
        
        # Ensure the new threshold is valid (greater than t1)
        new_threshold = result.blur_threshold
        if new_threshold <= self.t1:
            # Adjust t1 to be below the new threshold
            self.t1 = new_threshold * 0.5
            logger.debug(f"Adjusted t1 to {self.t1} to accommodate new threshold")
        
        old_threshold = self.t2
        self.t2 = new_threshold
        logger.info(f"Blur threshold updated from calibration: {old_threshold} -> {new_threshold}")
    
    def get_settings(self) -> BlurSettings:
        """Get current blur detection settings.
        
        Returns:
            BlurSettings with current threshold, auto_mode, and deblur_enabled
        """
        return BlurSettings(
            threshold=self.t2,
            auto_mode=self._auto_mode,
            deblur_enabled=True  # BlurDetector doesn't control deblur enable/disable
        )
    
    def set_settings(self, settings: BlurSettings) -> None:
        """Set blur detection settings.
        
        Updates threshold and auto_mode from the provided settings.
        Note: deblur_enabled is managed by DeblurManager, not BlurDetector.
        
        Args:
            settings: BlurSettings with new values
            
        Raises:
            ValueError: If settings.threshold <= t1
        """
        if settings.threshold <= self.t1:
            raise ValueError(f"Threshold must be greater than t1={self.t1}. Got {settings.threshold}")
        
        self.t2 = settings.threshold
        self._auto_mode = settings.auto_mode
        logger.debug(f"Settings applied: threshold={settings.threshold}, auto_mode={settings.auto_mode}")
