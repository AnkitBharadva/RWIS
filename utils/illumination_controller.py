"""Illumination Controller for image brightness adjustment.

This module provides gamma correction-based illumination control with
support for auto-calibration and manual mode switching.

Feature: ocr-enhancement-improvements
Validates: Requirements 5.4, 8.3, 8.4, 8.5
"""

import logging
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from pipelines.calibration_manager import CalibrationResult

logger = logging.getLogger(__name__)


@dataclass
class IlluminationSettings:
    """Illumination processing settings.
    
    Attributes:
        gamma_value: Gamma correction value (< 1 brightens, > 1 darkens)
        low_light_threshold: Luminance threshold for low-light detection (0-255)
        auto_mode: Whether to use auto-calibrated values
        enabled: Whether illumination adjustment is enabled
    """
    gamma_value: float = 1.0
    low_light_threshold: int = 80
    auto_mode: bool = True
    enabled: bool = True


class IlluminationController:
    """Controls image illumination with gamma correction.
    
    The controller provides:
    - Gamma correction for brightness adjustment
    - Low-light detection based on mean luminance
    - Auto/manual mode switching
    - Lookup table caching for performance
    
    Gamma correction formula: output = input^gamma
    - gamma < 1: brightens the image (values increase)
    - gamma > 1: darkens the image (values decrease)
    - gamma = 1: no change
    
    Attributes:
        DEFAULT_GAMMA: Default gamma value (1.0 = no change)
        DEFAULT_LOW_LIGHT_THRESHOLD: Default threshold for low-light detection
        MIN_GAMMA: Minimum allowed gamma value
        MAX_GAMMA: Maximum allowed gamma value
    """
    
    DEFAULT_GAMMA = 1.0
    DEFAULT_LOW_LIGHT_THRESHOLD = 80
    MIN_GAMMA = 0.1
    MAX_GAMMA = 5.0
    
    def __init__(
        self,
        gamma_value: float = DEFAULT_GAMMA,
        low_light_threshold: int = DEFAULT_LOW_LIGHT_THRESHOLD,
        auto_mode: bool = True
    ):
        """Initialize illumination controller.
        
        Args:
            gamma_value: Initial gamma value (< 1 brightens, > 1 darkens)
            low_light_threshold: Luminance threshold for low-light detection (0-255)
            auto_mode: Whether to use auto-calibrated values
            
        Raises:
            ValueError: If gamma_value is not positive
            ValueError: If low_light_threshold is not in range [0, 255]
        """
        if gamma_value <= 0:
            raise ValueError(f"gamma_value must be positive, got {gamma_value}")
        if not 0 <= low_light_threshold <= 255:
            raise ValueError(f"low_light_threshold must be in [0, 255], got {low_light_threshold}")
        
        self._gamma_value = gamma_value
        self._low_light_threshold = low_light_threshold
        self._auto_mode = auto_mode
        self._enabled = True
        
        # Lookup table cache for gamma correction
        self._gamma_luts: dict[float, np.ndarray] = {}
    
    def _get_gamma_lut(self, gamma: float) -> np.ndarray:
        """Get or create gamma lookup table.
        
        Caches lookup tables for frequently used gamma values to improve
        performance. Tables are keyed by gamma rounded to 2 decimal places.
        
        Uses the standard gamma correction formula: output = input^gamma
        - gamma < 1: brightens the image (values increase)
        - gamma > 1: darkens the image (values decrease)
        
        Args:
            gamma: Gamma value for the lookup table
            
        Returns:
            Lookup table for gamma correction (256 uint8 values)
        """
        # Round gamma to 2 decimal places for caching
        gamma_key = round(gamma, 2)
        
        if gamma_key not in self._gamma_luts:
            # Standard gamma correction formula: output = input^gamma
            # gamma < 1 brightens, gamma > 1 darkens
            table = np.array([
                ((i / 255.0) ** gamma) * 255
                for i in range(256)
            ]).astype(np.uint8)
            self._gamma_luts[gamma_key] = table
        
        return self._gamma_luts[gamma_key]

    def apply_gamma(self, image: np.ndarray, gamma: Optional[float] = None) -> np.ndarray:
        """Apply gamma correction to image.
        
        Uses the standard gamma correction formula: output = input^gamma
        - gamma < 1: brightens the image (values increase)
        - gamma > 1: darkens the image (values decrease)
        
        Args:
            image: Input BGR image as numpy array
            gamma: Gamma value to apply. If None, uses the controller's gamma_value.
            
        Returns:
            Gamma-corrected image
            
        Raises:
            ValueError: If image is None or empty
            ValueError: If gamma is not positive
        """
        if image is None or image.size == 0:
            raise ValueError("Image cannot be None or empty")
        
        gamma_to_use = gamma if gamma is not None else self._gamma_value
        
        if gamma_to_use <= 0:
            raise ValueError(f"gamma must be positive, got {gamma_to_use}")
        
        # If gamma is 1.0, no change needed
        if abs(gamma_to_use - 1.0) < 1e-6:
            return image.copy()
        
        lut = self._get_gamma_lut(gamma_to_use)
        return cv2.LUT(image, lut)
    
    def increase_illumination(self, image: np.ndarray, amount: float = 0.1) -> np.ndarray:
        """Increase image brightness by reducing gamma.
        
        Applies gamma correction with gamma < 1 to brighten the image.
        The amount parameter controls how much to reduce gamma from 1.0.
        
        Args:
            image: Input BGR image as numpy array
            amount: Amount to reduce gamma (0.0 to 0.9). Higher values = brighter.
                   The resulting gamma will be (1.0 - amount), clamped to MIN_GAMMA.
            
        Returns:
            Brightened image
            
        Raises:
            ValueError: If image is None or empty
            ValueError: If amount is not in range [0.0, 0.9]
        """
        if image is None or image.size == 0:
            raise ValueError("Image cannot be None or empty")
        
        if not 0.0 <= amount <= 0.9:
            raise ValueError(f"amount must be in [0.0, 0.9], got {amount}")
        
        # Calculate gamma: lower gamma = brighter image
        gamma = max(self.MIN_GAMMA, 1.0 - amount)
        
        logger.debug(f"Increasing illumination with gamma={gamma:.2f}")
        return self.apply_gamma(image, gamma)
    
    def decrease_illumination(self, image: np.ndarray, amount: float = 0.1) -> np.ndarray:
        """Decrease image brightness by increasing gamma.
        
        Applies gamma correction with gamma > 1 to darken the image.
        The amount parameter controls how much to increase gamma from 1.0.
        
        Args:
            image: Input BGR image as numpy array
            amount: Amount to increase gamma (0.0 to 4.0). Higher values = darker.
                   The resulting gamma will be (1.0 + amount), clamped to MAX_GAMMA.
            
        Returns:
            Darkened image
            
        Raises:
            ValueError: If image is None or empty
            ValueError: If amount is negative
        """
        if image is None or image.size == 0:
            raise ValueError("Image cannot be None or empty")
        
        if amount < 0:
            raise ValueError(f"amount must be non-negative, got {amount}")
        
        # Calculate gamma: higher gamma = darker image
        gamma = min(self.MAX_GAMMA, 1.0 + amount)
        
        logger.debug(f"Decreasing illumination with gamma={gamma:.2f}")
        return self.apply_gamma(image, gamma)
    
    def is_low_light(self, image: np.ndarray) -> bool:
        """Check if image is low-light based on mean luminance.
        
        Converts the image to grayscale and computes mean luminance.
        Returns True if mean luminance is below the low_light_threshold.
        
        Args:
            image: Input BGR image as numpy array
            
        Returns:
            True if image is considered low-light
        """
        if image is None or image.size == 0:
            return False
        
        mean_luminance = self.get_mean_luminance(image)
        return mean_luminance < self._low_light_threshold
    
    def get_mean_luminance(self, image: np.ndarray) -> float:
        """Get mean luminance of image.
        
        Converts the image to grayscale and computes the mean pixel value.
        
        Args:
            image: Input BGR image as numpy array (H, W, 3) or grayscale (H, W)
            
        Returns:
            Mean luminance value (0-255)
        """
        if image is None or image.size == 0:
            return 0.0
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        return float(np.mean(gray))

    def get_settings(self) -> IlluminationSettings:
        """Get current illumination settings.
        
        Returns:
            IlluminationSettings dataclass with current values
        """
        return IlluminationSettings(
            gamma_value=self._gamma_value,
            low_light_threshold=self._low_light_threshold,
            auto_mode=self._auto_mode,
            enabled=self._enabled
        )
    
    def set_settings(self, settings: IlluminationSettings) -> None:
        """Set illumination settings.
        
        Args:
            settings: IlluminationSettings dataclass with new values
            
        Raises:
            ValueError: If gamma_value is not positive
            ValueError: If low_light_threshold is not in range [0, 255]
        """
        if settings.gamma_value <= 0:
            raise ValueError(f"gamma_value must be positive, got {settings.gamma_value}")
        if not 0 <= settings.low_light_threshold <= 255:
            raise ValueError(f"low_light_threshold must be in [0, 255], got {settings.low_light_threshold}")
        
        self._gamma_value = settings.gamma_value
        self._low_light_threshold = settings.low_light_threshold
        self._auto_mode = settings.auto_mode
        self._enabled = settings.enabled
        
        logger.info(
            f"Illumination settings updated: gamma={self._gamma_value:.2f}, "
            f"threshold={self._low_light_threshold}, auto_mode={self._auto_mode}"
        )
    
    def set_auto_mode(self, enabled: bool) -> None:
        """Enable or disable auto mode.
        
        When auto mode is enabled, the controller uses auto-calibrated values.
        When disabled, it uses manually set values.
        
        Args:
            enabled: True to enable auto mode, False for manual mode
        """
        self._auto_mode = enabled
        logger.info(f"Illumination auto mode {'enabled' if enabled else 'disabled'}")
    
    def is_auto_mode(self) -> bool:
        """Check if auto mode is enabled.
        
        Returns:
            True if auto mode is enabled
        """
        return self._auto_mode
    
    def set_gamma_value(self, gamma: float) -> None:
        """Set gamma value.
        
        Args:
            gamma: New gamma value (must be positive)
            
        Raises:
            ValueError: If gamma is not positive
        """
        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma}")
        
        self._gamma_value = gamma
        logger.debug(f"Gamma value set to {gamma:.2f}")
    
    def get_gamma_value(self) -> float:
        """Get current gamma value.
        
        Returns:
            Current gamma value
        """
        return self._gamma_value
    
    def set_low_light_threshold(self, threshold: int) -> None:
        """Set low-light threshold.
        
        Args:
            threshold: New threshold value (0-255)
            
        Raises:
            ValueError: If threshold is not in range [0, 255]
        """
        if not 0 <= threshold <= 255:
            raise ValueError(f"threshold must be in [0, 255], got {threshold}")
        
        self._low_light_threshold = threshold
        logger.debug(f"Low-light threshold set to {threshold}")
    
    def get_low_light_threshold(self) -> int:
        """Get current low-light threshold.
        
        Returns:
            Current low-light threshold (0-255)
        """
        return self._low_light_threshold
    
    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable illumination adjustment.
        
        Args:
            enabled: True to enable, False to disable
        """
        self._enabled = enabled
        logger.info(f"Illumination adjustment {'enabled' if enabled else 'disabled'}")
    
    def is_enabled(self) -> bool:
        """Check if illumination adjustment is enabled.
        
        Returns:
            True if enabled
        """
        return self._enabled
    
    def update_from_calibration(self, result: "CalibrationResult") -> None:
        """Update settings from calibration result.
        
        Only updates settings if auto mode is enabled.
        
        Args:
            result: CalibrationResult from CalibrationManager
        """
        if not self._auto_mode:
            logger.debug("Auto mode disabled, skipping calibration update")
            return
        
        self._gamma_value = result.gamma_value
        self._low_light_threshold = result.low_light_threshold
        
        logger.info(
            f"Illumination updated from calibration: gamma={self._gamma_value:.2f}, "
            f"threshold={self._low_light_threshold}"
        )
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """Process image with illumination adjustment.
        
        If enabled and the image is low-light, applies gamma correction
        to brighten the image.
        
        Args:
            image: Input BGR image as numpy array
            
        Returns:
            Processed image (may be unchanged if not low-light or disabled)
        """
        if not self._enabled:
            return image
        
        if image is None or image.size == 0:
            return image
        
        # Check if low-light and apply gamma correction
        if self.is_low_light(image):
            logger.debug(f"Low-light detected, applying gamma={self._gamma_value:.2f}")
            return self.apply_gamma(image, self._gamma_value)
        
        return image
