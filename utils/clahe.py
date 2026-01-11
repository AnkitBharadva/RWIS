"""CLAHE (Contrast Limited Adaptive Histogram Equalization) enhancement utility.

This module provides CLAHE enhancement that operates only on the L-channel
in LAB color space, preserving color information while improving contrast.

Enhanced for low-light conditions with adaptive gamma correction.
"""

from typing import Tuple

import cv2
import numpy as np


class CLAHEEnhancer:
    """Applies CLAHE enhancement to the L-channel only in LAB color space.
    
    This ensures that color information (A and B channels) remains unchanged
    while improving luminance contrast for better detection performance.
    
    Enhanced features:
    - L-channel only CLAHE (preserves color)
    - Optional adaptive gamma correction for low-light
    - Configurable enhancement strength
    
    Attributes:
        clip_limit: Threshold for contrast limiting (default: 2.0)
        tile_grid_size: Size of grid for histogram equalization (default: (8, 8))
        enable_gamma: Whether to apply adaptive gamma for low-light (default: True)
        gamma_threshold: Mean luminance below which gamma is applied (default: 80)
    """
    
    def __init__(
        self, 
        clip_limit: float = 2.0, 
        tile_grid_size: Tuple[int, int] = (8, 8),
        enable_gamma: bool = True,
        gamma_threshold: int = 80
    ):
        """Initialize CLAHE enhancer with parameters.
        
        Args:
            clip_limit: Threshold for contrast limiting. Higher values give more contrast.
            tile_grid_size: Size of grid for histogram equalization. Smaller tiles
                give more local contrast enhancement.
            enable_gamma: Whether to apply adaptive gamma for low-light images.
            gamma_threshold: Mean luminance below which gamma correction is applied.
        """
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.enable_gamma = enable_gamma
        self.gamma_threshold = gamma_threshold
        self._clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size
        )
        
        # Pre-compute gamma lookup tables for common gamma values
        self._gamma_luts = {}
    
    def _get_gamma_lut(self, gamma: float) -> np.ndarray:
        """Get or create gamma lookup table.
        
        Args:
            gamma: Gamma value (< 1 brightens, > 1 darkens)
            
        Returns:
            Lookup table for gamma correction
        """
        # Round gamma to 2 decimal places for caching
        gamma_key = round(gamma, 2)
        
        if gamma_key not in self._gamma_luts:
            inv_gamma = 1.0 / gamma
            table = np.array([
                ((i / 255.0) ** inv_gamma) * 255
                for i in np.arange(0, 256)
            ]).astype("uint8")
            self._gamma_luts[gamma_key] = table
        
        return self._gamma_luts[gamma_key]
    
    def _apply_adaptive_gamma(self, l_channel: np.ndarray) -> np.ndarray:
        """Apply adaptive gamma correction based on mean luminance.
        
        For low-light images (mean luminance below threshold), applies
        gamma correction to brighten the image before CLAHE.
        
        Args:
            l_channel: L channel from LAB color space
            
        Returns:
            Gamma-corrected L channel (or original if not needed)
        """
        mean_luminance = np.mean(l_channel)
        
        if mean_luminance < self.gamma_threshold:
            # Calculate adaptive gamma: lower luminance = lower gamma (more brightening)
            # Gamma range: 0.4 (very dark) to 1.0 (threshold)
            gamma = 0.4 + (mean_luminance / self.gamma_threshold) * 0.6
            gamma = max(0.4, min(1.0, gamma))
            
            lut = self._get_gamma_lut(gamma)
            return cv2.LUT(l_channel, lut)
        
        return l_channel
    
    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """Apply CLAHE enhancement to L-channel only in LAB color space.
        
        The enhancement process:
        1. Convert BGR to LAB color space
        2. Optionally apply adaptive gamma to L channel (for low-light)
        3. Apply CLAHE to the L (luminance) channel only
        4. Convert back to BGR color space
        
        This preserves the A and B channels (color information) while
        improving contrast in the luminance channel.
        
        Args:
            frame: Input BGR image as numpy array with shape (H, W, 3)
            
        Returns:
            Enhanced BGR image with same shape as input
            
        Raises:
            ValueError: If input frame is not a valid BGR image
        """
        if frame is None or frame.size == 0:
            raise ValueError("Input frame cannot be None or empty")
        
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            raise ValueError("Input frame must be a 3-channel BGR image")
        
        # Convert BGR to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        # Split into L, A, B channels
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Apply adaptive gamma for low-light images (before CLAHE)
        if self.enable_gamma:
            l_channel = self._apply_adaptive_gamma(l_channel)
        
        # Apply CLAHE to L-channel only
        l_enhanced = self._clahe.apply(l_channel)
        
        # Merge channels back (A and B unchanged)
        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        
        # Convert back to BGR
        enhanced_frame = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced_frame
    
    def enhance_roi(self, roi: np.ndarray, for_ocr: bool = False) -> np.ndarray:
        """Apply CLAHE enhancement to an ROI, with optional OCR optimization.
        
        For OCR, applies stronger enhancement to improve text visibility.
        
        Args:
            roi: Input BGR ROI image as numpy array with shape (H, W, 3)
            for_ocr: If True, applies stronger enhancement for OCR
            
        Returns:
            Enhanced BGR ROI with same shape as input
            
        Raises:
            ValueError: If input ROI is not a valid BGR image
        """
        if roi is None or roi.size == 0:
            raise ValueError("Input ROI cannot be None or empty")
        
        # Handle grayscale input
        if len(roi.shape) == 2:
            # Convert grayscale to BGR for processing
            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
            was_grayscale = True
        else:
            was_grayscale = False
        
        if len(roi.shape) != 3 or roi.shape[2] != 3:
            raise ValueError("Input ROI must be a 3-channel BGR image or grayscale")
        
        # For OCR, use higher clip limit for more contrast
        if for_ocr:
            ocr_clahe = cv2.createCLAHE(
                clipLimit=4.0,  # Higher clip limit for OCR
                tileGridSize=(4, 4)  # Smaller tiles for more local enhancement
            )
        else:
            ocr_clahe = self._clahe
        
        # Convert BGR to LAB color space
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        
        # Split into L, A, B channels
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Apply adaptive gamma for low-light ROIs
        if self.enable_gamma:
            l_channel = self._apply_adaptive_gamma(l_channel)
        
        # Apply CLAHE to L-channel only
        l_enhanced = ocr_clahe.apply(l_channel)
        
        # Merge channels back (A and B unchanged)
        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        
        # Convert back to BGR
        enhanced_roi = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        # Convert back to grayscale if input was grayscale
        if was_grayscale:
            enhanced_roi = cv2.cvtColor(enhanced_roi, cv2.COLOR_BGR2GRAY)
        
        return enhanced_roi
    
    def is_low_light(self, frame: np.ndarray) -> bool:
        """Check if a frame is low-light based on mean luminance.
        
        Args:
            frame: Input BGR image
            
        Returns:
            True if frame is considered low-light
        """
        if frame is None or frame.size == 0:
            return False
        
        if len(frame.shape) == 2:
            # Grayscale
            mean_luminance = np.mean(frame)
        else:
            # Convert to LAB and check L channel
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
            mean_luminance = np.mean(l_channel)
        
        return mean_luminance < self.gamma_threshold
