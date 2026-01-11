"""Deblur Manager for the Railway Wagon Inspection Pipeline.

This module provides the DeblurManager class that manages conditional deblurring
with N-th frame execution logic. It coordinates between blur detection, ROI resizing,
CLAHE enhancement, and MPRNet deblurring to optimize processing resources.

Key features:
- Conditional deblurring based on blur score threshold
- N-th frame execution (runs MPRNet every N frames, caches results)
- ROI resizing before deblurring (max 256px width)
- CLAHE enhancement for low-light ROIs
- Cache management for wagon exit cleanup
- Enable/disable deblurring functionality
- Status tracking for frontend display

Requirements: 5.1, 5.2, 5.3, 6.1, 6.2, 6.3, 6.5, 4.5, 7.2, 7.3, 2.1, 2.5, 2.6
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Dict, Optional, Tuple, TYPE_CHECKING

from utils.roi_utils import resize_roi_for_deblur
from utils.clahe import CLAHEEnhancer
from utils.data_models import DeblurStatus, DeblurStatusType

if TYPE_CHECKING:
    from pipelines.mprnet_wrapper import MPRNetDeblur
    from pipelines.blur_detector import BlurDetector


# Configure module logger
logger = logging.getLogger(__name__)


class DeblurManager:
    """Manages conditional deblurring with N-th frame execution.
    
    This class coordinates the deblurring process by:
    1. Resizing ROIs to max_width before processing
    2. Computing blur scores to determine if deblurring is needed
    3. Running MPRNet only every N frames (caching results for intermediate frames)
    4. Applying CLAHE enhancement for low-light conditions
    5. Managing cache cleanup when wagons exit tracking
    6. Tracking deblur status for frontend display
    7. Supporting enable/disable of deblurring
    
    The N-th frame execution logic ensures that MPRNet doesn't run on every frame,
    reducing GPU load while still providing deblurred ROIs for OCR.
    
    IMPORTANT: Blur score interpretation:
    - HIGH score = SHARP image (skip deblur)
    - LOW score = BLURRY image (apply deblur)
    
    Attributes:
        mprnet: MPRNet deblurring wrapper instance
        blur_detector: Blur detector instance for computing blur scores
        frame_interval: Run MPRNet every N frames (default: 3)
        max_roi_width: Maximum ROI width before deblurring (default: 256)
        clahe: CLAHE enhancer for low-light ROIs
        deblur_enabled: Whether deblurring is enabled (default: True)
    """
    
    def __init__(
        self,
        mprnet: "MPRNetDeblur",
        blur_detector: "BlurDetector",
        frame_interval: int = 3,
        max_roi_width: int = 256,
        enable_clahe: bool = True,
        deblur_enabled: bool = True
    ):
        """Initialize DeblurManager with dependencies and configuration.
        
        Args:
            mprnet: MPRNet deblurring wrapper instance (must be loaded)
            blur_detector: Blur detector instance for blur score computation
            frame_interval: Run MPRNet every N frames. Must be >= 1.
                           Default is 3 (run on frames 0, 3, 6, 9, ...).
            max_roi_width: Maximum ROI width in pixels. ROIs wider than this
                          will be resized before deblurring. Default is 256.
            enable_clahe: Whether to apply CLAHE enhancement for low-light ROIs.
            deblur_enabled: Whether deblurring is enabled. Default is True.
        
        Raises:
            ValueError: If frame_interval < 1 or max_roi_width < 1
        """
        if frame_interval < 1:
            raise ValueError(f"frame_interval must be >= 1, got {frame_interval}")
        if max_roi_width < 1:
            raise ValueError(f"max_roi_width must be >= 1, got {max_roi_width}")
        
        self.mprnet = mprnet
        self.blur_detector = blur_detector
        self.frame_interval = frame_interval
        self.max_roi_width = max_roi_width
        self.enable_clahe = enable_clahe
        self._deblur_enabled = deblur_enabled
        
        # Initialize CLAHE enhancer for low-light ROIs
        self.clahe = CLAHEEnhancer(
            clip_limit=3.0,  # Slightly higher for ROIs
            tile_grid_size=(4, 4),  # Smaller tiles for ROIs
            enable_gamma=True,
            gamma_threshold=80
        )
        
        # Cache for deblurred ROIs per wagon
        # Key: wagon_id, Value: (deblurred_roi, source_frame_index)
        self._cache: Dict[int, Tuple[np.ndarray, int]] = {}
        
        # Frame counter per wagon (for N-th frame logic)
        # Key: wagon_id, Value: frame_count since first detection
        self._frame_counters: Dict[int, int] = {}
        
        # Last deblur status for frontend display
        self._last_status: Optional[DeblurStatus] = None
    
    def set_deblur_enabled(self, enabled: bool) -> None:
        """Enable or disable deblurring.
        
        When disabled, process_roi will skip all deblurring operations
        and return the resized ROI directly.
        
        Args:
            enabled: True to enable deblurring, False to disable
        """
        self._deblur_enabled = enabled
        logger.info(f"Deblurring {'enabled' if enabled else 'disabled'}")
    
    def is_deblur_enabled(self) -> bool:
        """Check if deblurring is enabled.
        
        Returns:
            True if deblurring is enabled, False otherwise
        """
        return self._deblur_enabled
    
    def get_last_status(self) -> Optional[DeblurStatus]:
        """Get status of last deblur operation.
        
        Returns the status from the most recent call to process_roi.
        Useful for frontend display of deblur status.
        
        Returns:
            DeblurStatus object with details of last operation,
            or None if process_roi hasn't been called yet
        """
        return self._last_status
    
    def was_deblur_applied(self) -> bool:
        """Check if deblur was applied to current frame.
        
        Convenience method to check if the last process_roi call
        resulted in deblurring being applied.
        
        Returns:
            True if deblur was applied in last process_roi call,
            False otherwise (including if process_roi hasn't been called)
        """
        if self._last_status is None:
            return False
        return self._last_status.applied
    
    def _enhance_low_light_roi(self, roi: np.ndarray) -> np.ndarray:
        """Apply CLAHE enhancement if ROI is low-light.
        
        Args:
            roi: Input ROI image
            
        Returns:
            Enhanced ROI if low-light, otherwise original ROI
        """
        if not self.enable_clahe:
            return roi
        
        # Check if ROI is low-light
        if self.clahe.is_low_light(roi):
            return self.clahe.enhance_roi(roi, for_ocr=True)
        
        return roi
    
    def _log_deblur_operation(
        self,
        frame_index: int,
        wagon_id: int,
        blur_score_before: float,
        blur_score_after: Optional[float] = None
    ) -> None:
        """Log deblur operation details.
        
        Args:
            frame_index: Current frame index in the video
            wagon_id: Unique ID of the wagon
            blur_score_before: Blur score before deblurring
            blur_score_after: Blur score after deblurring (optional)
        """
        if blur_score_after is not None:
            logger.info(
                f"Deblur applied: frame={frame_index}, wagon_id={wagon_id}, "
                f"blur_score_before={blur_score_before:.2f}, "
                f"blur_score_after={blur_score_after:.2f}"
            )
        else:
            logger.info(
                f"Deblur applied: frame={frame_index}, wagon_id={wagon_id}, "
                f"blur_score_before={blur_score_before:.2f}"
            )
    
    def process_roi(
        self,
        roi: np.ndarray,
        wagon_id: int,
        frame_index: int
    ) -> Tuple[np.ndarray, bool, Optional[int]]:
        """Process ROI with conditional deblurring and enhancement.
        
        The processing logic:
        1. Resize ROI to max_width (maintaining aspect ratio)
        2. If deblurring is disabled, return resized ROI with DISABLED status
        3. Compute blur score on resized ROI
        4. Determine if deblurring is needed based on blur score:
           - HIGH score (>= t2): Sharp image, skip deblurring
           - MODERATE score (t1 <= score < t2): Apply deblurring
           - LOW score (< t1): Too blurry, deblurring won't help
        5. If deblurring needed:
           a. Check N-th frame condition
           b. If frame_count % N == 0: run MPRNet, cache result
           c. Else: return cached result (if available)
        6. Apply CLAHE enhancement for low-light ROIs
        7. Update status tracking for frontend display
        
        Args:
            roi: Input ROI image as numpy array (BGR or grayscale)
            wagon_id: Unique ID of the wagon this ROI belongs to
            frame_index: Current frame index in the video
        
        Returns:
            Tuple of (processed_roi, deblur_applied, source_frame):
            - processed_roi: The processed ROI (resized, possibly deblurred/enhanced)
            - deblur_applied: True if deblurring was applied, False otherwise
            - source_frame: Frame index where deblurring was performed,
                           or None if no deblurring was applied
        
        Raises:
            ValueError: If roi is None or empty
        """
        if roi is None or roi.size == 0:
            raise ValueError("ROI cannot be None or empty")
        
        # Step 1: Resize ROI to max_width
        resized_roi, scale_factor = resize_roi_for_deblur(roi, self.max_roi_width)
        
        # Step 2: Check if deblurring is disabled
        if not self._deblur_enabled:
            # Apply CLAHE enhancement for low-light and return
            enhanced_roi = self._enhance_low_light_roi(resized_roi)
            
            # Update status - DISABLED
            self._last_status = DeblurStatus(
                enabled=False,
                applied=False,
                blur_score_before=0.0,  # Not computed when disabled
                blur_score_after=None,
                status_type=DeblurStatusType.DISABLED
            )
            
            return enhanced_roi, False, None
        
        # Step 3: Compute blur score on resized ROI
        blur_score = self.blur_detector.compute_blur_score(resized_roi)
        
        # Step 4: Check if deblurring is needed using the corrected logic
        # needs_deblur returns True when: t1 <= blur_score < t2 (moderate blur)
        should_deblur = self.blur_detector.needs_deblur(blur_score)
        
        if not should_deblur:
            # Either sharp enough (score >= t2) or too blurry (score < t1)
            # Apply CLAHE enhancement for low-light and return
            enhanced_roi = self._enhance_low_light_roi(resized_roi)
            
            # Update status - SKIPPED
            self._last_status = DeblurStatus(
                enabled=True,
                applied=False,
                blur_score_before=blur_score,
                blur_score_after=None,
                status_type=DeblurStatusType.SKIPPED
            )
            
            return enhanced_roi, False, None
        
        # Step 5: Moderate blur - check N-th frame condition
        # Initialize frame counter for new wagons
        if wagon_id not in self._frame_counters:
            self._frame_counters[wagon_id] = 0
        
        frame_count = self._frame_counters[wagon_id]
        
        # Check if this is an N-th frame (should run MPRNet)
        is_nth_frame = (frame_count % self.frame_interval == 0)
        
        # Increment frame counter for next call
        self._frame_counters[wagon_id] = frame_count + 1
        
        if is_nth_frame:
            # Run MPRNet and cache result
            deblurred_roi = self.mprnet.deblur_roi(resized_roi)
            # Apply CLAHE enhancement after deblurring for low-light
            deblurred_roi = self._enhance_low_light_roi(deblurred_roi)
            self._cache[wagon_id] = (deblurred_roi, frame_index)
            
            # Compute blur score after deblurring (optional, for logging)
            blur_score_after = self.blur_detector.compute_blur_score(deblurred_roi)
            
            # Log the deblur operation
            self._log_deblur_operation(
                frame_index=frame_index,
                wagon_id=wagon_id,
                blur_score_before=blur_score,
                blur_score_after=blur_score_after
            )
            
            # Update status - ACTIVE
            self._last_status = DeblurStatus(
                enabled=True,
                applied=True,
                blur_score_before=blur_score,
                blur_score_after=blur_score_after,
                status_type=DeblurStatusType.ACTIVE
            )
            
            return deblurred_roi, True, frame_index
        else:
            # Return cached result if available
            cached = self.get_cached_roi(wagon_id)
            if cached is not None:
                cached_roi, source_frame = cached
                
                # Update status - ACTIVE (using cache)
                self._last_status = DeblurStatus(
                    enabled=True,
                    applied=True,
                    blur_score_before=blur_score,
                    blur_score_after=None,  # Not recomputed for cached
                    status_type=DeblurStatusType.ACTIVE
                )
                
                return cached_roi, True, source_frame
            else:
                # No cache available (shouldn't happen in normal flow)
                # Fall back to running MPRNet
                deblurred_roi = self.mprnet.deblur_roi(resized_roi)
                deblurred_roi = self._enhance_low_light_roi(deblurred_roi)
                self._cache[wagon_id] = (deblurred_roi, frame_index)
                
                # Compute blur score after deblurring
                blur_score_after = self.blur_detector.compute_blur_score(deblurred_roi)
                
                # Log the deblur operation
                self._log_deblur_operation(
                    frame_index=frame_index,
                    wagon_id=wagon_id,
                    blur_score_before=blur_score,
                    blur_score_after=blur_score_after
                )
                
                # Update status - ACTIVE
                self._last_status = DeblurStatus(
                    enabled=True,
                    applied=True,
                    blur_score_before=blur_score,
                    blur_score_after=blur_score_after,
                    status_type=DeblurStatusType.ACTIVE
                )
                
                return deblurred_roi, True, frame_index
    
    def get_cached_roi(self, wagon_id: int) -> Optional[Tuple[np.ndarray, int]]:
        """Retrieve cached deblurred ROI for a wagon.
        
        Args:
            wagon_id: Unique ID of the wagon
        
        Returns:
            Tuple of (deblurred_roi, source_frame_index) if cached,
            None if no cache exists for this wagon
        """
        return self._cache.get(wagon_id)
    
    def clear_cache(self, wagon_id: int) -> None:
        """Clear cache for a wagon when it exits tracking.
        
        This should be called when a wagon exits the tracking area
        to free memory and prevent stale cache entries.
        
        Args:
            wagon_id: Unique ID of the wagon to clear cache for
        """
        if wagon_id in self._cache:
            del self._cache[wagon_id]
        if wagon_id in self._frame_counters:
            del self._frame_counters[wagon_id]
    
    def get_frame_count(self, wagon_id: int) -> int:
        """Get the current frame count for a wagon.
        
        Useful for debugging and testing the N-th frame logic.
        
        Args:
            wagon_id: Unique ID of the wagon
        
        Returns:
            Current frame count for the wagon, or 0 if not tracked
        """
        return self._frame_counters.get(wagon_id, 0)
    
    def get_cache_size(self) -> int:
        """Get the number of cached ROIs.
        
        Useful for monitoring memory usage.
        
        Returns:
            Number of wagons with cached deblurred ROIs
        """
        return len(self._cache)
    
    def clear_all_caches(self) -> None:
        """Clear all caches and frame counters.
        
        Useful for resetting state between video processing sessions.
        """
        self._cache.clear()
        self._frame_counters.clear()
    
    def get_blur_info(self, roi: np.ndarray) -> dict:
        """Get detailed blur information for an ROI.
        
        Useful for debugging and logging.
        
        Args:
            roi: Input ROI image
            
        Returns:
            Dictionary with blur score, decision, and level
        """
        resized_roi, _ = resize_roi_for_deblur(roi, self.max_roi_width)
        blur_score = self.blur_detector.compute_blur_score(resized_roi)
        
        return {
            'blur_score': blur_score,
            'decision': self.blur_detector.get_blur_decision(blur_score).value,
            'level': self.blur_detector.get_blur_level(blur_score),
            'needs_deblur': self.blur_detector.needs_deblur(blur_score),
            'is_low_light': self.clahe.is_low_light(resized_roi) if self.enable_clahe else False
        }
