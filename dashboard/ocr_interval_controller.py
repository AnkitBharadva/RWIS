"""
OCR frame interval controller for managing OCR execution frequency.

This module provides the OCRIntervalController class that controls when OCR
should be executed based on a configurable frame interval.

Requirements: 4.2, 4.3, 4.4, 5.2, 5.3
"""


class OCRIntervalController:
    """Controls OCR execution frequency based on frame interval.
    
    This class manages when OCR should run based on a configurable interval.
    For example, with interval=5, OCR runs on frames 0, 5, 10, 15, etc.
    
    Attributes:
        MIN_INTERVAL: Minimum allowed interval value (1)
        MAX_INTERVAL: Maximum allowed interval value (30)
        DEFAULT_INTERVAL: Default interval value (5)
        
    Requirements:
        - 4.2: Slider allows values from 1 to 30
        - 4.3: Default OCR frame interval is 5
        - 4.4: OCR runs only on every Nth frame
        - 5.2: Status shows "ACTIVE" when OCR is performed
        - 5.3: Status shows "SKIPPED (frame N of M)" when skipped
    """
    
    MIN_INTERVAL = 1
    MAX_INTERVAL = 30
    DEFAULT_INTERVAL = 5
    
    def __init__(self, interval: int = DEFAULT_INTERVAL):
        """Initialize the OCR interval controller.
        
        Args:
            interval: Frame interval for OCR execution (1-30).
                     OCR runs every Nth frame. Default is 5.
                     
        Requirements:
            - 4.3: Default OCR frame interval is 5
        """
        self._interval = self._clamp_interval(interval)
    
    @property
    def interval(self) -> int:
        """Get the current OCR frame interval."""
        return self._interval
    
    @interval.setter
    def interval(self, value: int) -> None:
        """Set the OCR frame interval.
        
        Args:
            value: New interval value (will be clamped to 1-30 range)
        """
        self._interval = self._clamp_interval(value)
    
    def _clamp_interval(self, value: int) -> int:
        """Clamp interval value to valid range.
        
        Args:
            value: Interval value to clamp
            
        Returns:
            Clamped value between MIN_INTERVAL and MAX_INTERVAL
        """
        return max(self.MIN_INTERVAL, min(self.MAX_INTERVAL, value))
    
    def should_run_ocr(self, frame_index: int) -> bool:
        """Determine if OCR should run on the given frame.
        
        OCR runs when frame_index is divisible by the interval.
        For interval=5: runs on frames 0, 5, 10, 15, etc.
        
        Args:
            frame_index: Current frame index (0-based)
            
        Returns:
            True if OCR should run on this frame, False otherwise
            
        Requirements:
            - 4.4: OCR runs only on every Nth frame
            - 4.6: OCR is skipped due to interval on non-matching frames
        """
        return frame_index % self._interval == 0
    
    def get_status_text(self, frame_index: int) -> str:
        """Get the OCR status text for the given frame.
        
        Returns "ACTIVE" when OCR runs, or "SKIPPED (frame N of M)"
        when OCR is skipped due to interval.
        
        Args:
            frame_index: Current frame index (0-based)
            
        Returns:
            Status text string
            
        Requirements:
            - 5.2: Status shows "ACTIVE" when OCR is performed
            - 5.3: Status shows "SKIPPED (frame N of M)" when skipped
        """
        if self.should_run_ocr(frame_index):
            return "ACTIVE"
        else:
            # Calculate position within the interval cycle
            position_in_cycle = (frame_index % self._interval) + 1
            return f"SKIPPED (frame {position_in_cycle} of {self._interval})"
