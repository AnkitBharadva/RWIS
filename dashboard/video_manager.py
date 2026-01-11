"""
Video capture manager for the Mission Control Dashboard.

Handles RTSP streams and video file playback with frame-skipping
for UI responsiveness.
"""

import cv2
import numpy as np
from typing import Optional, Tuple


class VideoManager:
    """
    Manages video capture with frame-skipping for UI responsiveness.
    
    Attributes:
        frame_skip: Number of frames to skip between processed frames (default 3)
        cap: OpenCV VideoCapture instance
        frame_count: Counter for frame-skipping logic
    """
    
    def __init__(self, frame_skip: int = 3):
        """
        Initialize video manager with frame skip interval.
        
        Args:
            frame_skip: Process every Nth frame (default 3)
        """
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_skip = max(1, frame_skip)  # Ensure at least 1
        self.frame_count = 0
    
    def connect(self, source: str) -> bool:
        """
        Connect to video source (RTSP URL or file path).
        
        Args:
            source: RTSP URL or path to video file
            
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Release any existing capture
            self.release()
            
            self.cap = cv2.VideoCapture(source)
            self.frame_count = 0
            
            return self.cap.isOpened()
        except Exception:
            self.cap = None
            return False
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read frame with frame-skipping logic.
        
        Only returns a frame every N frames (based on frame_skip setting).
        Intermediate frames are grabbed but not decoded for efficiency.
        
        Returns:
            Tuple of (success, frame) where frame is None if skipped or failed
        """
        if not self.is_connected():
            return False, None
        
        # Grab frames until we reach the skip interval
        for _ in range(self.frame_skip - 1):
            grabbed = self.cap.grab()
            if not grabbed:
                return False, None
            self.frame_count += 1
        
        # Read and decode the Nth frame
        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
        
        return ret, frame if ret else None
    
    def release(self) -> None:
        """
        Gracefully release video capture resources.
        """
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.frame_count = 0
    
    def is_connected(self) -> bool:
        """
        Check if video source is connected.
        
        Returns:
            True if connected and capture is open, False otherwise
        """
        return self.cap is not None and self.cap.isOpened()
