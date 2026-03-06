"""Wagon detector module using YOLOv11n for railway wagon detection.

This module provides the WagonDetector class that uses Ultralytics YOLOv11n
model to detect wagons in video frames. It enforces strict input validation
to ensure only RAW or CLAHE-enhanced frames are processed (never deblurred).

Requirements: 2.1, 2.2, 2.3, 8.3
"""

from typing import List, Optional, Tuple, Set
import numpy as np

from utils.data_models import BoundingBox, WagonDetection
from utils.roi_utils import extract_roi, validate_roi


class FrameProcessingState:
    """Tracks the processing state of frames to prevent deblurred input.
    
    This class maintains a set of frame identifiers that have been deblurred,
    allowing the WagonDetector to reject frames that have undergone deblurring.
    """
    
    def __init__(self):
        """Initialize the frame processing state tracker."""
        self._deblurred_frame_ids: Set[int] = set()
    
    def mark_deblurred(self, frame_id: int) -> None:
        """Mark a frame as having been deblurred.
        
        Args:
            frame_id: Unique identifier for the frame (e.g., id(frame) or frame index)
        """
        self._deblurred_frame_ids.add(frame_id)
    
    def is_deblurred(self, frame_id: int) -> bool:
        """Check if a frame has been deblurred.
        
        Args:
            frame_id: Unique identifier for the frame
            
        Returns:
            True if the frame has been marked as deblurred, False otherwise
        """
        return frame_id in self._deblurred_frame_ids
    
    def clear(self) -> None:
        """Clear all tracked frame states."""
        self._deblurred_frame_ids.clear()
    
    def remove(self, frame_id: int) -> None:
        """Remove a frame from tracking.
        
        Args:
            frame_id: Unique identifier for the frame to remove
        """
        self._deblurred_frame_ids.discard(frame_id)


class WagonDetector:
    """Wagon detector using Ultralytics YOLOv11n model.
    
    This class detects railway wagons in video frames using a YOLO model.
    It enforces strict input validation to ensure only RAW or CLAHE-enhanced
    frames are processed - deblurred frames are rejected.
    
    Attributes:
        model_path: Path to the YOLOv11n model weights
        confidence_threshold: Minimum confidence score for detections
        model: The loaded YOLO model instance
        frame_state: Tracker for frame processing states
    
    Requirements:
        - 2.1: Use Ultralytics YOLOv11n model for wagon detection
        - 2.2: Receive only RAW or CLAHE-enhanced frames
        - 2.3: NOT receive deblurred frames as input
        - 8.3: NOT feed deblurred frames into YOLO detection models
    """
    
    def __init__(
        self, 
        model_path: str, 
        confidence_threshold: float = 0.5,
        frame_state: Optional[FrameProcessingState] = None
    ):
        """Initialize the wagon detector with a YOLO model.
        
        Args:
            model_path: Path to the YOLOv11n model weights file
            confidence_threshold: Minimum confidence threshold for detections [0.0, 1.0]
            frame_state: Optional shared frame state tracker. If None, creates a new one.
            
        Raises:
            FileNotFoundError: If the model file does not exist
            ValueError: If confidence_threshold is not in [0.0, 1.0]
        """
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be between 0.0 and 1.0, got {confidence_threshold}"
            )
        
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.frame_state = frame_state if frame_state is not None else FrameProcessingState()
        self.model = None
        
        # Load the YOLO model
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the YOLOv11n model from the specified path.
        
        Raises:
            FileNotFoundError: If the model file does not exist
            RuntimeError: If the model fails to load
        """
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            # Force CPU mode for sm_120 GPU compatibility
            self.model.to('cpu')
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Wagon detection model not found at: {self.model_path}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load wagon detection model: {e}")
    
    def _validate_frame_not_deblurred(
        self, 
        frame: np.ndarray, 
        frame_id: Optional[int] = None
    ) -> None:
        """Validate that the frame has not been deblurred.
        
        Args:
            frame: The input frame to validate
            frame_id: Optional explicit frame ID. If None, uses id(frame).
            
        Raises:
            ValueError: If the frame has been marked as deblurred
        """
        fid = frame_id if frame_id is not None else id(frame)
        if self.frame_state.is_deblurred(fid):
            raise ValueError(
                "Deblurred frames cannot be passed to WagonDetector. "
                "Only RAW or CLAHE-enhanced frames are allowed. "
                "(Requirements 2.2, 2.3, 8.3)"
            )
    
    def _validate_frame(self, frame: np.ndarray) -> None:
        """Validate that the input frame is valid for detection.
        
        Args:
            frame: The input frame to validate
            
        Raises:
            ValueError: If the frame is invalid (None, empty, wrong type, or deblurred)
        """
        if frame is None:
            raise ValueError("Frame cannot be None")
        
        if not isinstance(frame, np.ndarray):
            raise ValueError(f"Frame must be a numpy array, got {type(frame)}")
        
        if frame.size == 0:
            raise ValueError("Frame cannot be empty")
        
        if len(frame.shape) < 2:
            raise ValueError(f"Frame must have at least 2 dimensions, got shape {frame.shape}")
    
    def detect(
        self, 
        frame: np.ndarray,
        frame_id: Optional[int] = None
    ) -> List[WagonDetection]:
        """Detect wagons in the given frame.
        
        This method runs YOLO inference on the frame and returns a list of
        wagon detections. The frame must be RAW or CLAHE-enhanced only -
        deblurred frames will be rejected.
        
        Args:
            frame: Input frame (BGR format, numpy array). Must be RAW or CLAHE-enhanced.
            frame_id: Optional explicit frame ID for deblur tracking. If None, uses id(frame).
            
        Returns:
            List of WagonDetection objects containing bbox, confidence, and class_id
            
        Raises:
            ValueError: If the frame is invalid or has been deblurred
        """
        # Validate frame
        self._validate_frame(frame)
        self._validate_frame_not_deblurred(frame, frame_id)
        
        # Run YOLO inference
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        detections = []
        for result in results:
            if result.boxes is None:
                continue
                
            boxes = result.boxes
            for i in range(len(boxes)):
                # Get bounding box coordinates (xyxy format)
                xyxy = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                
                # Get confidence and class
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                
                # Create detection object
                bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
                detection = WagonDetection(
                    bbox=bbox,
                    confidence=conf,
                    class_id=cls_id
                )
                detections.append(detection)
        
        return detections
    
    def extract_roi(
        self, 
        frame: np.ndarray, 
        detection: WagonDetection,
        padding: int = 0
    ) -> Tuple[Optional[np.ndarray], BoundingBox]:
        """Extract the wagon ROI from the frame based on a detection.
        
        This method extracts the region of interest for a detected wagon,
        clipping to frame boundaries if necessary.
        
        Args:
            frame: The source frame (BGR format, numpy array)
            detection: The wagon detection containing the bounding box
            padding: Optional padding to add around the detection (in pixels)
            
        Returns:
            Tuple of (roi_image, actual_bbox):
            - roi_image: The extracted ROI as a numpy array, or None if invalid
            - actual_bbox: The actual bounding box used (may be clipped to frame bounds)
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return None, detection.bbox
        
        bbox = detection.bbox
        
        # Apply padding if specified
        if padding > 0:
            padded_bbox = BoundingBox(
                x1=bbox.x1 - padding,
                y1=bbox.y1 - padding,
                x2=bbox.x2 + padding,
                y2=bbox.y2 + padding
            )
            return extract_roi(frame, padded_bbox, clip_to_bounds=True)
        
        return extract_roi(frame, bbox, clip_to_bounds=True)
    
    def mark_frame_deblurred(self, frame: np.ndarray, frame_id: Optional[int] = None) -> None:
        """Mark a frame as having been deblurred.
        
        Call this method after applying deblurring to a frame to prevent
        it from being passed to the detector.
        
        Args:
            frame: The frame that was deblurred
            frame_id: Optional explicit frame ID. If None, uses id(frame).
        """
        fid = frame_id if frame_id is not None else id(frame)
        self.frame_state.mark_deblurred(fid)
    
    def clear_frame_state(self) -> None:
        """Clear all tracked frame processing states."""
        self.frame_state.clear()

