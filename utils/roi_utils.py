"""ROI (Region of Interest) utility functions for the Railway Wagon Inspection Pipeline.

This module provides functions for extracting, validating, and scaling ROI coordinates
from wagon detections, ensuring all ROIs are valid and within frame boundaries.
"""

from typing import Tuple, Optional
import numpy as np

from utils.data_models import BoundingBox, WagonDetection


def validate_roi(bbox: BoundingBox, frame_width: int, frame_height: int) -> bool:
    """Check if a bounding box represents a valid ROI.
    
    A valid ROI must have:
    - x1 < x2 and y1 < y2 (positive dimensions)
    - All coordinates within frame boundaries
    - Non-zero width and height
    
    Args:
        bbox: The bounding box to validate
        frame_width: Width of the frame in pixels
        frame_height: Height of the frame in pixels
        
    Returns:
        True if the ROI is valid, False otherwise
    """
    # Check positive dimensions (x1 < x2, y1 < y2)
    if bbox.x1 >= bbox.x2 or bbox.y1 >= bbox.y2:
        return False
    
    # Check within frame boundaries
    if bbox.x1 < 0 or bbox.y1 < 0:
        return False
    if bbox.x2 > frame_width or bbox.y2 > frame_height:
        return False
    
    # Check non-zero dimensions (already implied by x1 < x2, y1 < y2, but explicit)
    if bbox.width <= 0 or bbox.height <= 0:
        return False
    
    return True


def extract_roi(
    frame: np.ndarray, 
    bbox: BoundingBox,
    clip_to_bounds: bool = True
) -> Tuple[Optional[np.ndarray], BoundingBox]:
    """Extract a Region of Interest from a frame, clipping to frame boundaries.
    
    Args:
        frame: The source frame (numpy array with shape [H, W, C] or [H, W])
        bbox: The bounding box defining the ROI
        clip_to_bounds: If True, clip coordinates to frame boundaries.
                       If False, return None for out-of-bounds ROIs.
        
    Returns:
        Tuple of (roi_image, clipped_bbox):
        - roi_image: The extracted ROI as a numpy array, or None if invalid
        - clipped_bbox: The actual bounding box used (may be clipped)
    """
    frame_height, frame_width = frame.shape[:2]
    
    if clip_to_bounds:
        # Clip coordinates to frame boundaries
        x1 = max(0, bbox.x1)
        y1 = max(0, bbox.y1)
        x2 = min(frame_width, bbox.x2)
        y2 = min(frame_height, bbox.y2)
        
        clipped_bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
    else:
        clipped_bbox = bbox
    
    # Validate the (possibly clipped) bounding box
    if not validate_roi(clipped_bbox, frame_width, frame_height):
        return None, clipped_bbox
    
    # Extract the ROI
    roi = frame[clipped_bbox.y1:clipped_bbox.y2, clipped_bbox.x1:clipped_bbox.x2]
    
    return roi, clipped_bbox


def scale_roi(
    bbox: BoundingBox, 
    scale_x: float, 
    scale_y: float,
    frame_width: Optional[int] = None,
    frame_height: Optional[int] = None
) -> BoundingBox:
    """Scale ROI coordinates by given factors.
    
    Useful for resizing ROI coordinates when working with different resolution frames.
    
    Args:
        bbox: The original bounding box
        scale_x: Horizontal scale factor
        scale_y: Vertical scale factor
        frame_width: Optional frame width to clip results to
        frame_height: Optional frame height to clip results to
        
    Returns:
        A new BoundingBox with scaled coordinates
    """
    x1 = int(bbox.x1 * scale_x)
    y1 = int(bbox.y1 * scale_y)
    x2 = int(bbox.x2 * scale_x)
    y2 = int(bbox.y2 * scale_y)
    
    # Optionally clip to frame boundaries
    if frame_width is not None:
        x1 = max(0, min(x1, frame_width))
        x2 = max(0, min(x2, frame_width))
    if frame_height is not None:
        y1 = max(0, min(y1, frame_height))
        y2 = max(0, min(y2, frame_height))
    
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


def resize_roi_for_deblur(
    roi: np.ndarray,
    max_width: int = 256
) -> Tuple[np.ndarray, float]:
    """Resize ROI to max_width maintaining aspect ratio for deblurring.
    
    This function resizes the ROI to a maximum width while preserving the
    aspect ratio. It does NOT upscale if the ROI width is already below max_width.
    
    Args:
        roi: The input ROI image (numpy array with shape [H, W, C] or [H, W])
        max_width: Maximum width in pixels (default: 256)
        
    Returns:
        Tuple of (resized_roi, scale_factor):
        - resized_roi: The resized ROI (or original if no resize needed)
        - scale_factor: The scale factor applied (1.0 if no resize)
    """
    import cv2
    
    if roi is None or roi.size == 0:
        return roi, 1.0
    
    height, width = roi.shape[:2]
    
    # Do not upscale if already smaller than max_width
    if width <= max_width:
        return roi, 1.0
    
    # Calculate scale factor to fit within max_width
    scale_factor = max_width / width
    
    # Calculate new dimensions maintaining aspect ratio
    new_width = max_width
    new_height = int(height * scale_factor)
    
    # Ensure minimum height of 1 pixel
    new_height = max(1, new_height)
    
    # Resize using INTER_AREA for downscaling (best quality)
    resized_roi = cv2.resize(roi, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return resized_roi, scale_factor


def extract_roi_from_detection(
    frame: np.ndarray,
    detection: WagonDetection,
    padding: int = 0
) -> Tuple[Optional[np.ndarray], BoundingBox]:
    """Extract ROI from a frame based on a wagon detection.
    
    Convenience function that extracts ROI from a WagonDetection object,
    with optional padding around the detection.
    
    Args:
        frame: The source frame
        detection: The wagon detection containing the bounding box
        padding: Optional padding to add around the detection (in pixels)
        
    Returns:
        Tuple of (roi_image, actual_bbox) where actual_bbox is the clipped bbox used
    """
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
