"""
OCR visualization module for drawing bounding boxes and text overlays on video frames.

This module provides the OCRVisualization class that handles drawing OCR detection
bounding boxes and text overlays with confidence scores on video frames.

Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 2.5
"""

from typing import List, Tuple
import cv2
import numpy as np

from utils.data_models import OCRDetection, BoundingBox


class OCRVisualization:
    """Handles OCR visualization on video frames.
    
    This class draws OCR bounding boxes and text overlays on video frames,
    with support for confidence-based coloring and ROI-to-frame coordinate
    transformation.
    
    Attributes:
        OCR_BOX_COLOR_NORMAL: BGR color for normal OCR boxes (cyan)
        OCR_BOX_COLOR_DEBLURRED: BGR color for deblurred OCR boxes (bright cyan)
        OCR_TEXT_COLOR_NORMAL: BGR color for normal confidence text (white)
        OCR_TEXT_COLOR_WARNING: BGR color for low confidence text (orange)
        CONFIDENCE_WARNING_THRESHOLD: Threshold below which warning color is used
        BOX_THICKNESS: Line thickness for bounding boxes
        LABEL_FONT: OpenCV font for text labels
        LABEL_SCALE: Font scale for labels
        LABEL_THICKNESS: Line thickness for label text
        PADDING: Padding around label text
        BACKGROUND_ALPHA: Alpha value for semi-transparent background
    
    Requirements:
        - 1.1: Draw bounding box around each detected text region
        - 1.2: Display OCR boxes in distinct color (cyan)
        - 1.3: Draw separate bounding boxes for each region
        - 1.4: Adjust coordinates from ROI-relative to frame-absolute
        - 2.1: Display extracted text as overlay near bounding box
        - 2.2: Include confidence score in percentage format
        - 2.3: Semi-transparent background for readability
        - 2.4: Position overlay to avoid obscuring detected region
        - 2.5: Warning color for low confidence detections
    """
    
    # Color constants (BGR format)
    OCR_BOX_COLOR_NORMAL = (255, 255, 0)      # Cyan
    OCR_BOX_COLOR_DEBLURRED = (255, 200, 0)   # Bright cyan
    OCR_TEXT_COLOR_NORMAL = (255, 255, 255)   # White
    OCR_TEXT_COLOR_WARNING = (0, 165, 255)    # Orange (low confidence)
    CONFIDENCE_WARNING_THRESHOLD = 0.5
    
    BOX_THICKNESS = 2
    LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
    LABEL_SCALE = 0.5
    LABEL_THICKNESS = 1
    PADDING = 4
    BACKGROUND_ALPHA = 0.7
    MAX_TEXT_LENGTH = 50
    ELLIPSIS = "..."
    
    def truncate_text(self, text: str) -> str:
        """Truncate text to 50 characters with ellipsis if longer.
        
        Args:
            text: The text to truncate
            
        Returns:
            Original text if 50 characters or less, otherwise truncated
            to 50 characters followed by ellipsis
            
        Requirements:
            - 5.4: Last detected text truncated to 50 characters with ellipsis
        """
        if len(text) <= self.MAX_TEXT_LENGTH:
            return text
        return text[:self.MAX_TEXT_LENGTH] + self.ELLIPSIS
    
    def draw_ocr_boxes(
        self,
        frame: np.ndarray,
        ocr_results: List[OCRDetection],
        wagon_bbox: BoundingBox,
        deblur_applied: bool = False
    ) -> np.ndarray:
        """Draw OCR bounding boxes on frame with ROI-to-frame coordinate adjustment.
        
        Args:
            frame: BGR frame to annotate (will be copied)
            ocr_results: List of OCR detections with ROI-relative coordinates
            wagon_bbox: Wagon bounding box for coordinate transformation
            deblur_applied: Whether deblurring was applied (affects box color)
            
        Returns:
            Annotated frame with OCR bounding boxes and text overlays
            
        Requirements:
            - 1.1: Draw bounding box around each detected text region
            - 1.2: Display OCR boxes in distinct color (cyan)
            - 1.3: Draw separate bounding boxes for each region
            - 1.4: Adjust coordinates from ROI-relative to frame-absolute
        """
        if frame is None or len(frame.shape) < 2:
            return frame
        
        if not ocr_results:
            return frame
        
        # Create a copy to avoid modifying the original
        annotated_frame = frame.copy()
        
        # Select box color based on deblur status
        box_color = self.OCR_BOX_COLOR_DEBLURRED if deblur_applied else self.OCR_BOX_COLOR_NORMAL
        
        for detection in ocr_results:
            # Adjust coordinates from ROI-relative to frame-absolute
            adjusted_bbox = self.adjust_coordinates(detection.bbox, wagon_bbox)
            
            # Draw bounding box
            cv2.rectangle(
                annotated_frame,
                (adjusted_bbox.x1, adjusted_bbox.y1),
                (adjusted_bbox.x2, adjusted_bbox.y2),
                box_color,
                self.BOX_THICKNESS
            )
            
            # Draw text overlay
            annotated_frame = self.draw_text_overlay(
                annotated_frame,
                detection.text,
                detection.confidence,
                (adjusted_bbox.x1, adjusted_bbox.y1),
                deblur_applied
            )
        
        return annotated_frame
    
    def draw_text_overlay(
        self,
        frame: np.ndarray,
        text: str,
        confidence: float,
        position: Tuple[int, int],
        deblur_applied: bool = False
    ) -> np.ndarray:
        """Draw text overlay with semi-transparent background.
        
        Args:
            frame: BGR frame to annotate (modified in place)
            text: Detected text to display
            confidence: OCR confidence score [0.0, 1.0]
            position: (x, y) position for the overlay (top-left of bbox)
            deblur_applied: Whether deblurring was applied
            
        Returns:
            Frame with text overlay added
            
        Requirements:
            - 2.1: Display extracted text as overlay near bounding box
            - 2.2: Include confidence score in percentage format
            - 2.3: Semi-transparent background for readability
            - 2.4: Position overlay to avoid obscuring detected region
            - 2.5: Warning color for low confidence detections
        """
        if frame is None:
            return frame
        
        # Format label text with confidence percentage
        confidence_pct = int(confidence * 100)
        label_text = f"{text} ({confidence_pct}%)"
        
        # Select text color based on confidence
        text_color = (
            self.OCR_TEXT_COLOR_WARNING 
            if confidence < self.CONFIDENCE_WARNING_THRESHOLD 
            else self.OCR_TEXT_COLOR_NORMAL
        )
        
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text,
            self.LABEL_FONT,
            self.LABEL_SCALE,
            self.LABEL_THICKNESS
        )
        
        # Position overlay above the bounding box to avoid obscuring detected region
        x, y = position
        label_y = y - self.PADDING - 2  # Position above the box
        
        # If label would be above frame, position it below the box
        if label_y - text_height - self.PADDING < 0:
            label_y = y + text_height + self.PADDING + 2
        
        # Calculate background rectangle coordinates
        bg_x1 = x
        bg_y1 = label_y - text_height - self.PADDING
        bg_x2 = x + text_width + 2 * self.PADDING
        bg_y2 = label_y + self.PADDING
        
        # Ensure background stays within frame bounds
        frame_height, frame_width = frame.shape[:2]
        bg_x1 = max(0, bg_x1)
        bg_y1 = max(0, bg_y1)
        bg_x2 = min(frame_width, bg_x2)
        bg_y2 = min(frame_height, bg_y2)
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (bg_x1, bg_y1),
            (bg_x2, bg_y2),
            (0, 0, 0),  # Black background
            cv2.FILLED
        )
        cv2.addWeighted(overlay, self.BACKGROUND_ALPHA, frame, 1 - self.BACKGROUND_ALPHA, 0, frame)
        
        # Draw text
        text_x = x + self.PADDING
        text_y = label_y
        
        # Ensure text position is within bounds
        text_x = max(self.PADDING, min(text_x, frame_width - text_width - self.PADDING))
        text_y = max(text_height + self.PADDING, min(text_y, frame_height - self.PADDING))
        
        cv2.putText(
            frame,
            label_text,
            (text_x, text_y),
            self.LABEL_FONT,
            self.LABEL_SCALE,
            text_color,
            self.LABEL_THICKNESS,
            cv2.LINE_AA
        )
        
        return frame
    
    def adjust_coordinates(
        self,
        ocr_bbox: BoundingBox,
        wagon_bbox: BoundingBox
    ) -> BoundingBox:
        """Convert ROI-relative coordinates to frame-absolute coordinates.
        
        The OCR detection bounding box is relative to the wagon ROI.
        This method transforms it to absolute frame coordinates by adding
        the wagon bounding box offset.
        
        Args:
            ocr_bbox: OCR bounding box with ROI-relative coordinates
            wagon_bbox: Wagon bounding box with frame-absolute coordinates
            
        Returns:
            BoundingBox with frame-absolute coordinates
            
        Requirements:
            - 1.4: Adjust coordinates from ROI-relative to frame-absolute
            
        Formula:
            frame_x = wagon_x1 + roi_x
            frame_y = wagon_y1 + roi_y
        """
        return BoundingBox(
            x1=wagon_bbox.x1 + ocr_bbox.x1,
            y1=wagon_bbox.y1 + ocr_bbox.y1,
            x2=wagon_bbox.x1 + ocr_bbox.x2,
            y2=wagon_bbox.y1 + ocr_bbox.y2
        )
