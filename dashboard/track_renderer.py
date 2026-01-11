"""
Track ID renderer for overlaying wagon track IDs on video frames.

This module provides the TrackIDRenderer class that draws track ID labels
on video frames for all tracked wagons, enabling visual identification
and tracking of individual wagons.

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
"""

from typing import List, Tuple
import cv2
import numpy as np

from utils.data_models import TrackedWagon, BoundingBox


class TrackIDRenderer:
    """Renders track IDs on video frames for tracked wagons.
    
    This class overlays Track_ID labels on video frames for all tracked wagons.
    Labels are positioned near the top of each wagon's bounding box with a
    contrasting color scheme for visibility.
    
    Attributes:
        LABEL_FONT: OpenCV font for track ID labels
        LABEL_SCALE: Font scale for labels
        LABEL_THICKNESS: Line thickness for label text
        LABEL_COLOR: BGR color for label text (cyan for contrast)
        BACKGROUND_COLOR: BGR color for label background (black)
        PADDING: Padding around label text
    
    Requirements:
        - 2.1: Display Track_ID on video overlay when wagon detected
        - 2.2: Display Track_ID near wagon's bounding box in readable font
        - 2.3: Use contrasting color for visibility
        - 2.4: Display Track_IDs for all detected wagons
        - 2.5: Update Track_ID display in real-time
    """
    
    LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
    LABEL_SCALE = 0.7
    LABEL_THICKNESS = 2
    LABEL_COLOR = (255, 255, 0)  # Cyan (BGR) for contrast
    BACKGROUND_COLOR = (0, 0, 0)  # Black background
    PADDING = 5
    
    def draw_track_ids(
        self,
        frame: np.ndarray,
        tracked_wagons: List[TrackedWagon]
    ) -> np.ndarray:
        """Draw track IDs on frame for all tracked wagons.
        
        Overlays Track_ID labels on the frame for each tracked wagon.
        Labels are positioned near the top of each wagon's bounding box
        with a black background for readability.
        
        Args:
            frame: BGR frame to annotate (will be modified in place)
            tracked_wagons: List of tracked wagons with IDs and bounding boxes
            
        Returns:
            Annotated frame with track ID labels
            
        Requirements:
            - 2.1: Display Track_ID on video overlay
            - 2.4: Display Track_IDs for all detected wagons
            - 2.5: Update in real-time
        """
        if frame is None or len(frame.shape) < 2:
            return frame
        
        # Create a copy to avoid modifying the original
        annotated_frame = frame.copy()
        
        for wagon in tracked_wagons:
            self._draw_single_track_id(annotated_frame, wagon)
        
        return annotated_frame
    
    def _draw_single_track_id(
        self,
        frame: np.ndarray,
        wagon: TrackedWagon
    ) -> None:
        """Draw a single track ID label on the frame.
        
        Args:
            frame: BGR frame to annotate (modified in place)
            wagon: Tracked wagon with ID and bounding box
        """
        # Generate label text
        label_text = f"ID: {wagon.track_id}"
        
        # Calculate label position near top of bounding box
        label_pos = self._calculate_label_position(wagon.bbox, frame.shape)
        
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text,
            self.LABEL_FONT,
            self.LABEL_SCALE,
            self.LABEL_THICKNESS
        )
        
        # Calculate background rectangle coordinates
        bg_x1 = label_pos[0] - self.PADDING
        bg_y1 = label_pos[1] - text_height - self.PADDING
        bg_x2 = label_pos[0] + text_width + self.PADDING
        bg_y2 = label_pos[1] + self.PADDING
        
        # Ensure background stays within frame bounds
        bg_x1 = max(0, bg_x1)
        bg_y1 = max(0, bg_y1)
        bg_x2 = min(frame.shape[1], bg_x2)
        bg_y2 = min(frame.shape[0], bg_y2)
        
        # Draw background rectangle
        cv2.rectangle(
            frame,
            (bg_x1, bg_y1),
            (bg_x2, bg_y2),
            self.BACKGROUND_COLOR,
            cv2.FILLED
        )
        
        # Draw track ID text
        cv2.putText(
            frame,
            label_text,
            label_pos,
            self.LABEL_FONT,
            self.LABEL_SCALE,
            self.LABEL_COLOR,
            self.LABEL_THICKNESS,
            cv2.LINE_AA
        )
    
    def _calculate_label_position(
        self,
        bbox: BoundingBox,
        frame_shape: Tuple[int, ...]
    ) -> Tuple[int, int]:
        """Calculate optimal label position near bounding box.
        
        Positions the label near the top of the bounding box, ensuring
        it stays within frame bounds.
        
        Args:
            bbox: Bounding box of the wagon
            frame_shape: Shape of the frame (height, width, channels)
            
        Returns:
            Tuple of (x, y) coordinates for label text origin
            
        Requirements:
            - 2.2: Display Track_ID near wagon's bounding box
        """
        frame_height, frame_width = frame_shape[0], frame_shape[1]
        
        # Position label at top-left of bounding box, slightly above
        x = bbox.x1
        y = bbox.y1 - 5  # 5 pixels above the box
        
        # Ensure label stays within frame bounds
        # Get approximate text dimensions for boundary checking
        text_height = int(self.LABEL_SCALE * 20)  # Approximate height
        
        # If label would be above frame, position it inside the box
        if y - text_height < 0:
            y = bbox.y1 + text_height + self.PADDING
        
        # Ensure x is within bounds
        x = max(self.PADDING, min(x, frame_width - 100))
        
        return (x, y)
