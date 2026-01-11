"""
OCR frame saver module for saving OCR frames with metadata.

This module provides the OCRFrameSaver class that handles saving frames
with OCR detections and accompanying JSON metadata files.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 6.2
"""

import os
import json
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from utils.data_models import OCRDetection, OCRFrameMetadata


class OCRFrameSaver:
    """Saves OCR frames with JSON metadata.
    
    This class handles saving annotated frames when OCR detects text,
    along with JSON metadata files containing detection information.
    
    Attributes:
        DEFAULT_OUTPUT_DIR: Default output directory for saved frames
        
    Requirements:
        - 3.1: Save annotated frame when OCR detects text
        - 3.2: Filename includes timestamp, frame index, and wagon ID
        - 3.3: Save JSON metadata with all required fields
        - 3.4: Store frames in configurable output directory
        - 3.5: Create output directory if it doesn't exist
        - 6.2: Include deblur status in metadata
    """
    
    DEFAULT_OUTPUT_DIR = "outputs/ocr_frames"
    
    def __init__(self, output_directory: Optional[str] = None):
        """Initialize the OCR frame saver.
        
        Args:
            output_directory: Directory to save frames. If None, uses DEFAULT_OUTPUT_DIR.
                            Directory will be created if it doesn't exist.
                            
        Requirements:
            - 3.4: Store frames in configurable output directory
            - 3.5: Create output directory if it doesn't exist
        """
        self._output_directory = output_directory or self.DEFAULT_OUTPUT_DIR
        self._ensure_output_directory()
    
    @property
    def output_directory(self) -> str:
        """Get the current output directory."""
        return self._output_directory
    
    @output_directory.setter
    def output_directory(self, value: str) -> None:
        """Set the output directory and ensure it exists.
        
        Args:
            value: New output directory path
        """
        self._output_directory = value or self.DEFAULT_OUTPUT_DIR
        self._ensure_output_directory()
    
    def _ensure_output_directory(self) -> None:
        """Create output directory if it doesn't exist.
        
        Requirements:
            - 3.5: Create output directory if it doesn't exist
        """
        Path(self._output_directory).mkdir(parents=True, exist_ok=True)
    
    def generate_filename(
        self,
        timestamp: datetime,
        frame_index: int,
        wagon_id: int
    ) -> str:
        """Generate filename with timestamp, frame index, and wagon ID.
        
        Format: ocr_{YYYYMMDD_HHMMSS_ffffff}_{frame_index}_{wagon_id}.jpg
        
        Args:
            timestamp: Timestamp of the frame capture
            frame_index: Index of the video frame
            wagon_id: ID of the wagon in the frame
            
        Returns:
            Generated filename string
            
        Requirements:
            - 3.2: Filename includes timestamp, frame index, and wagon ID
        """
        # Format timestamp as YYYYMMDD_HHMMSS_microseconds
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")
        
        # Build filename with all required components
        filename = f"ocr_{timestamp_str}_{frame_index}_{wagon_id}.jpg"
        
        return filename
    
    def _save_metadata(
        self,
        filepath: str,
        metadata: OCRFrameMetadata
    ) -> bool:
        """Save JSON metadata file alongside the frame.
        
        Creates a .json file with the same base name as the frame file,
        containing all required metadata fields.
        
        Args:
            filepath: Path to the saved frame file
            metadata: OCRFrameMetadata to serialize
            
        Returns:
            True if metadata was saved successfully, False otherwise
            
        Requirements:
            - 3.3: Save JSON metadata with all required fields
            - 6.2: Include deblur status in metadata
        """
        try:
            # Generate metadata filepath by replacing extension
            metadata_path = Path(filepath).with_suffix('.json')
            
            # Serialize metadata to dict with all required fields
            metadata_dict = {
                "timestamp": metadata.timestamp,
                "frame_index": metadata.frame_index,
                "wagon_id": metadata.wagon_id,
                "detections": metadata.detections,
                "deblur_applied": metadata.deblur_applied,
                "illumination_applied": metadata.illumination_applied
            }
            
            # Add optional fields if present
            if metadata.blur_score is not None:
                metadata_dict["blur_score"] = metadata.blur_score
            if metadata.luminance_level is not None:
                metadata_dict["luminance_level"] = metadata.luminance_level
            
            # Write JSON file with pretty formatting
            with open(metadata_path, 'w') as f:
                json.dump(metadata_dict, f, indent=2)
            
            return True
            
        except (IOError, OSError, TypeError) as e:
            # Return False on any save error
            return False
    
    def save_ocr_frame(
        self,
        frame: np.ndarray,
        ocr_detections: List[OCRDetection],
        metadata: OCRFrameMetadata
    ) -> Tuple[Optional[str], Optional[str]]:
        """Save frame and metadata, returns (frame_path, metadata_path).
        
        Saves the annotated frame as a JPEG file and creates a JSON
        metadata file alongside it with all detection information.
        
        Args:
            frame: BGR frame to save (numpy array)
            ocr_detections: List of OCR detections (used for validation)
            metadata: OCRFrameMetadata with frame information
            
        Returns:
            Tuple of (frame_path, metadata_path), or (None, None) if saving failed
            
        Requirements:
            - 3.1: Save annotated frame when OCR detects text
            - 3.2: Filename includes timestamp, frame index, and wagon ID
            - 3.3: Save JSON metadata with all required fields
            - 3.4: Store frames in configurable output directory
        """
        # Validate frame
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return (None, None)
        
        try:
            # Ensure output directory exists
            self._ensure_output_directory()
            
            # Parse timestamp from metadata (ISO format string)
            try:
                timestamp = datetime.fromisoformat(metadata.timestamp)
            except (ValueError, TypeError):
                timestamp = datetime.now()
            
            # Generate filename and full path
            filename = self.generate_filename(
                timestamp,
                metadata.frame_index,
                metadata.wagon_id
            )
            frame_path = os.path.join(self._output_directory, filename)
            
            # Save frame as JPEG
            success = cv2.imwrite(frame_path, frame)
            
            if not success:
                return (None, None)
            
            # Save metadata sidecar file
            metadata_saved = self._save_metadata(frame_path, metadata)
            
            if metadata_saved:
                metadata_path = str(Path(frame_path).with_suffix('.json'))
                return (frame_path, metadata_path)
            else:
                return (frame_path, None)
            
        except (IOError, OSError, cv2.error) as e:
            # Return None on any save error
            return (None, None)
