"""
Frame saver component for the Mission Control Dashboard.

This module provides automatic saving of processed frames based on
configurable triggers (deblur, illumination enhancement, OCR).

Feature: dashboard-enhancements
Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6
"""

import os
import json
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from dashboard.models import FrameSaveConfig, FrameMetadata, ProcessingType


class FrameSaver:
    """
    Manages automatic saving of processed frames.
    
    This class handles conditional frame saving based on configuration,
    generating appropriate filenames with timestamps, and writing
    metadata sidecar files.
    
    Attributes:
        config: FrameSaveConfig with saving preferences
        
    Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6
    """
    
    def __init__(self, config: FrameSaveConfig):
        """
        Initialize frame saver with configuration.
        
        Args:
            config: FrameSaveConfig specifying save triggers and output directory
        """
        self.config = config
        self._ensure_output_directory()
    
    def _ensure_output_directory(self) -> None:
        """
        Create output directory if it doesn't exist.
        
        Only creates directory if frame saving is enabled.
        """
        if self.config.enabled and self.config.output_directory:
            Path(self.config.output_directory).mkdir(parents=True, exist_ok=True)
    
    def should_save(
        self,
        deblur_applied: bool = False,
        illumination_applied: bool = False,
        ocr_performed: bool = False
    ) -> bool:
        """
        Determine if frame should be saved based on config and processing applied.
        
        Args:
            deblur_applied: Whether deblur processing was applied
            illumination_applied: Whether illumination enhancement (CLAHE/gamma) was applied
            ocr_performed: Whether OCR was performed on the frame
            
        Returns:
            True if frame should be saved, False otherwise
            
        Requirements: 5.1, 5.2, 5.3, 8.4
        """
        # If saving is disabled globally, never save
        if not self.config.enabled:
            return False
        
        # Check each trigger condition
        if deblur_applied and self.config.save_on_deblur:
            return True
        
        if illumination_applied and self.config.save_on_illumination:
            return True
        
        if ocr_performed and self.config.save_on_ocr:
            return True
        
        return False
    
    def _generate_filename(self, metadata: FrameMetadata) -> str:
        """
        Generate filename with timestamp, frame index, and processing info.
        
        Format: {timestamp}_{frame_idx}_{processing}.jpg
        
        Args:
            metadata: FrameMetadata with frame information
            
        Returns:
            Generated filename string
            
        Requirements: 5.4
        """
        # Format timestamp as YYYYMMDD_HHMMSS_microseconds
        timestamp_str = metadata.timestamp.strftime("%Y%m%d_%H%M%S_%f")
        
        # Join processing types with underscore
        processing_str = "_".join(metadata.processing_applied) if metadata.processing_applied else "raw"
        
        # Build filename
        filename = f"{timestamp_str}_{metadata.frame_index}_{processing_str}.jpg"
        
        return filename
    
    def _save_metadata(self, filepath: str, metadata: FrameMetadata) -> bool:
        """
        Write JSON sidecar file with frame metadata.
        
        Creates a .json file alongside the saved frame with metadata
        indicating which processing was applied.
        
        Args:
            filepath: Path to the saved frame file
            metadata: FrameMetadata to serialize
            
        Returns:
            True if metadata was saved successfully, False otherwise
            
        Requirements: 5.5
        """
        try:
            # Generate metadata filepath by replacing extension
            metadata_path = Path(filepath).with_suffix('.json')
            
            # Serialize metadata to dict
            metadata_dict = {
                "timestamp": metadata.timestamp.isoformat(),
                "frame_index": metadata.frame_index,
                "processing_applied": metadata.processing_applied,
                "wagon_id": metadata.wagon_id
            }
            
            # Write JSON file
            with open(metadata_path, 'w') as f:
                json.dump(metadata_dict, f, indent=2)
            
            return True
            
        except (IOError, OSError, TypeError) as e:
            # Log warning but don't fail - frame was already saved
            return False
    
    def save_frame(
        self,
        frame: np.ndarray,
        metadata: FrameMetadata
    ) -> Optional[str]:
        """
        Save frame to disk with metadata.
        
        Saves the frame as a JPEG file with a timestamp-based filename
        and creates a JSON sidecar file with metadata.
        
        Args:
            frame: BGR frame to save (numpy array)
            metadata: FrameMetadata with frame information
            
        Returns:
            Path to saved file, or None if saving is disabled or failed
            
        Requirements: 5.1, 5.2, 5.3, 5.4, 5.5
        """
        # Don't save if disabled
        if not self.config.enabled:
            return None
        
        # Validate frame
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return None
        
        try:
            # Ensure output directory exists
            self._ensure_output_directory()
            
            # Generate filename and full path
            filename = self._generate_filename(metadata)
            filepath = os.path.join(self.config.output_directory, filename)
            
            # Save frame as JPEG
            success = cv2.imwrite(filepath, frame)
            
            if not success:
                return None
            
            # Save metadata sidecar file
            self._save_metadata(filepath, metadata)
            
            return filepath
            
        except (IOError, OSError, cv2.error) as e:
            # Return None on any save error
            return None
    
    def update_config(self, config: FrameSaveConfig) -> None:
        """
        Update the frame saver configuration.
        
        Args:
            config: New FrameSaveConfig to use
        """
        self.config = config
        self._ensure_output_directory()
