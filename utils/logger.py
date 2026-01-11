"""Logging module for the Railway Wagon Inspection Pipeline.

This module provides the InspectionLogger class for logging wagon inspection
records to CSV and JSON formats, with optional debug frame saving.
"""

import csv
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from utils.data_models import BoundingBox, WagonRecord


class InspectionLogger:
    """Logger for wagon inspection records.
    
    Handles logging to CSV and JSON formats, and optionally saves
    annotated debug frames.
    
    Attributes:
        output_dir: Directory for output files
        formats: List of output formats ('csv', 'json')
        enable_debug: Whether to save debug frames
    """
    
    def __init__(
        self,
        output_dir: str,
        formats: Optional[List[str]] = None,
        enable_debug: bool = False
    ):
        """Initialize the logger.
        
        Args:
            output_dir: Directory for output files
            formats: List of output formats ('csv', 'json'). Defaults to both.
            enable_debug: Whether to save debug frames
        """
        self.output_dir = Path(output_dir)
        self.formats = formats if formats is not None else ['csv', 'json']
        self.enable_debug = enable_debug
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.debug_frames_dir = self.output_dir / 'debug_frames'
        if self.enable_debug:
            self.debug_frames_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.csv_path = self.output_dir / 'logs.csv'
        self.json_path = self.output_dir / 'logs.json'
        
        # Internal state
        self._csv_file = None
        self._csv_writer = None
        self._json_records: List[Dict[str, Any]] = []
        self._csv_initialized = False
        
        # CSV field names matching WagonRecord
        self._csv_fieldnames = [
            'timestamp', 'wagon_id', 'count_index', 'blur_score',
            'frame_index', 'damage_detected', 'damage_classes',
            'damage_bboxes', 'ocr_text', 'ocr_confidence',
            'deblur_applied', 'deblur_source_frame'
        ]
    
    def _init_csv(self) -> None:
        """Initialize CSV file with headers if not already done."""
        if self._csv_initialized:
            return
        
        if 'csv' in self.formats:
            self._csv_file = open(self.csv_path, 'w', newline='', encoding='utf-8')
            self._csv_writer = csv.DictWriter(
                self._csv_file,
                fieldnames=self._csv_fieldnames
            )
            self._csv_writer.writeheader()
            self._csv_initialized = True
    
    def _record_to_dict(self, record: WagonRecord) -> Dict[str, Any]:
        """Convert WagonRecord to a dictionary for logging.
        
        Args:
            record: The wagon record to convert
            
        Returns:
            Dictionary representation of the record
        """
        # Convert damage_bboxes to serializable format
        damage_bboxes_serialized = [
            {'x1': bbox.x1, 'y1': bbox.y1, 'x2': bbox.x2, 'y2': bbox.y2}
            for bbox in record.damage_bboxes
        ]
        
        return {
            'timestamp': record.timestamp,
            'wagon_id': record.wagon_id,
            'count_index': record.count_index,
            'blur_score': record.blur_score,
            'frame_index': record.frame_index,
            'damage_detected': record.damage_detected,
            'damage_classes': record.damage_classes,
            'damage_bboxes': damage_bboxes_serialized,
            'ocr_text': record.ocr_text,
            'ocr_confidence': record.ocr_confidence,
            'deblur_applied': record.deblur_applied,
            'deblur_source_frame': record.deblur_source_frame
        }
    
    def _record_to_csv_dict(self, record: WagonRecord) -> Dict[str, Any]:
        """Convert WagonRecord to a CSV-compatible dictionary.
        
        Args:
            record: The wagon record to convert
            
        Returns:
            Dictionary with string-serialized complex fields
        """
        base_dict = self._record_to_dict(record)
        
        # Serialize lists to JSON strings for CSV
        base_dict['damage_classes'] = json.dumps(base_dict['damage_classes'])
        base_dict['damage_bboxes'] = json.dumps(base_dict['damage_bboxes'])
        
        return base_dict
    
    def log_wagon(self, record: WagonRecord) -> None:
        """Log a wagon inspection record.
        
        Writes the record to CSV and/or JSON based on configured formats.
        
        Args:
            record: The wagon record to log
        """
        record_dict = self._record_to_dict(record)
        
        # Write to CSV
        if 'csv' in self.formats:
            self._init_csv()
            csv_dict = self._record_to_csv_dict(record)
            self._csv_writer.writerow(csv_dict)
        
        # Append to JSON records
        if 'json' in self.formats:
            self._json_records.append(record_dict)
    
    def save_debug_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        annotations: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[str]:
        """Save an annotated debug frame if debug mode is enabled.
        
        Args:
            frame: The frame to save (BGR format)
            frame_index: Index of the frame
            annotations: Optional list of annotation dictionaries with keys:
                - 'bbox': BoundingBox or dict with x1, y1, x2, y2
                - 'label': Text label to display
                - 'color': BGR tuple (default: green)
        
        Returns:
            Path to saved frame if saved, None otherwise
        """
        if not self.enable_debug:
            return None
        
        # Make a copy to avoid modifying original
        annotated_frame = frame.copy()
        
        # Draw annotations if provided
        if annotations:
            for ann in annotations:
                bbox = ann.get('bbox')
                if bbox is None:
                    continue
                
                # Handle both BoundingBox and dict
                if isinstance(bbox, BoundingBox):
                    x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
                else:
                    x1, y1 = bbox.get('x1', 0), bbox.get('y1', 0)
                    x2, y2 = bbox.get('x2', 0), bbox.get('y2', 0)
                
                color = ann.get('color', (0, 255, 0))  # Default green
                label = ann.get('label', '')
                
                # Draw rectangle
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label if provided
                if label:
                    cv2.putText(
                        annotated_frame, label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2
                    )
        
        # Save frame
        filename = f'frame_{frame_index:06d}.jpg'
        filepath = self.debug_frames_dir / filename
        cv2.imwrite(str(filepath), annotated_frame)
        
        return str(filepath)
    
    def flush(self) -> None:
        """Flush all pending writes to disk.
        
        Ensures CSV buffer is flushed and JSON file is written.
        """
        # Flush CSV
        if self._csv_file is not None:
            self._csv_file.flush()
        
        # Write JSON file
        if 'json' in self.formats and self._json_records:
            with open(self.json_path, 'w', encoding='utf-8') as f:
                json.dump(self._json_records, f, indent=2)
    
    def close(self) -> None:
        """Close all open file handles and flush remaining data."""
        self.flush()
        
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures files are closed."""
        self.close()
        return False
