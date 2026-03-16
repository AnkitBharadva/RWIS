"""Parallel ROI processor for improved pipeline performance.

This module provides parallel processing of wagon ROIs to reduce latency.
Instead of processing wagons sequentially, multiple wagons are processed
concurrently using ThreadPoolExecutor.

Key optimizations:
- Parallel OCR processing across multiple ROIs
- Parallel damage detection
- Async deblurring with result caching
- Non-blocking logging operations
"""

from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from typing import List, Tuple, Optional, Dict
import numpy as np
import logging

from utils.data_models import (
    TrackedWagon, WagonRecord, DamageDetection, OCRResult
)
from utils.roi_utils import extract_roi, resize_roi_for_deblur


logger = logging.getLogger(__name__)


class ParallelROIProcessor:
    """Processes multiple wagon ROIs in parallel for better performance."""
    
    def __init__(
        self,
        ocr_pipeline,
        damage_detector,
        deblur_manager,
        blur_detector,
        logger_instance,
        max_workers: int = 4
    ):
        """Initialize parallel processor.
        
        Args:
            ocr_pipeline: OCR pipeline instance
            damage_detector: Damage detector instance
            deblur_manager: Deblur manager instance
            blur_detector: Blur detector instance
            logger_instance: Logger instance
            max_workers: Maximum number of parallel workers
        """
        self.ocr_pipeline = ocr_pipeline
        self.damage_detector = damage_detector
        self.deblur_manager = deblur_manager
        self.blur_detector = blur_detector
        self.logger = logger_instance
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
    def process_wagon_batch(
        self,
        frame: np.ndarray,
        tracked_wagons: List[TrackedWagon],
        frame_index: int,
        max_roi_width: int = 256
    ) -> List[WagonRecord]:
        """Process multiple wagons in parallel.
        
        Args:
            frame: Input frame
            tracked_wagons: List of tracked wagons
            frame_index: Current frame index
            max_roi_width: Maximum ROI width for deblurring
            
        Returns:
            List of wagon records
        """
        # Filter wagons that need processing
        wagons_to_process = [
            w for w in tracked_wagons 
            if w.crossed_line and w.count_index is not None
        ]
        
        if not wagons_to_process:
            return []
        
        # Submit all wagon processing tasks
        futures = {}
        for wagon in wagons_to_process:
            future = self.executor.submit(
                self._process_single_wagon,
                frame,
                wagon,
                frame_index,
                max_roi_width
            )
            futures[future] = wagon
        
        # Collect results as they complete
        records = []
        for future in as_completed(futures):
            try:
                record = future.result(timeout=5.0)  # 5 second timeout
                if record:
                    records.append(record)
                    # Async logging (non-blocking)
                    self.executor.submit(self.logger.log_wagon, record)
            except Exception as e:
                wagon = futures[future]
                logger.error(f"Error processing wagon {wagon.track_id}: {e}")
        
        return records
    
    def _process_single_wagon(
        self,
        frame: np.ndarray,
        wagon: TrackedWagon,
        frame_index: int,
        max_roi_width: int
    ) -> Optional[WagonRecord]:
        """Process a single wagon (runs in parallel).
        
        Args:
            frame: Input frame
            wagon: Tracked wagon
            frame_index: Current frame index
            max_roi_width: Maximum ROI width
            
        Returns:
            WagonRecord or None if processing failed
        """
        try:
            # Extract ROI
            raw_roi, actual_bbox = extract_roi(frame, wagon.bbox, clip_to_bounds=True)
            if raw_roi is None or raw_roi.size == 0:
                return None
            
            # Deblurring and blur detection
            processed_roi = raw_roi
            deblur_applied = False
            deblur_source_frame = None
            blur_score = 0.0
            
            if self.deblur_manager is not None:
                try:
                    processed_roi, deblur_applied, deblur_source_frame = \
                        self.deblur_manager.process_roi(
                            raw_roi,
                            wagon.track_id,
                            frame_index
                        )
                    # Get blur score
                    resized_roi, _ = resize_roi_for_deblur(raw_roi, max_roi_width)
                    blur_score = self.blur_detector.compute_blur_score(resized_roi)
                except Exception as e:
                    logger.warning(f"Deblur failed for wagon {wagon.track_id}: {e}")
                    processed_roi = raw_roi
            else:
                blur_score = self.blur_detector.compute_blur_score(raw_roi)
            
            # Parallel OCR and damage detection using nested futures
            ocr_future = self.executor.submit(self._extract_ocr, processed_roi)
            damage_future = self.executor.submit(self._detect_damage, raw_roi, wagon.track_id)
            
            # Wait for both to complete
            ocr_result = ocr_future.result(timeout=2.0)
            damage_detections = damage_future.result(timeout=2.0)
            
            # Create wagon record
            record = WagonRecord(
                wagon_id=wagon.track_id,
                count_index=wagon.count_index,
                frame_index=frame_index,
                timestamp=wagon.timestamp,
                bbox=wagon.bbox,
                confidence=wagon.confidence,
                blur_score=blur_score,
                ocr_text=ocr_result.text if ocr_result else "",
                ocr_confidence=ocr_result.confidence if ocr_result else 0.0,
                damage_detections=damage_detections,
                deblur_applied=deblur_applied,
                deblur_source_frame=deblur_source_frame
            )
            
            return record
            
        except Exception as e:
            logger.error(f"Error in _process_single_wagon for wagon {wagon.track_id}: {e}")
            return None
    
    def _extract_ocr(self, roi: np.ndarray) -> OCRResult:
        """Extract OCR from ROI."""
        try:
            return self.ocr_pipeline.extract_text(roi)
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return OCRResult(text="", confidence=0.0, bbox=None)
    
    def _detect_damage(self, roi: np.ndarray, wagon_id: int) -> List[DamageDetection]:
        """Detect damage on ROI."""
        try:
            return self.damage_detector.detect(roi, wagon_id)
        except Exception as e:
            logger.error(f"Damage detection failed: {e}")
            return []
    
    def shutdown(self):
        """Shutdown the executor."""
        self.executor.shutdown(wait=True)
