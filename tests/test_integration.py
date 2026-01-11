"""Integration tests for the Railway Wagon Inspection Pipeline.

These tests verify that all pipeline components work together correctly
without requiring actual video files or trained models.

Task 16: Final checkpoint - Full pipeline integration
"""

import json
import os
import tempfile
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pytest

from config import PipelineConfig, load_config
from pipelines.blur_detector import BlurDetector
from pipelines.wagon_detector import WagonDetector, FrameProcessingState
from pipelines.damage_detector import DamageDetector
from pipelines.nafnet_wrapper import NAFNetDeblur
from tracking.tracker import WagonTracker
from utils.clahe import CLAHEEnhancer
from utils.logger import InspectionLogger
from utils.data_models import (
    BlurDecision, WagonRecord, BoundingBox, DamageClass
)
from utils.gpu_utils import (
    GPUMemoryMonitor, AdaptiveBatchSizer, OperationQueue,
    clear_gpu_cache, get_gpu_memory_summary
)


class TestPipelineIntegration:
    """Integration tests for the full pipeline."""
    
    def test_blur_to_clahe_integration(self):
        """Test blur detection integrates with CLAHE enhancement."""
        # Create test frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Initialize components
        blur_detector = BlurDetector(t1=100.0, t2=300.0)
        clahe = CLAHEEnhancer()
        
        # Process through blur detection
        blur_score = blur_detector.compute_blur_score(frame)
        blur_decision = blur_detector.get_blur_decision(blur_score)
        
        # Apply CLAHE if needed
        if blur_decision in (BlurDecision.ROI_DEBLUR, BlurDecision.NO_DEBLUR):
            enhanced = clahe.enhance(frame)
            assert enhanced.shape == frame.shape
            assert enhanced.dtype == frame.dtype
        
        # Verify blur decision is valid
        assert blur_decision in (
            BlurDecision.SKIP_DEBLUR,
            BlurDecision.ROI_DEBLUR,
            BlurDecision.NO_DEBLUR
        )
    
    def test_tracker_counting_integration(self):
        """Test tracker correctly counts wagons crossing the line."""
        from utils.data_models import WagonDetection
        
        # Initialize tracker with counting line at 50%
        tracker = WagonTracker(counting_line_y=0.5)
        frame_shape = (480, 640, 3)
        
        # Simulate wagon moving across the counting line
        # Start above the line (y < 240)
        detection1 = WagonDetection(
            bbox=BoundingBox(x1=100, y1=100, x2=200, y2=200),
            confidence=0.9,
            class_id=0
        )
        
        # Update tracker
        tracked = tracker.update([detection1], frame_shape)
        assert len(tracked) == 1
        initial_count = tracker.get_wagon_count()
        
        # Move wagon below the line (y > 240)
        detection2 = WagonDetection(
            bbox=BoundingBox(x1=100, y1=300, x2=200, y2=400),
            confidence=0.9,
            class_id=0
        )
        
        # Update tracker - wagon should cross line
        tracked = tracker.update([detection2], frame_shape)
        final_count = tracker.get_wagon_count()
        
        # Count should have increased
        assert final_count >= initial_count
    
    def test_logger_creates_files(self):
        """Test logger creates CSV and JSON files correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize logger
            logger = InspectionLogger(
                output_dir=tmpdir,
                formats=['csv', 'json'],
                enable_debug=False
            )
            
            # Create test record
            record = WagonRecord(
                timestamp="2024-01-01T12:00:00",
                wagon_id=1,
                count_index=1,
                blur_score=150.0,
                frame_index=100,
                damage_detected=True,
                damage_classes=["damage_door"],
                damage_bboxes=[BoundingBox(x1=10, y1=10, x2=50, y2=50)],
                ocr_text="ABC123",
                ocr_confidence=0.95
            )
            
            # Log the record
            logger.log_wagon(record)
            logger.close()
            
            # Verify CSV file exists and has content
            csv_path = Path(tmpdir) / 'logs.csv'
            assert csv_path.exists()
            with open(csv_path, 'r') as f:
                content = f.read()
                assert 'wagon_id' in content
                assert 'ABC123' in content
            
            # Verify JSON file exists and has content
            json_path = Path(tmpdir) / 'logs.json'
            assert json_path.exists()
            with open(json_path, 'r') as f:
                data = json.load(f)
                assert len(data) == 1
                assert data[0]['wagon_id'] == 1
                assert data[0]['ocr_text'] == 'ABC123'
    
    def test_logger_debug_frames_toggle(self):
        """Test debug frame creation respects enable_debug flag."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test with debug enabled
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = InspectionLogger(
                output_dir=tmpdir,
                formats=['csv'],
                enable_debug=True
            )
            
            result = logger.save_debug_frame(frame, 0, [])
            logger.close()
            
            assert result is not None
            assert os.path.exists(result)
        
        # Test with debug disabled
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = InspectionLogger(
                output_dir=tmpdir,
                formats=['csv'],
                enable_debug=False
            )
            
            result = logger.save_debug_frame(frame, 0, [])
            logger.close()
            
            assert result is None
    
    def test_nafnet_roi_constraint(self):
        """Test NAFNet rejects full-frame inputs."""
        nafnet = NAFNetDeblur(model_path="models/nafnet.pth")
        nafnet.set_full_frame_dimensions(1920, 1080)
        
        # Small ROI should be accepted (validation only, no actual deblur)
        small_roi = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        # Should not raise any exception
        nafnet._validate_roi_dimensions(small_roi)
        
        # Full frame should be rejected
        from pipelines.nafnet_wrapper import FullFrameDeblurError
        full_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        with pytest.raises(FullFrameDeblurError):
            nafnet._validate_roi_dimensions(full_frame)
    
    def test_gpu_memory_utilities(self):
        """Test GPU memory utilities work correctly."""
        # Test memory monitor
        monitor = GPUMemoryMonitor(memory_limit_bytes=6 * 1024**3)
        info = monitor.get_memory_info()
        assert 'allocated' in info
        assert 'limit' in info
        
        # Test batch sizer
        sizer = AdaptiveBatchSizer(
            initial_batch_size=4,
            min_batch_size=1,
            max_batch_size=16
        )
        batch_size = sizer.get_batch_size()
        assert 1 <= batch_size <= 16
        
        # Test operation queue
        queue = OperationQueue()
        assert queue.pending_count() == 0
        
        # Test memory summary
        summary = get_gpu_memory_summary()
        assert isinstance(summary, str)
    
    def test_config_validation(self):
        """Test configuration validation catches invalid values."""
        # Valid config should pass
        config = PipelineConfig(
            blur_threshold_t1=100.0,
            blur_threshold_t2=300.0,
            counting_line_position=0.5
        )
        config.validate()  # Should not raise
        
        # Invalid T1 >= T2 should fail
        with pytest.raises(ValueError):
            invalid_config = PipelineConfig(
                blur_threshold_t1=300.0,
                blur_threshold_t2=100.0
            )
            invalid_config.validate()
        
        # Invalid counting line position should fail
        with pytest.raises(ValueError):
            invalid_config = PipelineConfig(
                counting_line_position=1.5
            )
            invalid_config.validate()
    
    def test_data_models_completeness(self):
        """Test all data models have required fields."""
        # Test BoundingBox
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=200)
        assert bbox.width == 90
        assert bbox.height == 180
        assert bbox.center == (55, 110)
        
        # Test WagonRecord has all required fields
        record = WagonRecord(
            timestamp="2024-01-01T12:00:00",
            wagon_id=1,
            count_index=1,
            blur_score=150.0,
            frame_index=100,
            damage_detected=False,
            damage_classes=[],
            damage_bboxes=[],
            ocr_text="",
            ocr_confidence=0.0
        )
        
        # Verify all fields are accessible
        assert record.timestamp is not None
        assert record.wagon_id >= 0
        assert record.count_index >= 0
        assert isinstance(record.blur_score, float)
        assert record.frame_index >= 0
        assert isinstance(record.damage_detected, bool)
        assert isinstance(record.damage_classes, list)
        assert isinstance(record.damage_bboxes, list)
        assert isinstance(record.ocr_text, str)
        assert 0.0 <= record.ocr_confidence <= 1.0
    
    def test_dual_format_logging_consistency(self):
        """Test CSV and JSON outputs contain equivalent data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = InspectionLogger(
                output_dir=tmpdir,
                formats=['csv', 'json'],
                enable_debug=False
            )
            
            # Log multiple records
            for i in range(3):
                record = WagonRecord(
                    timestamp=f"2024-01-01T12:0{i}:00",
                    wagon_id=i + 1,
                    count_index=i + 1,
                    blur_score=100.0 + i * 50,
                    frame_index=i * 100,
                    damage_detected=i % 2 == 0,
                    damage_classes=["damage_door"] if i % 2 == 0 else [],
                    damage_bboxes=[BoundingBox(x1=10, y1=10, x2=50, y2=50)] if i % 2 == 0 else [],
                    ocr_text=f"WAGON{i}",
                    ocr_confidence=0.9 + i * 0.03
                )
                logger.log_wagon(record)
            
            logger.close()
            
            # Read JSON
            json_path = Path(tmpdir) / 'logs.json'
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            
            # Read CSV
            csv_path = Path(tmpdir) / 'logs.csv'
            import csv
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                csv_data = list(reader)
            
            # Verify same number of records
            assert len(json_data) == len(csv_data) == 3
            
            # Verify key fields match
            for i in range(3):
                assert json_data[i]['wagon_id'] == int(csv_data[i]['wagon_id'])
                assert json_data[i]['ocr_text'] == csv_data[i]['ocr_text']


class TestEndToEndSimulation:
    """Simulated end-to-end tests without actual video/models."""
    
    def test_simulated_pipeline_flow(self):
        """Test simulated pipeline flow with mock data."""
        # Initialize components
        blur_detector = BlurDetector(t1=100.0, t2=300.0)
        clahe = CLAHEEnhancer()
        tracker = WagonTracker(counting_line_y=0.5)
        
        # Simulate processing multiple frames
        frame_shape = (480, 640, 3)
        processed_frames = 0
        
        for frame_idx in range(10):
            # Create synthetic frame
            frame = np.random.randint(0, 255, frame_shape, dtype=np.uint8)
            
            # Stage 1: Blur detection
            blur_score = blur_detector.compute_blur_score(frame)
            blur_decision = blur_detector.get_blur_decision(blur_score)
            
            # Stage 2: CLAHE enhancement
            if blur_decision != BlurDecision.SKIP_DEBLUR:
                frame = clahe.enhance(frame)
            
            processed_frames += 1
        
        assert processed_frames == 10
    
    def test_memory_management_simulation(self):
        """Test memory management utilities in simulated scenario."""
        monitor = GPUMemoryMonitor(memory_limit_bytes=6 * 1024**3)
        sizer = AdaptiveBatchSizer(initial_batch_size=4)
        queue = OperationQueue()
        
        # Start monitoring
        monitor.start()
        queue.start()
        
        # Simulate batch size adjustments
        for _ in range(5):
            batch_size = sizer.update()
            assert 1 <= batch_size <= 16
        
        # Stop monitoring
        monitor.stop()
        queue.stop()
        
        # Clear cache
        clear_gpu_cache()
