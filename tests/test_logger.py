"""
Property-based tests for the InspectionLogger.

Feature: railway-wagon-inspection
Properties: 13, 14, 16
Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 10.5
"""

import csv
import json
import os
import tempfile
import shutil
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings

from utils.data_models import BoundingBox, WagonRecord
from utils.logger import InspectionLogger


# Strategy for generating valid timestamps
valid_timestamp = st.text(
    alphabet=st.sampled_from('0123456789-T:.Z'),
    min_size=10,
    max_size=30
)

# Strategy for generating valid wagon IDs (positive integers)
valid_wagon_id = st.integers(min_value=1, max_value=100000)

# Strategy for generating valid count indices (positive integers)
valid_count_index = st.integers(min_value=1, max_value=100000)

# Strategy for generating valid blur scores
valid_blur_score = st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False)

# Strategy for generating valid frame indices (non-negative integers)
valid_frame_index = st.integers(min_value=0, max_value=1000000)

# Strategy for generating valid confidence scores [0.0, 1.0]
valid_confidence = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

# Strategy for generating valid OCR text (ASCII only to avoid encoding issues)
valid_ocr_text = st.text(
    alphabet=st.sampled_from('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_ '),
    min_size=0,
    max_size=50
)

# Strategy for generating valid damage class names
valid_damage_class = st.sampled_from([
    'bamboo_door', 'breakage', 'close_door', 'damage_door', 'dent', 'open_door'
])


@st.composite
def valid_bounding_box(draw):
    """Generate a valid bounding box with x1 < x2 and y1 < y2."""
    x1 = draw(st.integers(min_value=0, max_value=1000))
    y1 = draw(st.integers(min_value=0, max_value=1000))
    width = draw(st.integers(min_value=1, max_value=500))
    height = draw(st.integers(min_value=1, max_value=500))
    return BoundingBox(x1=x1, y1=y1, x2=x1 + width, y2=y1 + height)


@st.composite
def valid_wagon_record(draw):
    """Generate a valid WagonRecord with all required fields including deblur metadata."""
    damage_detected = draw(st.booleans())
    
    # Generate damage classes and bboxes only if damage detected
    if damage_detected:
        num_damages = draw(st.integers(min_value=1, max_value=5))
        damage_classes = [draw(valid_damage_class) for _ in range(num_damages)]
        damage_bboxes = [draw(valid_bounding_box()) for _ in range(num_damages)]
    else:
        damage_classes = []
        damage_bboxes = []
    
    # Generate deblur metadata
    deblur_applied = draw(st.booleans())
    deblur_source_frame = draw(valid_frame_index) if deblur_applied else None
    
    return WagonRecord(
        timestamp=draw(valid_timestamp),
        wagon_id=draw(valid_wagon_id),
        count_index=draw(valid_count_index),
        blur_score=draw(valid_blur_score),
        frame_index=draw(valid_frame_index),
        damage_detected=damage_detected,
        damage_classes=damage_classes,
        damage_bboxes=damage_bboxes,
        ocr_text=draw(valid_ocr_text),
        ocr_confidence=draw(valid_confidence),
        deblur_applied=deblur_applied,
        deblur_source_frame=deblur_source_frame
    )


class TestLogRecordCompleteness:
    """
    Property 13: Log Record Completeness
    
    For any wagon record written to the log, the record SHALL contain all required fields:
    - timestamp (non-empty string)
    - wagon_id (positive integer)
    - count_index (positive integer)
    - blur_score (float)
    - frame_index (non-negative integer)
    - damage_detected (boolean)
    - damage_classes (list, may be empty)
    - ocr_text (string, may be empty)
    - ocr_confidence (float in [0.0, 1.0])
    - deblur_applied (boolean)
    - deblur_source_frame (integer or None)
    
    Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.7
    """

    @given(record=valid_wagon_record())
    @settings(max_examples=100)
    def test_json_record_contains_all_required_fields(self, record):
        """
        Feature: railway-wagon-inspection, Property 13: Log Record Completeness
        
        Generate random wagon records and verify all required fields present in JSON output.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = InspectionLogger(
                output_dir=tmpdir,
                formats=['json'],
                enable_debug=False
            )
            
            logger.log_wagon(record)
            logger.flush()
            
            # Read JSON and verify fields
            json_path = Path(tmpdir) / 'logs.json'
            with open(json_path, 'r') as f:
                records = json.load(f)
            
            assert len(records) == 1
            logged = records[0]
            
            # Verify all required fields are present
            assert 'timestamp' in logged
            assert 'wagon_id' in logged
            assert 'count_index' in logged
            assert 'blur_score' in logged
            assert 'frame_index' in logged
            assert 'damage_detected' in logged
            assert 'damage_classes' in logged
            assert 'damage_bboxes' in logged
            assert 'ocr_text' in logged
            assert 'ocr_confidence' in logged
            assert 'deblur_applied' in logged
            assert 'deblur_source_frame' in logged
            
            # Verify field types and constraints
            assert isinstance(logged['timestamp'], str)
            assert isinstance(logged['wagon_id'], int) and logged['wagon_id'] > 0
            assert isinstance(logged['count_index'], int) and logged['count_index'] > 0
            assert isinstance(logged['blur_score'], (int, float))
            assert isinstance(logged['frame_index'], int) and logged['frame_index'] >= 0
            assert isinstance(logged['damage_detected'], bool)
            assert isinstance(logged['damage_classes'], list)
            assert isinstance(logged['ocr_text'], str)
            assert isinstance(logged['ocr_confidence'], (int, float))
            assert 0.0 <= logged['ocr_confidence'] <= 1.0
            assert isinstance(logged['deblur_applied'], bool)
            # deblur_source_frame can be int or None
            assert logged['deblur_source_frame'] is None or isinstance(logged['deblur_source_frame'], int)
            
            # If deblur_applied is True, deblur_source_frame should be a valid frame index
            if logged['deblur_applied']:
                assert logged['deblur_source_frame'] is not None
                assert logged['deblur_source_frame'] >= 0
            
            logger.close()

    @given(record=valid_wagon_record())
    @settings(max_examples=100)
    def test_csv_record_contains_all_required_fields(self, record):
        """
        Feature: railway-wagon-inspection, Property 13: Log Record Completeness
        
        Generate random wagon records and verify all required fields present in CSV output.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = InspectionLogger(
                output_dir=tmpdir,
                formats=['csv'],
                enable_debug=False
            )
            
            logger.log_wagon(record)
            logger.close()  # Close before reading
            
            # Read CSV and verify fields
            csv_path = Path(tmpdir) / 'logs.csv'
            with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            assert len(rows) == 1
            logged = rows[0]
            
            # Verify all required fields are present in CSV including deblur metadata
            required_fields = [
                'timestamp', 'wagon_id', 'count_index', 'blur_score',
                'frame_index', 'damage_detected', 'damage_classes',
                'damage_bboxes', 'ocr_text', 'ocr_confidence',
                'deblur_applied', 'deblur_source_frame'
            ]
            
            for field in required_fields:
                assert field in logged, f"Missing field: {field}"


class TestDualFormatLoggingConsistency:
    """
    Property 14: Dual Format Logging Consistency
    
    For any wagon record logged, both CSV and JSON outputs SHALL contain the same data.
    Parsing the CSV row and the corresponding JSON entry SHALL produce equivalent
    WagonRecord objects.
    
    Validates: Requirements 8.5
    """

    @given(record=valid_wagon_record())
    @settings(max_examples=100)
    def test_csv_and_json_produce_equivalent_records(self, record):
        """
        Feature: railway-wagon-inspection, Property 14: Dual Format Logging Consistency
        
        Generate random wagon records, log to both formats, parse CSV and JSON,
        verify they produce equivalent records.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = InspectionLogger(
                output_dir=tmpdir,
                formats=['csv', 'json'],
                enable_debug=False
            )
            
            logger.log_wagon(record)
            logger.close()  # Close before reading
            
            # Read JSON
            json_path = Path(tmpdir) / 'logs.json'
            with open(json_path, 'r', encoding='utf-8') as f:
                json_records = json.load(f)
            
            assert len(json_records) == 1
            json_record = json_records[0]
            
            # Read CSV
            csv_path = Path(tmpdir) / 'logs.csv'
            with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                csv_rows = list(reader)
            
            assert len(csv_rows) == 1
            csv_record = csv_rows[0]
            
            # Parse CSV fields (some are JSON-encoded strings)
            csv_damage_classes = json.loads(csv_record['damage_classes'])
            csv_damage_bboxes = json.loads(csv_record['damage_bboxes'])
            
            # Compare all fields
            assert json_record['timestamp'] == csv_record['timestamp']
            assert json_record['wagon_id'] == int(csv_record['wagon_id'])
            assert json_record['count_index'] == int(csv_record['count_index'])
            assert abs(json_record['blur_score'] - float(csv_record['blur_score'])) < 1e-6
            assert json_record['frame_index'] == int(csv_record['frame_index'])
            assert json_record['damage_detected'] == (csv_record['damage_detected'] == 'True')
            assert json_record['damage_classes'] == csv_damage_classes
            assert json_record['damage_bboxes'] == csv_damage_bboxes
            assert json_record['ocr_text'] == csv_record['ocr_text']
            assert abs(json_record['ocr_confidence'] - float(csv_record['ocr_confidence'])) < 1e-6
            
            # Compare deblur metadata
            assert json_record['deblur_applied'] == (csv_record['deblur_applied'] == 'True')
            
            # Handle deblur_source_frame (can be None or int)
            csv_deblur_source = csv_record['deblur_source_frame']
            if csv_deblur_source == '' or csv_deblur_source == 'None':
                assert json_record['deblur_source_frame'] is None
            else:
                assert json_record['deblur_source_frame'] == int(csv_deblur_source)

    @given(records=st.lists(valid_wagon_record(), min_size=1, max_size=10))
    @settings(max_examples=100)
    def test_multiple_records_consistency(self, records):
        """
        Feature: railway-wagon-inspection, Property 14: Dual Format Logging Consistency
        
        Verify consistency across multiple records logged to both formats.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = InspectionLogger(
                output_dir=tmpdir,
                formats=['csv', 'json'],
                enable_debug=False
            )
            
            for record in records:
                logger.log_wagon(record)
            logger.close()  # Close before reading
            
            # Read JSON
            json_path = Path(tmpdir) / 'logs.json'
            with open(json_path, 'r', encoding='utf-8') as f:
                json_records = json.load(f)
            
            # Read CSV
            csv_path = Path(tmpdir) / 'logs.csv'
            with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                csv_rows = list(reader)
            
            # Same number of records
            assert len(json_records) == len(csv_rows) == len(records)
            
            # Each record should match
            for i, (json_rec, csv_rec) in enumerate(zip(json_records, csv_rows)):
                assert json_rec['wagon_id'] == int(csv_rec['wagon_id'])
                assert json_rec['count_index'] == int(csv_rec['count_index'])
                assert json_rec['deblur_applied'] == (csv_rec['deblur_applied'] == 'True')


class TestDebugFrameToggle:
    """
    Property 16: Debug Frame Toggle
    
    For any pipeline run:
    - If enable_debug_frames is True, debug frames SHALL be saved to outputs/debug_frames/
    - If enable_debug_frames is False, no debug frames SHALL be created
    
    Validates: Requirements 10.5
    """

    @given(
        frame_indices=st.lists(st.integers(min_value=0, max_value=10000), min_size=1, max_size=5, unique=True),
        enable_debug=st.booleans()
    )
    @settings(max_examples=100)
    def test_debug_frames_created_based_on_toggle(self, frame_indices, enable_debug):
        """
        Feature: railway-wagon-inspection, Property 16: Debug Frame Toggle
        
        Test with enable_debug_frames True and False, verify frames are
        created/not created accordingly.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = InspectionLogger(
                output_dir=tmpdir,
                formats=['json'],
                enable_debug=enable_debug
            )
            
            # Create a simple test frame
            test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Save debug frames
            saved_paths = []
            for idx in frame_indices:
                result = logger.save_debug_frame(test_frame, idx)
                saved_paths.append(result)
            
            logger.close()
            
            debug_frames_dir = Path(tmpdir) / 'debug_frames'
            
            if enable_debug:
                # Debug frames should be created
                assert debug_frames_dir.exists()
                
                # All save operations should return paths
                for path in saved_paths:
                    assert path is not None
                    assert os.path.exists(path)
                
                # Count files in debug_frames directory
                files = list(debug_frames_dir.glob('*.jpg'))
                assert len(files) == len(frame_indices)
            else:
                # No debug frames should be created
                for path in saved_paths:
                    assert path is None
                
                # Directory might not exist or should be empty
                if debug_frames_dir.exists():
                    files = list(debug_frames_dir.glob('*.jpg'))
                    assert len(files) == 0

    @given(frame_index=st.integers(min_value=0, max_value=10000))
    @settings(max_examples=100)
    def test_debug_frame_with_annotations(self, frame_index):
        """
        Feature: railway-wagon-inspection, Property 16: Debug Frame Toggle
        
        Verify debug frames with annotations are saved correctly when enabled.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = InspectionLogger(
                output_dir=tmpdir,
                formats=['json'],
                enable_debug=True
            )
            
            # Create a test frame
            test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Create annotations
            annotations = [
                {
                    'bbox': BoundingBox(x1=100, y1=100, x2=200, y2=200),
                    'label': 'wagon_1',
                    'color': (0, 255, 0)
                },
                {
                    'bbox': {'x1': 300, 'y1': 150, 'x2': 400, 'y2': 250},
                    'label': 'damage',
                    'color': (0, 0, 255)
                }
            ]
            
            result = logger.save_debug_frame(test_frame, frame_index, annotations)
            logger.close()
            
            assert result is not None
            assert os.path.exists(result)
            
            # Verify the file is a valid image
            import cv2
            saved_frame = cv2.imread(result)
            assert saved_frame is not None
            assert saved_frame.shape == test_frame.shape
