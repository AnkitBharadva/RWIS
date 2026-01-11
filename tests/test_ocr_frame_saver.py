"""
Property-based tests for OCR frame saver module.

Feature: ocr-visual-enhancements
Property 9: Metadata File Contains Required Fields
Validates: Requirements 3.3, 6.2
"""

import pytest
import tempfile
import json
import os
import numpy as np
from hypothesis import given, strategies as st, settings
from datetime import datetime
from pathlib import Path

from dashboard.ocr_frame_saver import OCRFrameSaver
from utils.data_models import OCRDetection, OCRFrameMetadata, BoundingBox


# Strategy for generating frame indices
frame_index_strategy = st.integers(min_value=0, max_value=1000000)

# Strategy for generating wagon IDs
wagon_id_strategy = st.integers(min_value=0, max_value=10000)

# Strategy for generating confidence values
confidence_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)

# Strategy for generating text strings
text_strategy = st.text(
    alphabet=st.characters(whitelist_categories=('L', 'N', 'P', 'S')),
    min_size=1,
    max_size=50
)

# Strategy for generating bounding box coordinates
bbox_coord_strategy = st.integers(min_value=0, max_value=1000)


@st.composite
def bounding_box_strategy(draw):
    """Generate a valid BoundingBox with x1 < x2 and y1 < y2."""
    x1 = draw(bbox_coord_strategy)
    y1 = draw(bbox_coord_strategy)
    x2 = draw(st.integers(min_value=x1 + 1, max_value=x1 + 500))
    y2 = draw(st.integers(min_value=y1 + 1, max_value=y1 + 500))
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


@st.composite
def ocr_detection_strategy(draw):
    """Generate an OCRDetection with random values."""
    return OCRDetection(
        text=draw(text_strategy),
        confidence=draw(confidence_strategy),
        bbox=draw(bounding_box_strategy()),
        wagon_id=draw(wagon_id_strategy),
        frame_index=draw(frame_index_strategy)
    )


@st.composite
def detection_dict_strategy(draw):
    """Generate a detection dictionary as stored in metadata."""
    bbox = draw(bounding_box_strategy())
    return {
        "text": draw(text_strategy),
        "confidence": draw(confidence_strategy),
        "bbox": {
            "x1": bbox.x1,
            "y1": bbox.y1,
            "x2": bbox.x2,
            "y2": bbox.y2
        }
    }


@st.composite
def ocr_frame_metadata_strategy(draw):
    """Generate an OCRFrameMetadata with random values."""
    timestamp = datetime.now().isoformat()
    detections = draw(st.lists(detection_dict_strategy(), min_size=0, max_size=5))
    
    return OCRFrameMetadata(
        timestamp=timestamp,
        frame_index=draw(frame_index_strategy),
        wagon_id=draw(wagon_id_strategy),
        detections=detections,
        deblur_applied=draw(st.booleans()),
        illumination_applied=draw(st.booleans()),
        blur_score=draw(st.one_of(st.none(), st.floats(min_value=0.0, max_value=1000.0, allow_nan=False))),
        luminance_level=draw(st.one_of(st.none(), st.floats(min_value=0.0, max_value=255.0, allow_nan=False)))
    )


class TestMetadataFileContainsRequiredFields:
    """
    Property 9: Metadata File Contains Required Fields
    
    For any saved OCR frame, the accompanying JSON metadata file SHALL contain:
    - timestamp
    - frame_index
    - wagon_id
    - detections array
    - deblur_applied boolean
    
    Feature: ocr-visual-enhancements, Property 9: Metadata File Contains Required Fields
    Validates: Requirements 3.3, 6.2
    """

    @given(metadata=ocr_frame_metadata_strategy())
    @settings(max_examples=100)
    def test_metadata_contains_all_required_fields(self, metadata):
        """
        Feature: ocr-visual-enhancements, Property 9: Metadata File Contains Required Fields
        
        Verify that saved metadata contains all required fields.
        Validates: Requirements 3.3, 6.2
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            saver = OCRFrameSaver(output_directory=tmpdir)
            
            # Create a test frame
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            
            # Save the frame
            frame_path, metadata_path = saver.save_ocr_frame(
                frame=frame,
                ocr_detections=[],
                metadata=metadata
            )
            
            # Verify frame was saved
            assert frame_path is not None, "Frame should be saved"
            assert metadata_path is not None, "Metadata should be saved"
            
            # Load and verify metadata
            with open(metadata_path, 'r') as f:
                saved_metadata = json.load(f)
            
            # Property: Metadata must contain timestamp
            assert "timestamp" in saved_metadata, \
                "Metadata must contain 'timestamp' field"
            
            # Property: Metadata must contain frame_index
            assert "frame_index" in saved_metadata, \
                "Metadata must contain 'frame_index' field"
            assert saved_metadata["frame_index"] == metadata.frame_index, \
                f"frame_index mismatch: {saved_metadata['frame_index']} != {metadata.frame_index}"
            
            # Property: Metadata must contain wagon_id
            assert "wagon_id" in saved_metadata, \
                "Metadata must contain 'wagon_id' field"
            assert saved_metadata["wagon_id"] == metadata.wagon_id, \
                f"wagon_id mismatch: {saved_metadata['wagon_id']} != {metadata.wagon_id}"
            
            # Property: Metadata must contain detections array
            assert "detections" in saved_metadata, \
                "Metadata must contain 'detections' field"
            assert isinstance(saved_metadata["detections"], list), \
                "detections must be a list"
            
            # Property: Metadata must contain deblur_applied boolean
            assert "deblur_applied" in saved_metadata, \
                "Metadata must contain 'deblur_applied' field"
            assert isinstance(saved_metadata["deblur_applied"], bool), \
                "deblur_applied must be a boolean"
            assert saved_metadata["deblur_applied"] == metadata.deblur_applied, \
                f"deblur_applied mismatch: {saved_metadata['deblur_applied']} != {metadata.deblur_applied}"

    @given(metadata=ocr_frame_metadata_strategy())
    @settings(max_examples=100)
    def test_metadata_detections_preserve_content(self, metadata):
        """
        Feature: ocr-visual-enhancements, Property 9: Metadata File Contains Required Fields
        
        Verify that detection content is preserved in saved metadata.
        Validates: Requirements 3.3
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            saver = OCRFrameSaver(output_directory=tmpdir)
            
            # Create a test frame
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            
            # Save the frame
            frame_path, metadata_path = saver.save_ocr_frame(
                frame=frame,
                ocr_detections=[],
                metadata=metadata
            )
            
            if metadata_path is not None:
                # Load saved metadata
                with open(metadata_path, 'r') as f:
                    saved_metadata = json.load(f)
                
                # Property: Detections count must match
                assert len(saved_metadata["detections"]) == len(metadata.detections), \
                    f"Detection count mismatch: {len(saved_metadata['detections'])} != {len(metadata.detections)}"
                
                # Property: Each detection must preserve its content
                for i, (saved_det, orig_det) in enumerate(zip(saved_metadata["detections"], metadata.detections)):
                    assert saved_det["text"] == orig_det["text"], \
                        f"Detection {i} text mismatch"
                    assert abs(saved_det["confidence"] - orig_det["confidence"]) < 0.0001, \
                        f"Detection {i} confidence mismatch"

    @given(
        frame_index=frame_index_strategy,
        wagon_id=wagon_id_strategy,
        deblur_applied=st.booleans()
    )
    @settings(max_examples=100)
    def test_metadata_round_trip_preserves_values(self, frame_index, wagon_id, deblur_applied):
        """
        Feature: ocr-visual-enhancements, Property 9: Metadata File Contains Required Fields
        
        Verify that metadata values are preserved through save/load cycle.
        Validates: Requirements 3.3, 6.2
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            saver = OCRFrameSaver(output_directory=tmpdir)
            
            # Create metadata with specific values
            metadata = OCRFrameMetadata(
                timestamp=datetime.now().isoformat(),
                frame_index=frame_index,
                wagon_id=wagon_id,
                detections=[],
                deblur_applied=deblur_applied,
                illumination_applied=False
            )
            
            # Create a test frame
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            
            # Save the frame
            frame_path, metadata_path = saver.save_ocr_frame(
                frame=frame,
                ocr_detections=[],
                metadata=metadata
            )
            
            assert metadata_path is not None, "Metadata should be saved"
            
            # Load and verify
            with open(metadata_path, 'r') as f:
                saved_metadata = json.load(f)
            
            # Property: Values must be preserved
            assert saved_metadata["frame_index"] == frame_index
            assert saved_metadata["wagon_id"] == wagon_id
            assert saved_metadata["deblur_applied"] == deblur_applied


class TestOCRFrameSaverFilename:
    """Unit tests for OCRFrameSaver filename generation."""

    @given(
        frame_index=frame_index_strategy,
        wagon_id=wagon_id_strategy
    )
    @settings(max_examples=100)
    def test_filename_contains_required_components(self, frame_index, wagon_id):
        """
        Verify that generated filenames contain timestamp, frame_index, and wagon_id.
        Validates: Requirements 3.2
        """
        saver = OCRFrameSaver()
        timestamp = datetime.now()
        
        filename = saver.generate_filename(timestamp, frame_index, wagon_id)
        
        # Property: Filename must end with .jpg
        assert filename.endswith(".jpg"), \
            f"Filename must end with .jpg, got: {filename}"
        
        # Property: Filename must contain frame index
        assert str(frame_index) in filename, \
            f"Filename must contain frame index {frame_index}, got: {filename}"
        
        # Property: Filename must contain wagon ID
        assert str(wagon_id) in filename, \
            f"Filename must contain wagon ID {wagon_id}, got: {filename}"
        
        # Property: Filename must contain timestamp components
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        assert timestamp_str in filename, \
            f"Filename must contain timestamp {timestamp_str}, got: {filename}"


class TestOCRFrameSaverDirectory:
    """Unit tests for OCRFrameSaver directory handling."""

    def test_creates_output_directory_if_not_exists(self):
        """
        Verify that output directory is created if it doesn't exist.
        Validates: Requirements 3.5
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = os.path.join(tmpdir, "new_subdir", "ocr_frames")
            
            # Directory should not exist yet
            assert not os.path.exists(new_dir)
            
            # Create saver with new directory
            saver = OCRFrameSaver(output_directory=new_dir)
            
            # Directory should now exist
            assert os.path.exists(new_dir), \
                f"Output directory should be created: {new_dir}"

    def test_default_output_directory(self):
        """Verify default output directory is set correctly."""
        saver = OCRFrameSaver()
        assert saver.output_directory == OCRFrameSaver.DEFAULT_OUTPUT_DIR

    def test_custom_output_directory(self):
        """Verify custom output directory is used."""
        with tempfile.TemporaryDirectory() as tmpdir:
            saver = OCRFrameSaver(output_directory=tmpdir)
            assert saver.output_directory == tmpdir


class TestOCRFrameSaverEdgeCases:
    """Unit tests for OCRFrameSaver edge cases."""

    def test_save_with_none_frame_returns_none(self):
        """Verify that saving None frame returns (None, None)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            saver = OCRFrameSaver(output_directory=tmpdir)
            
            metadata = OCRFrameMetadata(
                timestamp=datetime.now().isoformat(),
                frame_index=0,
                wagon_id=1,
                detections=[],
                deblur_applied=False,
                illumination_applied=False
            )
            
            frame_path, metadata_path = saver.save_ocr_frame(
                frame=None,
                ocr_detections=[],
                metadata=metadata
            )
            
            assert frame_path is None
            assert metadata_path is None

    def test_save_with_empty_frame_returns_none(self):
        """Verify that saving empty frame returns (None, None)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            saver = OCRFrameSaver(output_directory=tmpdir)
            
            metadata = OCRFrameMetadata(
                timestamp=datetime.now().isoformat(),
                frame_index=0,
                wagon_id=1,
                detections=[],
                deblur_applied=False,
                illumination_applied=False
            )
            
            # Empty array
            frame = np.array([])
            
            frame_path, metadata_path = saver.save_ocr_frame(
                frame=frame,
                ocr_detections=[],
                metadata=metadata
            )
            
            assert frame_path is None
            assert metadata_path is None
