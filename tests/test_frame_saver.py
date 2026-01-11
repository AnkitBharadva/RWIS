"""
Property-based tests for FrameSaver component.

Feature: dashboard-enhancements
Property 5: Conditional Frame Saving
Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 8.4
"""

import pytest
import tempfile
import shutil
import os
import json
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from datetime import datetime, timedelta
from pathlib import Path

from dashboard.frame_saver import FrameSaver
from dashboard.models import FrameSaveConfig, FrameMetadata, ProcessingType


# Strategy for generating boolean flags
bool_strategy = st.booleans()

# Strategy for generating frame indices
frame_index_strategy = st.integers(min_value=0, max_value=1000000)

# Strategy for generating wagon IDs (optional)
wagon_id_strategy = st.one_of(st.none(), st.integers(min_value=0, max_value=10000))

# Strategy for generating processing types
processing_type_strategy = st.lists(
    st.sampled_from(["deblur", "clahe", "gamma", "ocr"]),
    min_size=0,
    max_size=4,
    unique=True
)

# Strategy for generating timestamps
timestamp_strategy = st.datetimes(
    min_value=datetime(2020, 1, 1),
    max_value=datetime(2030, 12, 31)
)


# Strategy for generating FrameSaveConfig
@st.composite
def frame_save_config_strategy(draw):
    """Generate a FrameSaveConfig with random settings."""
    return FrameSaveConfig(
        enabled=draw(bool_strategy),
        save_on_deblur=draw(bool_strategy),
        save_on_illumination=draw(bool_strategy),
        save_on_ocr=draw(bool_strategy),
        output_directory=draw(st.just("outputs/test_frames"))
    )


# Strategy for generating FrameMetadata
@st.composite
def frame_metadata_strategy(draw):
    """Generate a FrameMetadata with random values."""
    return FrameMetadata(
        timestamp=draw(timestamp_strategy),
        frame_index=draw(frame_index_strategy),
        processing_applied=draw(processing_type_strategy),
        wagon_id=draw(wagon_id_strategy)
    )


# Strategy for generating processing event flags
@st.composite
def processing_event_strategy(draw):
    """Generate a set of processing event flags."""
    return {
        "deblur_applied": draw(bool_strategy),
        "illumination_applied": draw(bool_strategy),
        "ocr_performed": draw(bool_strategy)
    }


class TestConditionalFrameSaving:
    """
    Property 5: Conditional Frame Saving
    
    For any frame processing event:
    - If frame saving is disabled, no files SHALL be created
    - If deblur is applied AND save_on_deblur is True, the frame SHALL be saved
    - If illumination is applied AND save_on_illumination is True, the frame SHALL be saved
    - If OCR is performed AND save_on_ocr is True, the frame SHALL be saved
    - Saved files SHALL include metadata indicating processing type
    
    Feature: dashboard-enhancements, Property 5: Conditional Frame Saving
    Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 8.4
    """

    @given(
        config=frame_save_config_strategy(),
        events=processing_event_strategy()
    )
    @settings(max_examples=100)
    def test_disabled_saving_never_saves(self, config, events):
        """
        Feature: dashboard-enhancements, Property 5: Conditional Frame Saving
        
        Verify that when frame saving is disabled, should_save() always returns False.
        Validates: Requirements 8.4
        """
        # Force disabled
        config.enabled = False
        
        saver = FrameSaver(config)
        
        # Property: should_save() must return False when disabled
        result = saver.should_save(
            deblur_applied=events["deblur_applied"],
            illumination_applied=events["illumination_applied"],
            ocr_performed=events["ocr_performed"]
        )
        
        assert result is False, \
            "should_save() must return False when frame saving is disabled"

    @given(
        save_on_deblur=bool_strategy,
        save_on_illumination=bool_strategy,
        save_on_ocr=bool_strategy
    )
    @settings(max_examples=100)
    def test_deblur_trigger_respects_config(self, save_on_deblur, save_on_illumination, save_on_ocr):
        """
        Feature: dashboard-enhancements, Property 5: Conditional Frame Saving
        
        Verify that deblur trigger respects save_on_deblur config.
        Validates: Requirements 5.1
        """
        config = FrameSaveConfig(
            enabled=True,
            save_on_deblur=save_on_deblur,
            save_on_illumination=save_on_illumination,
            save_on_ocr=save_on_ocr,
            output_directory="outputs/test_frames"
        )
        
        saver = FrameSaver(config)
        
        # Test with only deblur applied
        result = saver.should_save(
            deblur_applied=True,
            illumination_applied=False,
            ocr_performed=False
        )
        
        # Property: Result should match save_on_deblur config
        if save_on_deblur:
            assert result is True, \
                "should_save() must return True when deblur applied and save_on_deblur is True"
        else:
            # May still be True if other triggers are enabled, but with only deblur applied
            # and save_on_deblur False, result should be False
            assert result is False, \
                "should_save() must return False when deblur applied but save_on_deblur is False"

    @given(
        save_on_deblur=bool_strategy,
        save_on_illumination=bool_strategy,
        save_on_ocr=bool_strategy
    )
    @settings(max_examples=100)
    def test_illumination_trigger_respects_config(self, save_on_deblur, save_on_illumination, save_on_ocr):
        """
        Feature: dashboard-enhancements, Property 5: Conditional Frame Saving
        
        Verify that illumination trigger respects save_on_illumination config.
        Validates: Requirements 5.2
        """
        config = FrameSaveConfig(
            enabled=True,
            save_on_deblur=save_on_deblur,
            save_on_illumination=save_on_illumination,
            save_on_ocr=save_on_ocr,
            output_directory="outputs/test_frames"
        )
        
        saver = FrameSaver(config)
        
        # Test with only illumination applied
        result = saver.should_save(
            deblur_applied=False,
            illumination_applied=True,
            ocr_performed=False
        )
        
        # Property: Result should match save_on_illumination config
        if save_on_illumination:
            assert result is True, \
                "should_save() must return True when illumination applied and save_on_illumination is True"
        else:
            assert result is False, \
                "should_save() must return False when illumination applied but save_on_illumination is False"

    @given(
        save_on_deblur=bool_strategy,
        save_on_illumination=bool_strategy,
        save_on_ocr=bool_strategy
    )
    @settings(max_examples=100)
    def test_ocr_trigger_respects_config(self, save_on_deblur, save_on_illumination, save_on_ocr):
        """
        Feature: dashboard-enhancements, Property 5: Conditional Frame Saving
        
        Verify that OCR trigger respects save_on_ocr config.
        Validates: Requirements 5.3
        """
        config = FrameSaveConfig(
            enabled=True,
            save_on_deblur=save_on_deblur,
            save_on_illumination=save_on_illumination,
            save_on_ocr=save_on_ocr,
            output_directory="outputs/test_frames"
        )
        
        saver = FrameSaver(config)
        
        # Test with only OCR performed
        result = saver.should_save(
            deblur_applied=False,
            illumination_applied=False,
            ocr_performed=True
        )
        
        # Property: Result should match save_on_ocr config
        if save_on_ocr:
            assert result is True, \
                "should_save() must return True when OCR performed and save_on_ocr is True"
        else:
            assert result is False, \
                "should_save() must return False when OCR performed but save_on_ocr is False"

    @given(
        config=frame_save_config_strategy(),
        events=processing_event_strategy()
    )
    @settings(max_examples=100)
    def test_no_processing_never_saves(self, config, events):
        """
        Feature: dashboard-enhancements, Property 5: Conditional Frame Saving
        
        Verify that when no processing is applied, should_save() returns False.
        """
        # Force enabled to test the logic
        config.enabled = True
        
        saver = FrameSaver(config)
        
        # Test with no processing applied
        result = saver.should_save(
            deblur_applied=False,
            illumination_applied=False,
            ocr_performed=False
        )
        
        # Property: should_save() must return False when no processing applied
        assert result is False, \
            "should_save() must return False when no processing is applied"

    @given(
        config=frame_save_config_strategy(),
        events=processing_event_strategy()
    )
    @settings(max_examples=100)
    def test_should_save_logic_correctness(self, config, events):
        """
        Feature: dashboard-enhancements, Property 5: Conditional Frame Saving
        
        Verify the complete should_save() logic matches expected behavior.
        Validates: Requirements 5.1, 5.2, 5.3, 8.4
        """
        saver = FrameSaver(config)
        
        result = saver.should_save(
            deblur_applied=events["deblur_applied"],
            illumination_applied=events["illumination_applied"],
            ocr_performed=events["ocr_performed"]
        )
        
        # Calculate expected result
        if not config.enabled:
            expected = False
        else:
            expected = (
                (events["deblur_applied"] and config.save_on_deblur) or
                (events["illumination_applied"] and config.save_on_illumination) or
                (events["ocr_performed"] and config.save_on_ocr)
            )
        
        # Property: Result must match expected logic
        assert result == expected, \
            f"should_save() returned {result}, expected {expected} for config={config}, events={events}"

    @given(metadata=frame_metadata_strategy())
    @settings(max_examples=100)
    def test_filename_format(self, metadata):
        """
        Feature: dashboard-enhancements, Property 5: Conditional Frame Saving
        
        Verify that generated filenames follow the expected format.
        Validates: Requirements 5.4
        """
        config = FrameSaveConfig(enabled=True, output_directory="outputs/test_frames")
        saver = FrameSaver(config)
        
        filename = saver._generate_filename(metadata)
        
        # Property: Filename must end with .jpg
        assert filename.endswith(".jpg"), \
            f"Filename must end with .jpg, got: {filename}"
        
        # Property: Filename must contain frame index
        assert str(metadata.frame_index) in filename, \
            f"Filename must contain frame index {metadata.frame_index}, got: {filename}"
        
        # Property: Filename must contain timestamp components
        timestamp_str = metadata.timestamp.strftime("%Y%m%d_%H%M%S")
        assert timestamp_str in filename, \
            f"Filename must contain timestamp {timestamp_str}, got: {filename}"

    @given(metadata=frame_metadata_strategy())
    @settings(max_examples=100)
    def test_saved_metadata_contains_processing_info(self, metadata):
        """
        Feature: dashboard-enhancements, Property 5: Conditional Frame Saving
        
        Verify that saved metadata includes processing type information.
        Validates: Requirements 5.5
        """
        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as tmpdir:
            config = FrameSaveConfig(enabled=True, output_directory=tmpdir)
            saver = FrameSaver(config)
            
            # Create a test frame
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            
            # Save the frame
            filepath = saver.save_frame(frame, metadata)
            
            if filepath is not None:
                # Check metadata file exists
                metadata_path = Path(filepath).with_suffix('.json')
                assert metadata_path.exists(), \
                    f"Metadata file should exist at {metadata_path}"
                
                # Load and verify metadata
                with open(metadata_path, 'r') as f:
                    saved_metadata = json.load(f)
                
                # Property: Metadata must contain processing_applied field
                assert "processing_applied" in saved_metadata, \
                    "Metadata must contain processing_applied field"
                
                # Property: processing_applied must match original
                assert saved_metadata["processing_applied"] == metadata.processing_applied, \
                    f"processing_applied mismatch: {saved_metadata['processing_applied']} != {metadata.processing_applied}"
                
                # Property: Metadata must contain frame_index
                assert "frame_index" in saved_metadata, \
                    "Metadata must contain frame_index field"
                assert saved_metadata["frame_index"] == metadata.frame_index, \
                    f"frame_index mismatch: {saved_metadata['frame_index']} != {metadata.frame_index}"
                
                # Property: Metadata must contain timestamp
                assert "timestamp" in saved_metadata, \
                    "Metadata must contain timestamp field"

    @given(
        config=frame_save_config_strategy(),
        metadata=frame_metadata_strategy()
    )
    @settings(max_examples=100)
    def test_disabled_save_frame_returns_none(self, config, metadata):
        """
        Feature: dashboard-enhancements, Property 5: Conditional Frame Saving
        
        Verify that save_frame() returns None when saving is disabled.
        Validates: Requirements 8.4
        """
        # Force disabled
        config.enabled = False
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config.output_directory = tmpdir
            saver = FrameSaver(config)
            
            # Create a test frame
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            
            # Attempt to save
            result = saver.save_frame(frame, metadata)
            
            # Property: Result must be None when disabled
            assert result is None, \
                "save_frame() must return None when saving is disabled"
            
            # Property: No files should be created
            files = list(Path(tmpdir).glob("*"))
            assert len(files) == 0, \
                f"No files should be created when saving is disabled, found: {files}"
