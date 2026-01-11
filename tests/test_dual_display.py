"""
Property-based tests for Dual Video Display component.

Feature: dashboard-enhancements
Property 1: Dual Frame Synchronization
Validates: Requirements 1.2, 1.3, 1.4
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings
from unittest.mock import MagicMock, patch

from dashboard.dual_display import DualVideoDisplay


# Strategy for generating valid BGR frames
@st.composite
def valid_bgr_frame(draw, min_size=100, max_size=800):
    """Generate a valid BGR frame with random content."""
    height = draw(st.integers(min_value=min_size, max_value=max_size))
    width = draw(st.integers(min_value=min_size, max_value=max_size))
    # Generate random pixel values
    frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return frame


# Strategy for generating frame indices
frame_index_strategy = st.integers(min_value=0, max_value=100000)


# Strategy for generating frame pairs with the same index
@st.composite
def synchronized_frame_pair(draw):
    """Generate a pair of frames (raw and processed) with a frame index."""
    frame_index = draw(frame_index_strategy)
    raw_frame = draw(valid_bgr_frame())
    processed_frame = draw(valid_bgr_frame())
    return raw_frame, processed_frame, frame_index


class TestDualFrameSynchronization:
    """
    Property 1: Dual Frame Synchronization
    
    For any frame update cycle, both the raw frame display and processed frame
    display SHALL show frames from the same frame index. The raw frame SHALL
    contain no overlays or modifications, while the processed frame SHALL
    contain all detection annotations.
    
    Validates: Requirements 1.2, 1.3, 1.4
    """

    @given(data=synchronized_frame_pair())
    @settings(max_examples=100)
    def test_frame_index_synchronization(self, data):
        """
        Feature: dashboard-enhancements, Property 1: Dual Frame Synchronization
        
        Generate random frame pairs with indices.
        Verify both displays show same frame index.
        
        Validates: Requirements 1.2, 1.3, 1.4
        """
        raw_frame, processed_frame, frame_index = data
        
        display = DualVideoDisplay()
        
        # Mock streamlit components to avoid actual rendering
        with patch('dashboard.dual_display.st') as mock_st:
            mock_st.columns.return_value = [MagicMock(), MagicMock()]
            mock_st.markdown = MagicMock()
            mock_st.image = MagicMock()
            
            # Render the frames
            display.render(raw_frame, processed_frame, frame_index)
        
        # Verify the frame index is tracked correctly
        assert display.get_frame_index() == frame_index, \
            f"Frame index should be {frame_index}, got {display.get_frame_index()}"
        
        # Verify the current_frame_index attribute is synchronized
        assert display.current_frame_index == frame_index, \
            f"current_frame_index should be {frame_index}"

    @given(frame_indices=st.lists(frame_index_strategy, min_size=2, max_size=10))
    @settings(max_examples=100)
    def test_frame_index_updates_correctly(self, frame_indices):
        """
        Feature: dashboard-enhancements, Property 1: Dual Frame Synchronization
        
        Verify that frame index updates correctly across multiple render calls.
        """
        display = DualVideoDisplay()
        
        with patch('dashboard.dual_display.st') as mock_st:
            mock_st.columns.return_value = [MagicMock(), MagicMock()]
            mock_st.markdown = MagicMock()
            mock_st.image = MagicMock()
            
            for expected_index in frame_indices:
                # Create dummy frames
                raw_frame = np.zeros((100, 100, 3), dtype=np.uint8)
                processed_frame = np.zeros((100, 100, 3), dtype=np.uint8)
                
                # Render with the current index
                display.render(raw_frame, processed_frame, expected_index)
                
                # Verify index is updated
                assert display.get_frame_index() == expected_index, \
                    f"Frame index should be {expected_index} after render"

    @given(raw_frame=valid_bgr_frame(), processed_frame=valid_bgr_frame())
    @settings(max_examples=100)
    def test_raw_frame_not_modified(self, raw_frame, processed_frame):
        """
        Feature: dashboard-enhancements, Property 1: Dual Frame Synchronization
        
        Verify that the raw frame is not modified during rendering.
        The raw frame should remain unchanged (no overlays added).
        
        Validates: Requirement 1.2
        """
        display = DualVideoDisplay()
        
        # Make a copy of the raw frame before rendering
        raw_frame_copy = raw_frame.copy()
        
        with patch('dashboard.dual_display.st') as mock_st:
            mock_st.columns.return_value = [MagicMock(), MagicMock()]
            mock_st.markdown = MagicMock()
            mock_st.image = MagicMock()
            
            # Render the frames
            display.render(raw_frame, processed_frame, frame_index=0)
        
        # Verify raw frame was not modified
        assert np.array_equal(raw_frame, raw_frame_copy), \
            "Raw frame should not be modified during rendering"

    @given(frame_index=frame_index_strategy)
    @settings(max_examples=100)
    def test_none_frames_handled(self, frame_index):
        """
        Feature: dashboard-enhancements, Property 1: Dual Frame Synchronization
        
        Verify that None frames are handled gracefully with placeholders.
        
        Validates: Requirement 1.6
        """
        display = DualVideoDisplay()
        
        with patch('dashboard.dual_display.st') as mock_st:
            mock_st.columns.return_value = [MagicMock(), MagicMock()]
            mock_st.markdown = MagicMock()
            mock_st.image = MagicMock()
            
            # Render with None frames
            display.render(None, None, frame_index)
        
        # Verify frame index is still tracked
        assert display.get_frame_index() == frame_index, \
            "Frame index should be tracked even with None frames"
        
        # Verify markdown was called for placeholders
        assert mock_st.markdown.called, \
            "Placeholders should be rendered when frames are None"

    @given(raw_frame=valid_bgr_frame())
    @settings(max_examples=100)
    def test_valid_frame_detection(self, raw_frame):
        """
        Feature: dashboard-enhancements, Property 1: Dual Frame Synchronization
        
        Verify that valid frames are correctly identified.
        """
        display = DualVideoDisplay()
        
        # Valid frame should be detected
        assert display._is_valid_frame(raw_frame), \
            "Valid BGR frame should be detected as valid"
        
        # None should be invalid
        assert not display._is_valid_frame(None), \
            "None should be detected as invalid"
        
        # Empty frame should be invalid
        empty_frame = np.array([])
        assert not display._is_valid_frame(empty_frame), \
            "Empty frame should be detected as invalid"

    @given(data=synchronized_frame_pair())
    @settings(max_examples=100)
    def test_labels_are_correct(self, data):
        """
        Feature: dashboard-enhancements, Property 1: Dual Frame Synchronization
        
        Verify that the correct labels are used for raw and processed frames.
        
        Validates: Requirement 1.5
        """
        raw_frame, processed_frame, frame_index = data
        
        display = DualVideoDisplay()
        
        # Verify label constants
        assert display.RAW_LABEL == "Raw Input", \
            "Raw label should be 'Raw Input'"
        assert display.PROCESSED_LABEL == "Processed Output", \
            "Processed label should be 'Processed Output'"

    def test_initial_frame_index_is_zero(self):
        """
        Verify that initial frame index is zero.
        """
        display = DualVideoDisplay()
        
        assert display.current_frame_index == 0, \
            "Initial frame index should be 0"
        assert display.get_frame_index() == 0, \
            "get_frame_index() should return 0 initially"
