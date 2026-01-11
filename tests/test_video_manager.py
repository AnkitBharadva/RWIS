"""
Property-based tests for video manager module.

Feature: streamlit-dashboard
Validates: Requirements 2.5, 7.1, 7.5
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings
from unittest.mock import MagicMock, patch

from dashboard.video_manager import VideoManager


# Strategy for generating valid frame skip intervals
frame_skip_strategy = st.integers(min_value=1, max_value=10)


class TestFrameSkipIntervalEnforcement:
    """
    Property 1: Frame Skip Interval Enforcement
    
    For any video processing session with configured frame_skip interval N,
    the dashboard SHALL process exactly 1 frame for every N frames captured.
    The frame_count modulo N determines which frames are processed.
    
    Validates: Requirements 7.1
    """

    @given(frame_skip=frame_skip_strategy, num_reads=st.integers(min_value=1, max_value=20))
    @settings(max_examples=100)
    def test_frame_skip_interval_enforcement(self, frame_skip, num_reads):
        """
        Feature: streamlit-dashboard, Property 1: Frame Skip Interval Enforcement
        
        Generate random frame counts and skip intervals.
        Verify exactly 1 frame processed per N frames.
        """
        manager = VideoManager(frame_skip=frame_skip)
        
        # Create a mock VideoCapture that always succeeds
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.grab.return_value = True
        
        # Create a dummy frame
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, dummy_frame)
        
        manager.cap = mock_cap
        
        # Track how many times read() is called (actual frame decoding)
        # and how many times grab() is called (frame skipping)
        initial_frame_count = manager.frame_count
        
        # Perform multiple read_frame calls
        frames_returned = 0
        for _ in range(num_reads):
            success, frame = manager.read_frame()
            if success and frame is not None:
                frames_returned += 1
        
        # Each read_frame call should:
        # - Call grab() (frame_skip - 1) times
        # - Call read() once
        # Total frames consumed per read_frame = frame_skip
        
        expected_grab_calls = num_reads * (frame_skip - 1)
        expected_read_calls = num_reads
        
        assert mock_cap.grab.call_count == expected_grab_calls, \
            f"Expected {expected_grab_calls} grab calls, got {mock_cap.grab.call_count}"
        assert mock_cap.read.call_count == expected_read_calls, \
            f"Expected {expected_read_calls} read calls, got {mock_cap.read.call_count}"
        
        # Verify frame_count is incremented correctly
        # Each read_frame increments by frame_skip (grab calls + 1 read)
        expected_frame_count = initial_frame_count + (num_reads * frame_skip)
        assert manager.frame_count == expected_frame_count, \
            f"Expected frame_count={expected_frame_count}, got {manager.frame_count}"

    @given(frame_skip=frame_skip_strategy)
    @settings(max_examples=100)
    def test_frame_skip_minimum_is_one(self, frame_skip):
        """
        Feature: streamlit-dashboard, Property 1: Frame Skip Interval Enforcement
        
        Verify frame_skip is always at least 1 (no negative or zero values).
        """
        manager = VideoManager(frame_skip=frame_skip)
        assert manager.frame_skip >= 1, \
            f"frame_skip should be at least 1, got {manager.frame_skip}"

    @given(invalid_skip=st.integers(min_value=-100, max_value=0))
    @settings(max_examples=100)
    def test_invalid_frame_skip_clamped_to_one(self, invalid_skip):
        """
        Feature: streamlit-dashboard, Property 1: Frame Skip Interval Enforcement
        
        Verify invalid frame_skip values (<=0) are clamped to 1.
        """
        manager = VideoManager(frame_skip=invalid_skip)
        assert manager.frame_skip == 1, \
            f"Invalid frame_skip {invalid_skip} should be clamped to 1, got {manager.frame_skip}"


class TestVideoResourceCleanup:
    """
    Property 2: Video Resource Cleanup
    
    For any video capture session, when release() is called,
    the cv2.VideoCapture resource SHALL be released.
    After release, is_connected() SHALL return False.
    
    Validates: Requirements 2.5, 7.5
    """

    @given(frame_skip=frame_skip_strategy)
    @settings(max_examples=100)
    def test_release_cleans_up_resources(self, frame_skip):
        """
        Feature: streamlit-dashboard, Property 2: Video Resource Cleanup
        
        Generate random video sessions.
        Verify release() properly cleans up, is_connected() returns False after.
        """
        manager = VideoManager(frame_skip=frame_skip)
        
        # Create a mock VideoCapture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        manager.cap = mock_cap
        manager.frame_count = 42  # Set some arbitrary frame count
        
        # Verify connected before release
        assert manager.is_connected() is True, \
            "Should be connected before release"
        
        # Release resources
        manager.release()
        
        # Verify cleanup
        mock_cap.release.assert_called_once()
        assert manager.cap is None, \
            "cap should be None after release"
        assert manager.frame_count == 0, \
            "frame_count should be reset to 0 after release"
        assert manager.is_connected() is False, \
            "is_connected() should return False after release"

    @given(frame_skip=frame_skip_strategy)
    @settings(max_examples=100)
    def test_release_idempotent(self, frame_skip):
        """
        Feature: streamlit-dashboard, Property 2: Video Resource Cleanup
        
        Verify calling release() multiple times is safe (idempotent).
        """
        manager = VideoManager(frame_skip=frame_skip)
        
        # Create a mock VideoCapture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        manager.cap = mock_cap
        
        # Release multiple times - should not raise
        manager.release()
        manager.release()
        manager.release()
        
        # Verify state is consistent
        assert manager.cap is None
        assert manager.is_connected() is False
        assert manager.frame_count == 0

    @given(frame_skip=frame_skip_strategy)
    @settings(max_examples=100)
    def test_is_connected_false_when_cap_none(self, frame_skip):
        """
        Feature: streamlit-dashboard, Property 2: Video Resource Cleanup
        
        Verify is_connected() returns False when cap is None.
        """
        manager = VideoManager(frame_skip=frame_skip)
        
        # Initially cap is None
        assert manager.cap is None
        assert manager.is_connected() is False

    @given(frame_skip=frame_skip_strategy)
    @settings(max_examples=100)
    def test_is_connected_false_when_cap_not_opened(self, frame_skip):
        """
        Feature: streamlit-dashboard, Property 2: Video Resource Cleanup
        
        Verify is_connected() returns False when cap exists but is not opened.
        """
        manager = VideoManager(frame_skip=frame_skip)
        
        # Create a mock VideoCapture that is not opened
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        manager.cap = mock_cap
        
        assert manager.is_connected() is False

    @given(frame_skip=frame_skip_strategy)
    @settings(max_examples=100)
    def test_connect_releases_existing_capture(self, frame_skip):
        """
        Feature: streamlit-dashboard, Property 2: Video Resource Cleanup
        
        Verify connect() releases any existing capture before creating new one.
        """
        manager = VideoManager(frame_skip=frame_skip)
        
        # Create an existing mock capture
        existing_cap = MagicMock()
        existing_cap.isOpened.return_value = True
        manager.cap = existing_cap
        
        # Mock cv2.VideoCapture to return a new mock
        with patch('dashboard.video_manager.cv2.VideoCapture') as mock_video_capture:
            new_cap = MagicMock()
            new_cap.isOpened.return_value = True
            mock_video_capture.return_value = new_cap
            
            # Connect to a new source
            manager.connect("test_source")
            
            # Verify existing capture was released
            existing_cap.release.assert_called_once()
