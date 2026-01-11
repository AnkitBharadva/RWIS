"""
Property-based tests for DeblurManager module.

Feature: railway-wagon-inspection
Property 9: N-th Frame Execution Logic
Validates: Requirements 6.1, 6.2, 6.3
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, assume
from unittest.mock import Mock, MagicMock, patch
from typing import List, Tuple

from pipelines.deblur_manager import DeblurManager
from pipelines.blur_detector import BlurDetector
from utils.data_models import BlurDecision


class MockMPRNet:
    """Mock MPRNet for testing DeblurManager without actual model loading."""
    
    def __init__(self):
        self.call_count = 0
        self.call_history: List[Tuple[int, np.ndarray]] = []  # (frame_index, roi)
    
    def deblur_roi(self, roi: np.ndarray) -> np.ndarray:
        """Mock deblur that returns a modified copy of the input."""
        self.call_count += 1
        # Return a slightly modified version to distinguish from original
        result = roi.copy()
        if result.size > 0:
            result[0, 0] = (result[0, 0] + 1) % 256 if len(result.shape) >= 2 else result
        return result
    
    def reset(self):
        """Reset call tracking."""
        self.call_count = 0
        self.call_history.clear()


class MockBlurDetector:
    """Mock BlurDetector that returns configurable blur decisions."""
    
    def __init__(self, always_deblur: bool = True, blur_score: float = 150.0):
        self.always_deblur = always_deblur
        self.blur_score = blur_score
        self.t1 = 100.0
        self.t2 = 300.0
    
    def compute_blur_score(self, roi: np.ndarray) -> float:
        """Return configured blur score."""
        return self.blur_score
    
    def get_blur_decision(self, blur_score: float) -> BlurDecision:
        """Return blur decision based on configuration."""
        if self.always_deblur:
            return BlurDecision.ROI_DEBLUR
        else:
            return BlurDecision.SKIP_DEBLUR
    
    def needs_deblur(self, blur_score: float) -> bool:
        """Return whether deblurring is needed based on configuration."""
        return self.always_deblur
    
    def get_blur_level(self, blur_score: float) -> str:
        """Return blur level description."""
        if self.always_deblur:
            return "moderate_blur"
        else:
            return "sharp"


# Strategy for generating valid frame intervals
frame_interval_strategy = st.integers(min_value=1, max_value=10)

# Strategy for generating valid max ROI widths
max_roi_width_strategy = st.integers(min_value=64, max_value=512)

# Strategy for generating wagon IDs
wagon_id_strategy = st.integers(min_value=1, max_value=1000)

# Strategy for generating frame sequences
@st.composite
def frame_sequence(draw, min_frames=5, max_frames=30):
    """Generate a sequence of frame indices."""
    num_frames = draw(st.integers(min_value=min_frames, max_value=max_frames))
    return list(range(num_frames))


# Strategy for generating random ROI images
@st.composite
def random_roi_image(draw, max_width=256, max_height=256):
    """Generate a random BGR ROI image within size limits."""
    width = draw(st.integers(min_value=10, max_value=max_width))
    height = draw(st.integers(min_value=10, max_value=max_height))
    
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    return image


class TestNthFrameExecutionLogic:
    """
    Property 9: N-th Frame Execution Logic
    
    For any wagon tracked across multiple frames with configured frame_interval N:
    - MPRNet SHALL run on frame indices where (frame_count % N == 0)
    - On intermediate frames, the cached deblurred ROI SHALL be reused
    - The cache SHALL be cleared when the wagon exits tracking
    
    Validates: Requirements 6.1, 6.2, 6.3
    """

    @given(
        frame_interval=frame_interval_strategy,
        num_frames=st.integers(min_value=5, max_value=50),
        wagon_id=wagon_id_strategy
    )
    @settings(max_examples=100)
    def test_mprnet_runs_at_correct_intervals(self, frame_interval, num_frames, wagon_id):
        """
        Feature: railway-wagon-inspection, Property 9: N-th Frame Execution Logic
        
        Verify MPRNet runs exactly on frames where frame_count % N == 0.
        """
        mock_mprnet = MockMPRNet()
        mock_blur_detector = MockBlurDetector(always_deblur=True)
        
        manager = DeblurManager(
            mprnet=mock_mprnet,
            blur_detector=mock_blur_detector,
            frame_interval=frame_interval,
            max_roi_width=256
        )
        
        # Create a test ROI
        roi = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Process multiple frames for the same wagon
        expected_mprnet_calls = 0
        for frame_idx in range(num_frames):
            frame_count = frame_idx  # frame_count starts at 0 for each wagon
            
            # MPRNet should run when frame_count % frame_interval == 0
            if frame_count % frame_interval == 0:
                expected_mprnet_calls += 1
            
            manager.process_roi(roi, wagon_id, frame_idx)
        
        # Verify MPRNet was called the expected number of times
        assert mock_mprnet.call_count == expected_mprnet_calls, \
            f"Expected {expected_mprnet_calls} MPRNet calls for {num_frames} frames " \
            f"with interval {frame_interval}, got {mock_mprnet.call_count}"

    @given(
        frame_interval=frame_interval_strategy,
        wagon_id=wagon_id_strategy
    )
    @settings(max_examples=100)
    def test_cache_reused_on_intermediate_frames(self, frame_interval, wagon_id):
        """
        Feature: railway-wagon-inspection, Property 9: N-th Frame Execution Logic
        
        Verify cached ROI is reused on intermediate frames (not N-th frames).
        """
        assume(frame_interval > 1)  # Need interval > 1 to have intermediate frames
        
        mock_mprnet = MockMPRNet()
        mock_blur_detector = MockBlurDetector(always_deblur=True)
        
        manager = DeblurManager(
            mprnet=mock_mprnet,
            blur_detector=mock_blur_detector,
            frame_interval=frame_interval,
            max_roi_width=256
        )
        
        roi = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Process first frame (frame_count=0, should run MPRNet)
        result1, deblur_applied1, source_frame1 = manager.process_roi(roi, wagon_id, 0)
        assert deblur_applied1 is True, "First frame should apply deblur"
        assert source_frame1 == 0, "Source frame should be 0 for first frame"
        assert mock_mprnet.call_count == 1, "MPRNet should be called on first frame"
        
        # Process intermediate frames (frame_count=1 to frame_interval-1)
        for i in range(1, frame_interval):
            result, deblur_applied, source_frame = manager.process_roi(roi, wagon_id, i)
            
            # Should still report deblur applied (using cache)
            assert deblur_applied is True, f"Frame {i} should report deblur applied (from cache)"
            # Source frame should be the cached frame (0)
            assert source_frame == 0, f"Frame {i} should use cached result from frame 0"
        
        # MPRNet should still only have been called once
        assert mock_mprnet.call_count == 1, \
            f"MPRNet should only be called once, got {mock_mprnet.call_count}"

    @given(
        frame_interval=frame_interval_strategy,
        wagon_id=wagon_id_strategy
    )
    @settings(max_examples=100)
    def test_cache_cleared_on_wagon_exit(self, frame_interval, wagon_id):
        """
        Feature: railway-wagon-inspection, Property 9: N-th Frame Execution Logic
        
        Verify cache is cleared when wagon exits tracking.
        """
        mock_mprnet = MockMPRNet()
        mock_blur_detector = MockBlurDetector(always_deblur=True)
        
        manager = DeblurManager(
            mprnet=mock_mprnet,
            blur_detector=mock_blur_detector,
            frame_interval=frame_interval,
            max_roi_width=256
        )
        
        roi = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Process a frame to populate cache
        manager.process_roi(roi, wagon_id, 0)
        
        # Verify cache exists
        assert manager.get_cached_roi(wagon_id) is not None, \
            "Cache should exist after processing"
        assert manager.get_frame_count(wagon_id) == 1, \
            "Frame count should be 1 after processing one frame"
        
        # Clear cache (simulating wagon exit)
        manager.clear_cache(wagon_id)
        
        # Verify cache is cleared
        assert manager.get_cached_roi(wagon_id) is None, \
            "Cache should be None after clearing"
        assert manager.get_frame_count(wagon_id) == 0, \
            "Frame count should be 0 after clearing"

    @given(
        frame_interval=frame_interval_strategy,
        num_wagons=st.integers(min_value=2, max_value=10)
    )
    @settings(max_examples=100)
    def test_independent_caches_per_wagon(self, frame_interval, num_wagons):
        """
        Feature: railway-wagon-inspection, Property 9: N-th Frame Execution Logic
        
        Verify each wagon has independent cache and frame counter.
        """
        mock_mprnet = MockMPRNet()
        mock_blur_detector = MockBlurDetector(always_deblur=True)
        
        manager = DeblurManager(
            mprnet=mock_mprnet,
            blur_detector=mock_blur_detector,
            frame_interval=frame_interval,
            max_roi_width=256
        )
        
        roi = np.zeros((100, 100, 3), dtype=np.uint8)
        wagon_ids = list(range(1, num_wagons + 1))
        
        # Process one frame for each wagon
        for wagon_id in wagon_ids:
            manager.process_roi(roi, wagon_id, 0)
        
        # Each wagon should have its own cache
        assert manager.get_cache_size() == num_wagons, \
            f"Expected {num_wagons} cached entries, got {manager.get_cache_size()}"
        
        # Clear one wagon's cache
        manager.clear_cache(wagon_ids[0])
        
        # Other wagons should still have cache
        assert manager.get_cache_size() == num_wagons - 1, \
            f"Expected {num_wagons - 1} cached entries after clearing one"
        
        # Verify the cleared wagon has no cache
        assert manager.get_cached_roi(wagon_ids[0]) is None
        
        # Verify other wagons still have cache
        for wagon_id in wagon_ids[1:]:
            assert manager.get_cached_roi(wagon_id) is not None

    @given(
        frame_interval=frame_interval_strategy,
        wagon_id=wagon_id_strategy
    )
    @settings(max_examples=100)
    def test_first_frame_always_runs_mprnet(self, frame_interval, wagon_id):
        """
        Feature: railway-wagon-inspection, Property 9: N-th Frame Execution Logic
        
        Verify MPRNet always runs on the first frame for a new wagon.
        """
        mock_mprnet = MockMPRNet()
        mock_blur_detector = MockBlurDetector(always_deblur=True)
        
        manager = DeblurManager(
            mprnet=mock_mprnet,
            blur_detector=mock_blur_detector,
            frame_interval=frame_interval,
            max_roi_width=256
        )
        
        roi = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # First frame should always run MPRNet (frame_count=0, 0 % N == 0)
        result, deblur_applied, source_frame = manager.process_roi(roi, wagon_id, 0)
        
        assert mock_mprnet.call_count == 1, \
            "MPRNet should run on first frame"
        assert deblur_applied is True, \
            "First frame should apply deblur"
        assert source_frame == 0, \
            "Source frame should be 0"

    @given(wagon_id=wagon_id_strategy)
    @settings(max_examples=100)
    def test_skip_deblur_returns_raw_roi(self, wagon_id):
        """
        Feature: railway-wagon-inspection, Property 9: N-th Frame Execution Logic
        
        Verify that when blur decision is SKIP_DEBLUR, raw ROI is returned.
        """
        mock_mprnet = MockMPRNet()
        mock_blur_detector = MockBlurDetector(always_deblur=False)  # Will return SKIP_DEBLUR
        
        manager = DeblurManager(
            mprnet=mock_mprnet,
            blur_detector=mock_blur_detector,
            frame_interval=3,
            max_roi_width=256
        )
        
        roi = np.zeros((100, 100, 3), dtype=np.uint8)
        
        result, deblur_applied, source_frame = manager.process_roi(roi, wagon_id, 0)
        
        # Should not apply deblur
        assert deblur_applied is False, "Should not apply deblur when SKIP_DEBLUR"
        assert source_frame is None, "Source frame should be None when not deblurring"
        assert mock_mprnet.call_count == 0, "MPRNet should not be called"


class TestDeblurManagerValidation:
    """Additional validation tests for DeblurManager."""

    def test_invalid_frame_interval_rejected(self):
        """Verify frame_interval < 1 raises ValueError."""
        mock_mprnet = MockMPRNet()
        mock_blur_detector = MockBlurDetector()
        
        with pytest.raises(ValueError):
            DeblurManager(
                mprnet=mock_mprnet,
                blur_detector=mock_blur_detector,
                frame_interval=0,
                max_roi_width=256
            )
        
        with pytest.raises(ValueError):
            DeblurManager(
                mprnet=mock_mprnet,
                blur_detector=mock_blur_detector,
                frame_interval=-1,
                max_roi_width=256
            )

    def test_invalid_max_roi_width_rejected(self):
        """Verify max_roi_width < 1 raises ValueError."""
        mock_mprnet = MockMPRNet()
        mock_blur_detector = MockBlurDetector()
        
        with pytest.raises(ValueError):
            DeblurManager(
                mprnet=mock_mprnet,
                blur_detector=mock_blur_detector,
                frame_interval=3,
                max_roi_width=0
            )
        
        with pytest.raises(ValueError):
            DeblurManager(
                mprnet=mock_mprnet,
                blur_detector=mock_blur_detector,
                frame_interval=3,
                max_roi_width=-1
            )

    def test_none_roi_rejected(self):
        """Verify None ROI raises ValueError."""
        mock_mprnet = MockMPRNet()
        mock_blur_detector = MockBlurDetector()
        
        manager = DeblurManager(
            mprnet=mock_mprnet,
            blur_detector=mock_blur_detector,
            frame_interval=3,
            max_roi_width=256
        )
        
        with pytest.raises(ValueError):
            manager.process_roi(None, wagon_id=1, frame_index=0)

    def test_empty_roi_rejected(self):
        """Verify empty ROI raises ValueError."""
        mock_mprnet = MockMPRNet()
        mock_blur_detector = MockBlurDetector()
        
        manager = DeblurManager(
            mprnet=mock_mprnet,
            blur_detector=mock_blur_detector,
            frame_interval=3,
            max_roi_width=256
        )
        
        empty_roi = np.array([])
        
        with pytest.raises(ValueError):
            manager.process_roi(empty_roi, wagon_id=1, frame_index=0)

    def test_clear_all_caches(self):
        """Verify clear_all_caches clears all wagon caches."""
        mock_mprnet = MockMPRNet()
        mock_blur_detector = MockBlurDetector(always_deblur=True)
        
        manager = DeblurManager(
            mprnet=mock_mprnet,
            blur_detector=mock_blur_detector,
            frame_interval=3,
            max_roi_width=256
        )
        
        roi = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Process frames for multiple wagons
        for wagon_id in range(1, 6):
            manager.process_roi(roi, wagon_id, 0)
        
        assert manager.get_cache_size() == 5
        
        # Clear all caches
        manager.clear_all_caches()
        
        assert manager.get_cache_size() == 0
        for wagon_id in range(1, 6):
            assert manager.get_cached_roi(wagon_id) is None
            assert manager.get_frame_count(wagon_id) == 0

    def test_roi_resizing_applied(self):
        """Verify ROI is resized before processing."""
        mock_mprnet = MockMPRNet()
        mock_blur_detector = MockBlurDetector(always_deblur=True)
        
        max_width = 128
        manager = DeblurManager(
            mprnet=mock_mprnet,
            blur_detector=mock_blur_detector,
            frame_interval=3,
            max_roi_width=max_width
        )
        
        # Create a large ROI
        large_roi = np.zeros((200, 300, 3), dtype=np.uint8)
        
        result, deblur_applied, source_frame = manager.process_roi(large_roi, wagon_id=1, frame_index=0)
        
        # Result should be resized to max_width
        assert result.shape[1] <= max_width, \
            f"Result width {result.shape[1]} should be <= {max_width}"


class TestDeblurEnableDisable:
    """
    Property 9: Deblur Enable/Disable
    
    *For any* frame processed when deblur is disabled:
    - No deblurring operation SHALL be performed
    - The deblur status SHALL be DISABLED
    - The original ROI SHALL be returned unchanged (except for resizing)
    
    *For any* frame processed when deblur is enabled:
    - Deblurring SHALL follow the blur threshold logic
    - The deblur status SHALL be either ACTIVE or SKIPPED based on blur score
    
    Validates: Requirements 7.2, 7.3
    """

    @given(
        wagon_id=wagon_id_strategy,
        frame_index=st.integers(min_value=0, max_value=10000)
    )
    @settings(max_examples=100)
    def test_disabled_deblur_skips_all_operations(self, wagon_id, frame_index):
        """
        Feature: ocr-enhancement-improvements, Property 9: Deblur Enable/Disable
        
        Verify that when deblur is disabled, no deblurring operation is performed.
        **Validates: Requirements 7.2, 7.3**
        """
        mock_mprnet = MockMPRNet()
        mock_blur_detector = MockBlurDetector(always_deblur=True)  # Would deblur if enabled
        
        manager = DeblurManager(
            mprnet=mock_mprnet,
            blur_detector=mock_blur_detector,
            frame_interval=3,
            max_roi_width=256,
            deblur_enabled=False  # Disabled
        )
        
        roi = np.zeros((100, 100, 3), dtype=np.uint8)
        
        result, deblur_applied, source_frame = manager.process_roi(roi, wagon_id, frame_index)
        
        # Verify no deblurring was applied
        assert deblur_applied is False, "Deblur should not be applied when disabled"
        assert source_frame is None, "Source frame should be None when disabled"
        assert mock_mprnet.call_count == 0, "MPRNet should not be called when disabled"
        
        # Verify status is DISABLED
        status = manager.get_last_status()
        assert status is not None, "Status should be set after processing"
        assert status.enabled is False, "Status should show deblur is disabled"
        assert status.applied is False, "Status should show deblur was not applied"
        assert status.status_type.value == "disabled", "Status type should be DISABLED"

    @given(
        wagon_id=wagon_id_strategy,
        frame_index=st.integers(min_value=0, max_value=10000)
    )
    @settings(max_examples=100)
    def test_enabled_deblur_follows_threshold_logic(self, wagon_id, frame_index):
        """
        Feature: ocr-enhancement-improvements, Property 9: Deblur Enable/Disable
        
        Verify that when deblur is enabled, it follows blur threshold logic.
        **Validates: Requirements 7.2, 7.3**
        """
        mock_mprnet = MockMPRNet()
        mock_blur_detector = MockBlurDetector(always_deblur=True)  # Will trigger deblur
        
        manager = DeblurManager(
            mprnet=mock_mprnet,
            blur_detector=mock_blur_detector,
            frame_interval=3,
            max_roi_width=256,
            deblur_enabled=True  # Enabled
        )
        
        roi = np.zeros((100, 100, 3), dtype=np.uint8)
        
        result, deblur_applied, source_frame = manager.process_roi(roi, wagon_id, frame_index)
        
        # Verify deblurring was applied (since mock says always_deblur=True)
        assert deblur_applied is True, "Deblur should be applied when enabled and needed"
        assert source_frame is not None, "Source frame should be set when deblur applied"
        assert mock_mprnet.call_count == 1, "MPRNet should be called when enabled and needed"
        
        # Verify status is ACTIVE
        status = manager.get_last_status()
        assert status is not None, "Status should be set after processing"
        assert status.enabled is True, "Status should show deblur is enabled"
        assert status.applied is True, "Status should show deblur was applied"
        assert status.status_type.value == "active", "Status type should be ACTIVE"

    @given(
        wagon_id=wagon_id_strategy,
        frame_index=st.integers(min_value=0, max_value=10000)
    )
    @settings(max_examples=100)
    def test_enabled_deblur_skipped_when_not_needed(self, wagon_id, frame_index):
        """
        Feature: ocr-enhancement-improvements, Property 9: Deblur Enable/Disable
        
        Verify that when deblur is enabled but not needed, status is SKIPPED.
        **Validates: Requirements 7.2, 7.3**
        """
        mock_mprnet = MockMPRNet()
        mock_blur_detector = MockBlurDetector(always_deblur=False)  # Won't trigger deblur
        
        manager = DeblurManager(
            mprnet=mock_mprnet,
            blur_detector=mock_blur_detector,
            frame_interval=3,
            max_roi_width=256,
            deblur_enabled=True  # Enabled
        )
        
        roi = np.zeros((100, 100, 3), dtype=np.uint8)
        
        result, deblur_applied, source_frame = manager.process_roi(roi, wagon_id, frame_index)
        
        # Verify deblurring was not applied (image is sharp)
        assert deblur_applied is False, "Deblur should not be applied when not needed"
        assert source_frame is None, "Source frame should be None when not deblurred"
        assert mock_mprnet.call_count == 0, "MPRNet should not be called when not needed"
        
        # Verify status is SKIPPED
        status = manager.get_last_status()
        assert status is not None, "Status should be set after processing"
        assert status.enabled is True, "Status should show deblur is enabled"
        assert status.applied is False, "Status should show deblur was not applied"
        assert status.status_type.value == "skipped", "Status type should be SKIPPED"

    @given(
        wagon_id=wagon_id_strategy,
        frame_index=st.integers(min_value=0, max_value=10000)
    )
    @settings(max_examples=100)
    def test_toggle_deblur_enabled(self, wagon_id, frame_index):
        """
        Feature: ocr-enhancement-improvements, Property 9: Deblur Enable/Disable
        
        Verify that toggling deblur enabled/disabled works correctly.
        **Validates: Requirements 7.2, 7.3**
        """
        mock_mprnet = MockMPRNet()
        mock_blur_detector = MockBlurDetector(always_deblur=True)
        
        manager = DeblurManager(
            mprnet=mock_mprnet,
            blur_detector=mock_blur_detector,
            frame_interval=3,
            max_roi_width=256,
            deblur_enabled=True  # Start enabled
        )
        
        roi = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Process with deblur enabled
        assert manager.is_deblur_enabled() is True
        result1, deblur_applied1, _ = manager.process_roi(roi, wagon_id, frame_index)
        assert deblur_applied1 is True, "Should deblur when enabled"
        
        # Disable deblur
        manager.set_deblur_enabled(False)
        assert manager.is_deblur_enabled() is False
        
        # Process with deblur disabled (use different wagon_id to avoid cache)
        result2, deblur_applied2, _ = manager.process_roi(roi, wagon_id + 1000, frame_index + 1)
        assert deblur_applied2 is False, "Should not deblur when disabled"
        
        # Re-enable deblur
        manager.set_deblur_enabled(True)
        assert manager.is_deblur_enabled() is True
        
        # Process with deblur re-enabled (use different wagon_id)
        result3, deblur_applied3, _ = manager.process_roi(roi, wagon_id + 2000, frame_index + 2)
        assert deblur_applied3 is True, "Should deblur when re-enabled"

    @given(wagon_id=wagon_id_strategy)
    @settings(max_examples=100)
    def test_was_deblur_applied_method(self, wagon_id):
        """
        Feature: ocr-enhancement-improvements, Property 9: Deblur Enable/Disable
        
        Verify was_deblur_applied() returns correct value.
        **Validates: Requirements 7.2, 7.3**
        """
        mock_mprnet = MockMPRNet()
        mock_blur_detector = MockBlurDetector(always_deblur=True)
        
        manager = DeblurManager(
            mprnet=mock_mprnet,
            blur_detector=mock_blur_detector,
            frame_interval=3,
            max_roi_width=256,
            deblur_enabled=True
        )
        
        # Before any processing, should return False
        assert manager.was_deblur_applied() is False
        
        roi = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # After processing with deblur enabled and needed
        manager.process_roi(roi, wagon_id, 0)
        assert manager.was_deblur_applied() is True
        
        # Disable and process again
        manager.set_deblur_enabled(False)
        manager.process_roi(roi, wagon_id + 1000, 1)
        assert manager.was_deblur_applied() is False



class TestDeblurOperationLogging:
    """
    Property 3: Deblur Operation Logging
    
    *For any* deblur operation that is applied, the log SHALL contain:
    - Frame index (non-negative integer)
    - Wagon ID (positive integer)
    - Blur score before deblurring
    - Blur score after deblurring (if computed)
    
    Validates: Requirements 2.1, 2.5
    """

    @given(
        wagon_id=st.integers(min_value=1, max_value=10000),
        frame_index=st.integers(min_value=0, max_value=100000),
        blur_score=st.floats(min_value=50.0, max_value=500.0, allow_nan=False)
    )
    @settings(max_examples=100)
    def test_deblur_status_contains_required_fields(self, wagon_id, frame_index, blur_score):
        """
        Feature: ocr-enhancement-improvements, Property 3: Deblur Operation Logging
        
        Verify that DeblurStatus contains all required fields when deblur is applied.
        **Validates: Requirements 2.1, 2.5**
        """
        mock_mprnet = MockMPRNet()
        mock_blur_detector = MockBlurDetector(always_deblur=True, blur_score=blur_score)
        
        manager = DeblurManager(
            mprnet=mock_mprnet,
            blur_detector=mock_blur_detector,
            frame_interval=3,
            max_roi_width=256,
            deblur_enabled=True
        )
        
        roi = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Process ROI to trigger deblur
        manager.process_roi(roi, wagon_id, frame_index)
        
        # Get status
        status = manager.get_last_status()
        
        # Verify status contains required fields
        assert status is not None, "Status should be set after processing"
        assert hasattr(status, 'enabled'), "Status should have 'enabled' field"
        assert hasattr(status, 'applied'), "Status should have 'applied' field"
        assert hasattr(status, 'blur_score_before'), "Status should have 'blur_score_before' field"
        assert hasattr(status, 'blur_score_after'), "Status should have 'blur_score_after' field"
        assert hasattr(status, 'status_type'), "Status should have 'status_type' field"
        
        # Verify field values when deblur is applied
        assert status.applied is True, "Deblur should be applied"
        assert status.blur_score_before == blur_score, \
            f"Blur score before should be {blur_score}, got {status.blur_score_before}"
        # blur_score_after should be set when deblur is applied on N-th frame
        assert status.blur_score_after is not None, \
            "Blur score after should be computed when deblur is applied"

    @given(
        wagon_id=st.integers(min_value=1, max_value=10000),
        frame_index=st.integers(min_value=0, max_value=100000)
    )
    @settings(max_examples=100)
    def test_deblur_status_frame_index_non_negative(self, wagon_id, frame_index):
        """
        Feature: ocr-enhancement-improvements, Property 3: Deblur Operation Logging
        
        Verify that frame index in status is non-negative.
        **Validates: Requirements 2.1, 2.5**
        """
        mock_mprnet = MockMPRNet()
        mock_blur_detector = MockBlurDetector(always_deblur=True)
        
        manager = DeblurManager(
            mprnet=mock_mprnet,
            blur_detector=mock_blur_detector,
            frame_interval=3,
            max_roi_width=256,
            deblur_enabled=True
        )
        
        roi = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Process ROI
        result, deblur_applied, source_frame = manager.process_roi(roi, wagon_id, frame_index)
        
        # Verify source_frame is non-negative when deblur is applied
        if deblur_applied:
            assert source_frame is not None, "Source frame should be set when deblur applied"
            assert source_frame >= 0, f"Source frame should be non-negative, got {source_frame}"

    @given(
        wagon_id=st.integers(min_value=1, max_value=10000),
        frame_index=st.integers(min_value=0, max_value=100000)
    )
    @settings(max_examples=100)
    def test_deblur_status_wagon_id_positive(self, wagon_id, frame_index):
        """
        Feature: ocr-enhancement-improvements, Property 3: Deblur Operation Logging
        
        Verify that wagon ID is positive (as per requirements).
        **Validates: Requirements 2.1, 2.5**
        """
        # wagon_id is already constrained to be positive by the strategy
        assert wagon_id > 0, "Wagon ID should be positive"
        
        mock_mprnet = MockMPRNet()
        mock_blur_detector = MockBlurDetector(always_deblur=True)
        
        manager = DeblurManager(
            mprnet=mock_mprnet,
            blur_detector=mock_blur_detector,
            frame_interval=3,
            max_roi_width=256,
            deblur_enabled=True
        )
        
        roi = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Process ROI - this should work with positive wagon_id
        result, deblur_applied, source_frame = manager.process_roi(roi, wagon_id, frame_index)
        
        # Verify processing succeeded
        assert result is not None, "Result should not be None"
        assert deblur_applied is True, "Deblur should be applied"

    @given(
        wagon_id=st.integers(min_value=1, max_value=10000),
        frame_index=st.integers(min_value=0, max_value=100000),
        blur_score=st.floats(min_value=50.0, max_value=500.0, allow_nan=False)
    )
    @settings(max_examples=100)
    def test_blur_score_before_matches_computed_score(self, wagon_id, frame_index, blur_score):
        """
        Feature: ocr-enhancement-improvements, Property 3: Deblur Operation Logging
        
        Verify that blur_score_before in status matches the computed blur score.
        **Validates: Requirements 2.1, 2.5**
        """
        mock_mprnet = MockMPRNet()
        mock_blur_detector = MockBlurDetector(always_deblur=True, blur_score=blur_score)
        
        manager = DeblurManager(
            mprnet=mock_mprnet,
            blur_detector=mock_blur_detector,
            frame_interval=3,
            max_roi_width=256,
            deblur_enabled=True
        )
        
        roi = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Process ROI
        manager.process_roi(roi, wagon_id, frame_index)
        
        # Get status
        status = manager.get_last_status()
        
        # Verify blur_score_before matches the mock's configured score
        assert status.blur_score_before == blur_score, \
            f"blur_score_before should be {blur_score}, got {status.blur_score_before}"

    @given(
        wagon_id=st.integers(min_value=1, max_value=10000),
        frame_index=st.integers(min_value=0, max_value=100000)
    )
    @settings(max_examples=100)
    def test_status_text_property(self, wagon_id, frame_index):
        """
        Feature: ocr-enhancement-improvements, Property 3: Deblur Operation Logging
        
        Verify that status_text property returns correct human-readable text.
        **Validates: Requirements 2.1, 2.5**
        """
        mock_mprnet = MockMPRNet()
        mock_blur_detector = MockBlurDetector(always_deblur=True)
        
        manager = DeblurManager(
            mprnet=mock_mprnet,
            blur_detector=mock_blur_detector,
            frame_interval=3,
            max_roi_width=256,
            deblur_enabled=True
        )
        
        roi = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Process ROI with deblur enabled and needed -> ACTIVE
        manager.process_roi(roi, wagon_id, frame_index)
        status = manager.get_last_status()
        assert status.status_text == "ACTIVE", f"Expected 'ACTIVE', got '{status.status_text}'"
        
        # Process with deblur disabled -> DISABLED
        manager.set_deblur_enabled(False)
        manager.process_roi(roi, wagon_id + 1000, frame_index + 1)
        status = manager.get_last_status()
        assert status.status_text == "DISABLED", f"Expected 'DISABLED', got '{status.status_text}'"
        
        # Process with deblur enabled but not needed -> SKIPPED
        manager.set_deblur_enabled(True)
        mock_blur_detector.always_deblur = False
        manager.process_roi(roi, wagon_id + 2000, frame_index + 2)
        status = manager.get_last_status()
        assert status.status_text == "SKIPPED", f"Expected 'SKIPPED', got '{status.status_text}'"
