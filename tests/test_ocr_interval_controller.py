"""
Property-based tests for OCR interval controller module.

Feature: ocr-visual-enhancements
Property 6: OCR Frame Interval Execution
Property 7: OCR Status Reflects Execution State
Validates: Requirements 4.4, 4.6, 5.2, 5.3
"""

import pytest
from hypothesis import given, strategies as st, settings

from dashboard.ocr_interval_controller import OCRIntervalController


class TestOCRFrameIntervalExecution:
    """
    Property 6: OCR Frame Interval Execution
    
    For any frame index F and OCR interval N, OCR SHALL execute
    if and only if F % N == 0.
    
    Validates: Requirements 4.4, 4.6
    """

    @given(
        frame_index=st.integers(min_value=0, max_value=10000),
        interval=st.integers(min_value=1, max_value=30)
    )
    @settings(max_examples=100)
    def test_ocr_executes_on_interval_frames(self, frame_index, interval):
        """
        Feature: ocr-visual-enhancements, Property 6: OCR Frame Interval Execution
        
        Verify that OCR executes if and only if frame_index % interval == 0.
        """
        controller = OCRIntervalController(interval=interval)
        
        should_run = controller.should_run_ocr(frame_index)
        expected = (frame_index % interval == 0)
        
        assert should_run == expected, \
            f"For frame {frame_index} with interval {interval}: " \
            f"expected should_run_ocr={expected}, got {should_run}"

    @given(interval=st.integers(min_value=1, max_value=30))
    @settings(max_examples=100)
    def test_ocr_always_runs_on_frame_zero(self, interval):
        """
        Feature: ocr-visual-enhancements, Property 6: OCR Frame Interval Execution
        
        Verify that OCR always runs on frame 0 regardless of interval.
        """
        controller = OCRIntervalController(interval=interval)
        
        assert controller.should_run_ocr(0) is True, \
            f"OCR should always run on frame 0, but got False for interval {interval}"

    @given(interval=st.integers(min_value=1, max_value=30))
    @settings(max_examples=100)
    def test_ocr_runs_on_exact_multiples(self, interval):
        """
        Feature: ocr-visual-enhancements, Property 6: OCR Frame Interval Execution
        
        Verify that OCR runs on exact multiples of the interval.
        """
        controller = OCRIntervalController(interval=interval)
        
        # Test several multiples
        for multiplier in range(1, 6):
            frame_index = interval * multiplier
            assert controller.should_run_ocr(frame_index) is True, \
                f"OCR should run on frame {frame_index} (multiple of {interval})"

    @given(interval=st.integers(min_value=2, max_value=30))
    @settings(max_examples=100)
    def test_ocr_skips_non_multiples(self, interval):
        """
        Feature: ocr-visual-enhancements, Property 6: OCR Frame Interval Execution
        
        Verify that OCR is skipped on non-multiples of the interval.
        """
        controller = OCRIntervalController(interval=interval)
        
        # Test frames that are not multiples (1 to interval-1)
        for offset in range(1, interval):
            frame_index = offset
            assert controller.should_run_ocr(frame_index) is False, \
                f"OCR should be skipped on frame {frame_index} (not a multiple of {interval})"

    @given(
        frame_index=st.integers(min_value=0, max_value=10000),
        interval=st.integers(min_value=1, max_value=30)
    )
    @settings(max_examples=100)
    def test_interval_execution_is_deterministic(self, frame_index, interval):
        """
        Feature: ocr-visual-enhancements, Property 6: OCR Frame Interval Execution
        
        Verify that the same frame index and interval always produce the same result.
        """
        controller = OCRIntervalController(interval=interval)
        
        # Call multiple times and verify consistency
        results = [controller.should_run_ocr(frame_index) for _ in range(3)]
        
        assert all(r == results[0] for r in results), \
            f"should_run_ocr should be deterministic for frame {frame_index}, interval {interval}"


class TestOCRStatusReflectsExecutionState:
    """
    Property 7: OCR Status Reflects Execution State
    
    For any frame where OCR executes, status SHALL be "ACTIVE".
    For any frame where OCR is skipped due to interval, status SHALL contain "SKIPPED".
    
    Validates: Requirements 5.2, 5.3
    """

    @given(
        frame_index=st.integers(min_value=0, max_value=10000),
        interval=st.integers(min_value=1, max_value=30)
    )
    @settings(max_examples=100)
    def test_status_active_when_ocr_runs(self, frame_index, interval):
        """
        Feature: ocr-visual-enhancements, Property 7: OCR Status Reflects Execution State
        
        Verify that status is "ACTIVE" when OCR should run.
        """
        controller = OCRIntervalController(interval=interval)
        
        if controller.should_run_ocr(frame_index):
            status = controller.get_status_text(frame_index)
            assert status == "ACTIVE", \
                f"Status should be 'ACTIVE' when OCR runs on frame {frame_index}, got '{status}'"

    @given(
        frame_index=st.integers(min_value=0, max_value=10000),
        interval=st.integers(min_value=2, max_value=30)
    )
    @settings(max_examples=100)
    def test_status_skipped_when_ocr_not_running(self, frame_index, interval):
        """
        Feature: ocr-visual-enhancements, Property 7: OCR Status Reflects Execution State
        
        Verify that status contains "SKIPPED" when OCR is skipped.
        """
        controller = OCRIntervalController(interval=interval)
        
        if not controller.should_run_ocr(frame_index):
            status = controller.get_status_text(frame_index)
            assert "SKIPPED" in status, \
                f"Status should contain 'SKIPPED' when OCR is skipped on frame {frame_index}, got '{status}'"

    @given(
        frame_index=st.integers(min_value=0, max_value=10000),
        interval=st.integers(min_value=2, max_value=30)
    )
    @settings(max_examples=100)
    def test_status_shows_frame_position_when_skipped(self, frame_index, interval):
        """
        Feature: ocr-visual-enhancements, Property 7: OCR Status Reflects Execution State
        
        Verify that skipped status shows correct frame position within interval.
        """
        controller = OCRIntervalController(interval=interval)
        
        if not controller.should_run_ocr(frame_index):
            status = controller.get_status_text(frame_index)
            
            # Calculate expected position
            expected_position = (frame_index % interval) + 1
            expected_format = f"SKIPPED (frame {expected_position} of {interval})"
            
            assert status == expected_format, \
                f"Expected '{expected_format}', got '{status}'"

    @given(interval=st.integers(min_value=1, max_value=30))
    @settings(max_examples=100)
    def test_status_consistency_with_should_run(self, interval):
        """
        Feature: ocr-visual-enhancements, Property 7: OCR Status Reflects Execution State
        
        Verify that status text is consistent with should_run_ocr result.
        """
        controller = OCRIntervalController(interval=interval)
        
        # Test a range of frames
        for frame_index in range(interval * 3):
            should_run = controller.should_run_ocr(frame_index)
            status = controller.get_status_text(frame_index)
            
            if should_run:
                assert status == "ACTIVE", \
                    f"Status should be 'ACTIVE' when should_run_ocr is True (frame {frame_index})"
            else:
                assert "SKIPPED" in status, \
                    f"Status should contain 'SKIPPED' when should_run_ocr is False (frame {frame_index})"


class TestOCRIntervalControllerDefaults:
    """Unit tests for OCRIntervalController default values and constraints."""

    def test_default_interval_is_five(self):
        """Verify default interval is 5 per requirement 4.3."""
        controller = OCRIntervalController()
        assert controller.interval == 5, \
            f"Default interval should be 5, got {controller.interval}"

    def test_min_interval_is_one(self):
        """Verify minimum interval is 1 per requirement 4.2."""
        controller = OCRIntervalController(interval=0)
        assert controller.interval == 1, \
            f"Minimum interval should be 1, got {controller.interval}"
        
        controller = OCRIntervalController(interval=-5)
        assert controller.interval == 1, \
            f"Negative interval should be clamped to 1, got {controller.interval}"

    def test_max_interval_is_thirty(self):
        """Verify maximum interval is 30 per requirement 4.2."""
        controller = OCRIntervalController(interval=50)
        assert controller.interval == 30, \
            f"Maximum interval should be 30, got {controller.interval}"

    def test_interval_setter_clamps_values(self):
        """Verify interval setter clamps values to valid range."""
        controller = OCRIntervalController()
        
        controller.interval = 0
        assert controller.interval == 1
        
        controller.interval = 100
        assert controller.interval == 30
        
        controller.interval = 15
        assert controller.interval == 15

    def test_class_constants(self):
        """Verify class constants are correctly defined."""
        assert OCRIntervalController.MIN_INTERVAL == 1
        assert OCRIntervalController.MAX_INTERVAL == 30
        assert OCRIntervalController.DEFAULT_INTERVAL == 5
