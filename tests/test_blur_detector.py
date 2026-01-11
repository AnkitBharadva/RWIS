"""
Property-based tests for blur detection module.

Feature: railway-wagon-inspection
Validates: Requirements 1.2, 1.3, 1.4, 1.5

Feature: ocr-enhancement-improvements
Validates: Requirements 3.5, 3.6, 4.7, 6.4

IMPORTANT: Laplacian variance interpretation:
- HIGH variance = SHARP image (many edges detected)
- LOW variance = BLURRY image (few edges detected)

So the threshold logic is:
- blur_score >= T2: SKIP_DEBLUR (image is sharp enough)
- T1 <= blur_score < T2: ROI_DEBLUR (moderate blur, deblur can help)
- blur_score < T1: NO_DEBLUR (too blurry, deblur won't help)
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings

from pipelines.blur_detector import BlurDetector
from pipelines.calibration_manager import CalibrationResult
from utils.data_models import BlurDecision, BlurSettings


# Strategy for generating valid blur thresholds where t1 < t2
@st.composite
def valid_blur_thresholds(draw):
    """Generate valid T1, T2 thresholds where T1 < T2."""
    t1 = draw(st.floats(min_value=0.0, max_value=500.0, allow_nan=False, allow_infinity=False))
    t2 = draw(st.floats(min_value=t1 + 0.1, max_value=1000.0, allow_nan=False, allow_infinity=False))
    return t1, t2


# Strategy for generating blur scores
blur_score_strategy = st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False)


class TestBlurDecisionLogic:
    """
    Property 1: Blur Decision Logic Consistency
    
    For any frame with a computed blur_score and configured thresholds T1 and T2 (where T1 < T2):
    - If blur_score >= T2, the blur decision SHALL be SKIP_DEBLUR (sharp image)
    - If T1 <= blur_score < T2, the blur decision SHALL be ROI_DEBLUR (moderate blur)
    - If blur_score < T1, the blur decision SHALL be NO_DEBLUR (too blurry)
    
    Remember: HIGH score = SHARP, LOW score = BLURRY
    
    Validates: Requirements 1.3, 1.4, 1.5
    """

    @given(thresholds=valid_blur_thresholds(), blur_score=blur_score_strategy)
    @settings(max_examples=100)
    def test_blur_decision_logic_consistency(self, thresholds, blur_score):
        """
        Feature: railway-wagon-inspection, Property 1: Blur Decision Logic Consistency
        
        Generate random blur_score values and T1/T2 thresholds.
        Verify decision matches expected logic for all combinations.
        
        Logic (HIGH score = SHARP, LOW score = BLURRY):
        - blur_score >= T2: SKIP_DEBLUR (sharp enough)
        - T1 <= blur_score < T2: ROI_DEBLUR (moderate blur, can help)
        - blur_score < T1: NO_DEBLUR (too blurry to recover)
        """
        t1, t2 = thresholds
        detector = BlurDetector(t1=t1, t2=t2)
        
        decision = detector.get_blur_decision(blur_score)
        
        # Verify decision matches expected logic
        if blur_score >= t2:
            # Sharp image - skip deblurring
            assert decision == BlurDecision.SKIP_DEBLUR, \
                f"Expected SKIP_DEBLUR for blur_score={blur_score} >= T2={t2} (sharp image)"
        elif blur_score >= t1:
            # Moderate blur - deblurring can help
            assert decision == BlurDecision.ROI_DEBLUR, \
                f"Expected ROI_DEBLUR for T1={t1} <= blur_score={blur_score} < T2={t2} (moderate blur)"
        else:
            # Too blurry - deblurring won't help
            assert decision == BlurDecision.NO_DEBLUR, \
                f"Expected NO_DEBLUR for blur_score={blur_score} < T1={t1} (too blurry)"

    @given(thresholds=valid_blur_thresholds())
    @settings(max_examples=100)
    def test_boundary_at_t1(self, thresholds):
        """
        Feature: railway-wagon-inspection, Property 1: Blur Decision Logic Consistency
        
        Test boundary condition: blur_score exactly at T1 should be ROI_DEBLUR.
        (T1 is the lower bound where deblurring can still help)
        """
        t1, t2 = thresholds
        detector = BlurDetector(t1=t1, t2=t2)
        
        # At exactly T1, should be ROI_DEBLUR (T1 <= blur_score < T2)
        decision = detector.get_blur_decision(t1)
        assert decision == BlurDecision.ROI_DEBLUR, \
            f"Expected ROI_DEBLUR at boundary T1={t1}"

    @given(thresholds=valid_blur_thresholds())
    @settings(max_examples=100)
    def test_boundary_at_t2(self, thresholds):
        """
        Feature: railway-wagon-inspection, Property 1: Blur Decision Logic Consistency
        
        Test boundary condition: blur_score exactly at T2 should be SKIP_DEBLUR.
        (T2 is the threshold above which image is sharp enough)
        """
        t1, t2 = thresholds
        detector = BlurDetector(t1=t1, t2=t2)
        
        # At exactly T2, should be SKIP_DEBLUR (blur_score >= T2 means sharp)
        decision = detector.get_blur_decision(t2)
        assert decision == BlurDecision.SKIP_DEBLUR, \
            f"Expected SKIP_DEBLUR at boundary T2={t2} (sharp enough)"


# Strategy for generating random frames
@st.composite
def random_frame(draw):
    """Generate a random BGR frame with valid dimensions."""
    height = draw(st.integers(min_value=10, max_value=200))
    width = draw(st.integers(min_value=10, max_value=200))
    # Generate random pixel values
    frame = draw(st.lists(
        st.lists(
            st.lists(st.integers(min_value=0, max_value=255), min_size=3, max_size=3),
            min_size=width, max_size=width
        ),
        min_size=height, max_size=height
    ))
    return np.array(frame, dtype=np.uint8)


# Simpler strategy using numpy directly for better performance
@st.composite
def random_frame_fast(draw):
    """Generate a random BGR frame using numpy for better performance."""
    height = draw(st.integers(min_value=10, max_value=100))
    width = draw(st.integers(min_value=10, max_value=100))
    # Use a seed to generate deterministic random data
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.default_rng(seed)
    frame = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    return frame


class TestBlurScoreDeterminism:
    """
    Property 2: Blur Score Computation Determinism
    
    For any input frame, computing the blur score twice SHALL produce identical results.
    The Laplacian variance computation must be deterministic.
    
    Validates: Requirements 1.2
    """

    @given(frame=random_frame_fast(), thresholds=valid_blur_thresholds())
    @settings(max_examples=100)
    def test_blur_score_determinism(self, frame, thresholds):
        """
        Feature: railway-wagon-inspection, Property 2: Blur Score Computation Determinism
        
        Generate random frames and verify computing blur_score twice produces identical results.
        """
        t1, t2 = thresholds
        detector = BlurDetector(t1=t1, t2=t2)
        
        # Compute blur score twice
        score1 = detector.compute_blur_score(frame)
        score2 = detector.compute_blur_score(frame)
        
        # Verify identical results
        assert score1 == score2, \
            f"Blur score not deterministic: {score1} != {score2}"

    @given(frame=random_frame_fast(), thresholds=valid_blur_thresholds())
    @settings(max_examples=100)
    def test_blur_score_with_copy(self, frame, thresholds):
        """
        Feature: railway-wagon-inspection, Property 2: Blur Score Computation Determinism
        
        Verify that computing blur score on a copy of the frame produces the same result.
        """
        t1, t2 = thresholds
        detector = BlurDetector(t1=t1, t2=t2)
        
        # Compute blur score on original and copy
        score_original = detector.compute_blur_score(frame)
        score_copy = detector.compute_blur_score(frame.copy())
        
        # Verify identical results
        assert score_original == score_copy, \
            f"Blur score differs between original and copy: {score_original} != {score_copy}"

    @given(thresholds=valid_blur_thresholds())
    @settings(max_examples=100)
    def test_blur_score_grayscale_input(self, thresholds):
        """
        Feature: railway-wagon-inspection, Property 2: Blur Score Computation Determinism
        
        Verify blur score computation works with grayscale input and is deterministic.
        """
        t1, t2 = thresholds
        detector = BlurDetector(t1=t1, t2=t2)
        
        # Create a grayscale frame
        gray_frame = np.random.randint(0, 256, size=(50, 50), dtype=np.uint8)
        
        # Compute blur score twice
        score1 = detector.compute_blur_score(gray_frame)
        score2 = detector.compute_blur_score(gray_frame)
        
        # Verify identical results
        assert score1 == score2, \
            f"Blur score not deterministic for grayscale: {score1} != {score2}"



# Strategy for generating valid calibration results
@st.composite
def calibration_result_strategy(draw):
    """Generate a valid CalibrationResult for testing."""
    blur_threshold = draw(st.floats(min_value=10.0, max_value=500.0, allow_nan=False, allow_infinity=False))
    low_light_threshold = draw(st.integers(min_value=20, max_value=200))
    gamma_value = draw(st.floats(min_value=0.5, max_value=2.5, allow_nan=False, allow_infinity=False))
    sample_count = draw(st.integers(min_value=1, max_value=100))
    
    return CalibrationResult(
        blur_threshold=blur_threshold,
        low_light_threshold=low_light_threshold,
        gamma_value=gamma_value,
        sample_count=sample_count,
        blur_scores=[blur_threshold],
        luminance_values=[float(low_light_threshold)]
    )


class TestManualModeOverride:
    """
    Property 6: Manual Mode Override
    
    For any user-specified threshold value when manual mode is selected:
    - The pipeline SHALL use the exact user-specified value
    - Auto-calibrated values SHALL NOT override manual settings
    - Switching to manual mode SHALL preserve the current threshold value
    
    Feature: ocr-enhancement-improvements, Property 6: Manual Mode Override
    Validates: Requirements 3.6, 4.7
    """

    @given(
        thresholds=valid_blur_thresholds(),
        manual_threshold=st.floats(min_value=50.0, max_value=800.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_manual_threshold_used_exactly(self, thresholds, manual_threshold):
        """
        Feature: ocr-enhancement-improvements, Property 6: Manual Mode Override
        
        When manual mode is selected, the pipeline SHALL use the exact
        user-specified threshold value.
        """
        t1, t2 = thresholds
        
        # Ensure manual_threshold is valid (greater than t1)
        if manual_threshold <= t1:
            manual_threshold = t1 + 10.0
        
        detector = BlurDetector(t1=t1, t2=t2, auto_mode=False)
        
        # Set manual threshold
        detector.set_threshold(manual_threshold)
        
        # Verify exact value is used
        assert detector.get_threshold() == manual_threshold, \
            f"Expected threshold {manual_threshold}, got {detector.get_threshold()}"

    @given(
        thresholds=valid_blur_thresholds(),
        calibration=calibration_result_strategy()
    )
    @settings(max_examples=100)
    def test_calibration_does_not_override_manual_mode(self, thresholds, calibration):
        """
        Feature: ocr-enhancement-improvements, Property 6: Manual Mode Override
        
        Auto-calibrated values SHALL NOT override manual settings.
        When manual mode is active, update_from_calibration should be ignored.
        """
        t1, t2 = thresholds
        detector = BlurDetector(t1=t1, t2=t2, auto_mode=False)
        
        # Store original threshold
        original_threshold = detector.get_threshold()
        
        # Attempt to update from calibration
        detector.update_from_calibration(calibration)
        
        # Verify threshold unchanged
        assert detector.get_threshold() == original_threshold, \
            f"Manual mode threshold was overridden: {original_threshold} -> {detector.get_threshold()}"

    @given(thresholds=valid_blur_thresholds())
    @settings(max_examples=100)
    def test_switching_to_manual_preserves_threshold(self, thresholds):
        """
        Feature: ocr-enhancement-improvements, Property 6: Manual Mode Override
        
        Switching to manual mode SHALL preserve the current threshold value.
        """
        t1, t2 = thresholds
        detector = BlurDetector(t1=t1, t2=t2, auto_mode=True)
        
        # Store threshold before switching
        threshold_before = detector.get_threshold()
        
        # Switch to manual mode
        detector.set_auto_mode(False)
        
        # Verify threshold preserved
        assert detector.get_threshold() == threshold_before, \
            f"Threshold changed when switching to manual: {threshold_before} -> {detector.get_threshold()}"
        
        # Verify mode changed
        assert not detector.is_auto_mode(), "Auto mode should be disabled"

    @given(
        thresholds=valid_blur_thresholds(),
        calibration=calibration_result_strategy()
    )
    @settings(max_examples=100)
    def test_auto_mode_accepts_calibration(self, thresholds, calibration):
        """
        Feature: ocr-enhancement-improvements, Property 6: Manual Mode Override
        
        Verify that auto mode DOES accept calibration updates (contrast to manual mode).
        """
        t1, t2 = thresholds
        detector = BlurDetector(t1=t1, t2=t2, auto_mode=True)
        
        # Update from calibration
        detector.update_from_calibration(calibration)
        
        # Verify threshold was updated to calibration value
        assert detector.get_threshold() == calibration.blur_threshold, \
            f"Auto mode should accept calibration: expected {calibration.blur_threshold}, got {detector.get_threshold()}"

    @given(thresholds=valid_blur_thresholds())
    @settings(max_examples=100)
    def test_settings_roundtrip(self, thresholds):
        """
        Feature: ocr-enhancement-improvements, Property 6: Manual Mode Override
        
        Verify that get_settings and set_settings preserve values correctly.
        """
        t1, t2 = thresholds
        detector = BlurDetector(t1=t1, t2=t2, auto_mode=False)
        
        # Get current settings
        settings = detector.get_settings()
        
        # Verify settings match detector state
        assert settings.threshold == detector.get_threshold()
        assert settings.auto_mode == detector.is_auto_mode()
        
        # Create new settings
        new_threshold = t1 + 50.0  # Ensure valid threshold
        new_settings = BlurSettings(
            threshold=new_threshold,
            auto_mode=True,
            deblur_enabled=True
        )
        
        # Apply new settings
        detector.set_settings(new_settings)
        
        # Verify settings applied
        assert detector.get_threshold() == new_threshold
        assert detector.is_auto_mode() == True
