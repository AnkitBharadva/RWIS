"""
Property-based tests for CalibrationManager module.

Feature: ocr-enhancement-improvements
Validates: Requirements 3.2, 3.3, 3.4, 4.2, 4.3, 4.4, 4.5
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, assume

from pipelines.calibration_manager import CalibrationManager, CalibrationResult


# Strategy for generating valid percentile values
percentile_strategy = st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)

# Strategy for generating valid sample sizes
sample_size_strategy = st.integers(min_value=1, max_value=100)

# Strategy for generating blur scores (typical range for Laplacian variance)
blur_score_strategy = st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False)

# Strategy for generating luminance values (0-255 range)
luminance_strategy = st.floats(min_value=0.0, max_value=255.0, allow_nan=False, allow_infinity=False)


@st.composite
def random_frame_fast(draw):
    """Generate a random BGR frame using numpy for better performance."""
    height = draw(st.integers(min_value=10, max_value=100))
    width = draw(st.integers(min_value=10, max_value=100))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.default_rng(seed)
    frame = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    return frame


@st.composite
def blur_scores_list(draw, min_size=1, max_size=100):
    """Generate a list of blur scores."""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    scores = draw(st.lists(blur_score_strategy, min_size=size, max_size=size))
    return scores


@st.composite
def luminance_values_list(draw, min_size=1, max_size=100):
    """Generate a list of luminance values."""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    values = draw(st.lists(luminance_strategy, min_size=size, max_size=size))
    return values


class TestBlurAutoCalibration:
    """
    Property 4: Blur Auto-Calibration
    
    For any set of sample frames provided to the calibration manager:
    - The computed blur_threshold SHALL be at the configured percentile of blur scores
    - The threshold SHALL be within the range of observed blur scores
    - Calibration SHALL complete when sample_count reaches configured sample_size
    
    Validates: Requirements 3.2, 3.3, 3.4
    """

    @given(
        blur_scores=blur_scores_list(min_size=5, max_size=50),
        blur_percentile=percentile_strategy
    )
    @settings(max_examples=100)
    def test_blur_threshold_at_configured_percentile(self, blur_scores, blur_percentile):
        """
        Feature: ocr-enhancement-improvements, Property 4: Blur Auto-Calibration
        
        Verify that the computed blur_threshold is at the configured percentile
        of the blur scores.
        """
        # Skip if all scores are identical (percentile is trivial)
        assume(len(set(blur_scores)) > 1)
        
        sample_size = len(blur_scores)
        manager = CalibrationManager(
            sample_size=sample_size,
            blur_percentile=blur_percentile
        )
        
        # Simulate adding samples by directly setting internal state
        # This tests the percentile computation logic directly
        manager._blur_scores = blur_scores.copy()
        manager._luminance_values = [128.0] * len(blur_scores)  # Dummy luminance
        manager._calibration_complete = True
        
        result = manager.compute_calibration()
        
        assert result is not None, "Calibration should return a result"
        
        # Verify threshold is at the configured percentile
        expected_threshold = np.percentile(blur_scores, blur_percentile)
        assert abs(result.blur_threshold - expected_threshold) < 1e-6, \
            f"Expected blur_threshold={expected_threshold}, got {result.blur_threshold}"

    @given(blur_scores=blur_scores_list(min_size=5, max_size=50))
    @settings(max_examples=100)
    def test_blur_threshold_within_observed_range(self, blur_scores):
        """
        Feature: ocr-enhancement-improvements, Property 4: Blur Auto-Calibration
        
        Verify that the computed blur_threshold is within the range of
        observed blur scores.
        """
        sample_size = len(blur_scores)
        manager = CalibrationManager(sample_size=sample_size)
        
        # Set internal state directly
        manager._blur_scores = blur_scores.copy()
        manager._luminance_values = [128.0] * len(blur_scores)
        manager._calibration_complete = True
        
        result = manager.compute_calibration()
        
        assert result is not None, "Calibration should return a result"
        
        min_score = min(blur_scores)
        max_score = max(blur_scores)
        
        assert min_score <= result.blur_threshold <= max_score, \
            f"blur_threshold={result.blur_threshold} not in range [{min_score}, {max_score}]"

    @given(
        sample_size=st.integers(min_value=1, max_value=50),
        frame=random_frame_fast()
    )
    @settings(max_examples=100)
    def test_calibration_completes_at_sample_size(self, sample_size, frame):
        """
        Feature: ocr-enhancement-improvements, Property 4: Blur Auto-Calibration
        
        Verify that calibration completes when sample_count reaches
        the configured sample_size.
        """
        manager = CalibrationManager(sample_size=sample_size)
        
        # Add samples until calibration completes
        for i in range(sample_size):
            is_complete = manager.add_sample(frame)
            
            if i < sample_size - 1:
                assert not is_complete, \
                    f"Calibration should not be complete at sample {i+1}/{sample_size}"
            else:
                assert is_complete, \
                    f"Calibration should be complete at sample {sample_size}"
        
        assert manager.is_calibration_complete(), \
            "Calibration should be marked complete"
        assert manager.get_sample_count() == sample_size, \
            f"Expected {sample_size} samples, got {manager.get_sample_count()}"

    @given(sample_size=st.integers(min_value=1, max_value=50))
    @settings(max_examples=100)
    def test_calibration_progress_tracking(self, sample_size):
        """
        Feature: ocr-enhancement-improvements, Property 4: Blur Auto-Calibration
        
        Verify that calibration progress is tracked correctly.
        """
        manager = CalibrationManager(sample_size=sample_size)
        
        # Create a simple test frame
        frame = np.random.randint(0, 256, size=(50, 50, 3), dtype=np.uint8)
        
        # Check progress at each step
        for i in range(sample_size):
            expected_progress = i / sample_size
            actual_progress = manager.get_calibration_progress()
            assert abs(actual_progress - expected_progress) < 1e-6, \
                f"Expected progress {expected_progress}, got {actual_progress}"
            
            manager.add_sample(frame)
        
        # After all samples, progress should be 1.0
        assert manager.get_calibration_progress() == 1.0, \
            "Progress should be 1.0 after all samples"


class TestIlluminationAutoCalibration:
    """
    Property 5: Illumination Auto-Calibration
    
    For any set of sample frames provided to the calibration manager:
    - The computed low_light_threshold SHALL be at the configured percentile of luminance values
    - The computed gamma_value SHALL be inversely proportional to mean luminance
    - Calibration SHALL complete when sample_count reaches configured sample_size
    
    Validates: Requirements 4.2, 4.3, 4.4, 4.5
    """

    @given(
        luminance_values=luminance_values_list(min_size=5, max_size=50),
        luminance_percentile=percentile_strategy
    )
    @settings(max_examples=100)
    def test_low_light_threshold_at_configured_percentile(self, luminance_values, luminance_percentile):
        """
        Feature: ocr-enhancement-improvements, Property 5: Illumination Auto-Calibration
        
        Verify that the computed low_light_threshold is at the configured
        percentile of luminance values.
        """
        # Skip if all values are identical
        assume(len(set(luminance_values)) > 1)
        
        sample_size = len(luminance_values)
        manager = CalibrationManager(
            sample_size=sample_size,
            luminance_percentile=luminance_percentile
        )
        
        # Set internal state directly
        manager._blur_scores = [100.0] * len(luminance_values)  # Dummy blur scores
        manager._luminance_values = luminance_values.copy()
        manager._calibration_complete = True
        
        result = manager.compute_calibration()
        
        assert result is not None, "Calibration should return a result"
        
        # Verify threshold is at the configured percentile
        expected_threshold = int(np.percentile(luminance_values, luminance_percentile))
        assert result.low_light_threshold == expected_threshold, \
            f"Expected low_light_threshold={expected_threshold}, got {result.low_light_threshold}"

    @given(luminance_values=luminance_values_list(min_size=5, max_size=50))
    @settings(max_examples=100)
    def test_gamma_inversely_proportional_to_luminance(self, luminance_values):
        """
        Feature: ocr-enhancement-improvements, Property 5: Illumination Auto-Calibration
        
        Verify that the computed gamma_value is inversely proportional to
        mean luminance (lower luminance -> lower gamma for brightening).
        """
        sample_size = len(luminance_values)
        manager = CalibrationManager(sample_size=sample_size)
        
        # Set internal state directly
        manager._blur_scores = [100.0] * len(luminance_values)
        manager._luminance_values = luminance_values.copy()
        manager._calibration_complete = True
        
        result = manager.compute_calibration()
        
        assert result is not None, "Calibration should return a result"
        
        mean_luminance = np.mean(luminance_values)
        
        # Gamma should be within valid range
        assert CalibrationManager.MIN_GAMMA <= result.gamma_value <= CalibrationManager.MAX_GAMMA, \
            f"gamma_value={result.gamma_value} not in [{CalibrationManager.MIN_GAMMA}, {CalibrationManager.MAX_GAMMA}]"
        
        # Lower luminance should result in lower gamma (for brightening)
        # Higher luminance should result in higher gamma (for darkening)
        # This is a monotonic relationship
        normalized_luminance = mean_luminance / 255.0
        expected_gamma = CalibrationManager.MIN_GAMMA + normalized_luminance * (CalibrationManager.MAX_GAMMA - CalibrationManager.MIN_GAMMA)
        
        assert abs(result.gamma_value - expected_gamma) < 1e-6, \
            f"Expected gamma={expected_gamma}, got {result.gamma_value}"

    @given(
        low_luminance_values=st.lists(
            st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False),
            min_size=5, max_size=20
        ),
        high_luminance_values=st.lists(
            st.floats(min_value=200.0, max_value=255.0, allow_nan=False, allow_infinity=False),
            min_size=5, max_size=20
        )
    )
    @settings(max_examples=100)
    def test_gamma_lower_for_dark_images(self, low_luminance_values, high_luminance_values):
        """
        Feature: ocr-enhancement-improvements, Property 5: Illumination Auto-Calibration
        
        Verify that dark images (low luminance) get lower gamma values
        than bright images (high luminance).
        """
        # Create manager for low luminance
        manager_low = CalibrationManager(sample_size=len(low_luminance_values))
        manager_low._blur_scores = [100.0] * len(low_luminance_values)
        manager_low._luminance_values = low_luminance_values.copy()
        manager_low._calibration_complete = True
        
        # Create manager for high luminance
        manager_high = CalibrationManager(sample_size=len(high_luminance_values))
        manager_high._blur_scores = [100.0] * len(high_luminance_values)
        manager_high._luminance_values = high_luminance_values.copy()
        manager_high._calibration_complete = True
        
        result_low = manager_low.compute_calibration()
        result_high = manager_high.compute_calibration()
        
        assert result_low is not None and result_high is not None
        
        # Low luminance should have lower gamma (for brightening)
        assert result_low.gamma_value < result_high.gamma_value, \
            f"Low luminance gamma ({result_low.gamma_value}) should be < high luminance gamma ({result_high.gamma_value})"

    @given(
        sample_size=st.integers(min_value=1, max_value=50),
        frame=random_frame_fast()
    )
    @settings(max_examples=100)
    def test_illumination_calibration_completes_at_sample_size(self, sample_size, frame):
        """
        Feature: ocr-enhancement-improvements, Property 5: Illumination Auto-Calibration
        
        Verify that illumination calibration completes when sample_count
        reaches the configured sample_size.
        """
        manager = CalibrationManager(sample_size=sample_size)
        
        # Add samples until calibration completes
        for i in range(sample_size):
            is_complete = manager.add_sample(frame)
            
            if i < sample_size - 1:
                assert not is_complete
            else:
                assert is_complete
        
        result = manager.compute_calibration()
        assert result is not None
        assert result.sample_count == sample_size


class TestCalibrationManagerEdgeCases:
    """Unit tests for edge cases and error handling."""

    def test_invalid_blur_percentile_raises_error(self):
        """Test that invalid blur percentile raises ValueError."""
        with pytest.raises(ValueError):
            CalibrationManager(blur_percentile=-1)
        
        with pytest.raises(ValueError):
            CalibrationManager(blur_percentile=101)

    def test_invalid_luminance_percentile_raises_error(self):
        """Test that invalid luminance percentile raises ValueError."""
        with pytest.raises(ValueError):
            CalibrationManager(luminance_percentile=-1)
        
        with pytest.raises(ValueError):
            CalibrationManager(luminance_percentile=101)

    def test_invalid_sample_size_raises_error(self):
        """Test that invalid sample size raises ValueError."""
        with pytest.raises(ValueError):
            CalibrationManager(sample_size=0)
        
        with pytest.raises(ValueError):
            CalibrationManager(sample_size=-1)

    def test_empty_calibration_returns_none(self):
        """Test that calibration with no samples returns None."""
        manager = CalibrationManager()
        result = manager.compute_calibration()
        assert result is None

    def test_reset_clears_state(self):
        """Test that reset clears all calibration state."""
        manager = CalibrationManager(sample_size=5)
        frame = np.random.randint(0, 256, size=(50, 50, 3), dtype=np.uint8)
        
        # Add some samples
        for _ in range(5):
            manager.add_sample(frame)
        
        assert manager.is_calibration_complete()
        
        # Reset
        manager.reset()
        
        assert not manager.is_calibration_complete()
        assert manager.get_sample_count() == 0
        assert manager.get_calibration_progress() == 0.0

    def test_calibration_result_contains_samples(self):
        """Test that calibration result contains the sample data."""
        manager = CalibrationManager(sample_size=5)
        frame = np.random.randint(0, 256, size=(50, 50, 3), dtype=np.uint8)
        
        for _ in range(5):
            manager.add_sample(frame)
        
        result = manager.compute_calibration()
        
        assert result is not None
        assert len(result.blur_scores) == 5
        assert len(result.luminance_values) == 5
        assert result.sample_count == 5
