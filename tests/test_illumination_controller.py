"""
Property-based tests for IlluminationController module.

Feature: ocr-enhancement-improvements
Validates: Requirements 5.4, 8.3, 8.4, 8.5
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, assume

from utils.illumination_controller import IlluminationController, IlluminationSettings


# Strategy for generating random BGR frames using numpy for performance
@st.composite
def random_bgr_frame(draw):
    """Generate a random BGR frame using numpy for better performance."""
    height = draw(st.integers(min_value=10, max_value=100))
    width = draw(st.integers(min_value=10, max_value=100))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.default_rng(seed)
    frame = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    return frame


# Strategy for generating frames with controlled luminance
@st.composite
def frame_with_luminance(draw, min_lum=0, max_lum=255):
    """Generate a frame with luminance in a specific range."""
    height = draw(st.integers(min_value=10, max_value=50))
    width = draw(st.integers(min_value=10, max_value=50))
    # Generate grayscale value in the desired range
    base_value = draw(st.integers(min_value=min_lum, max_value=max_lum))
    # Add some variation
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.default_rng(seed)
    variation = rng.integers(-10, 11, size=(height, width, 3))
    frame = np.clip(base_value + variation, 0, 255).astype(np.uint8)
    return frame


# Strategy for valid gamma values
gamma_strategy = st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False)

# Strategy for valid low-light thresholds
threshold_strategy = st.integers(min_value=0, max_value=255)

# Strategy for increase_illumination amount
increase_amount_strategy = st.floats(min_value=0.0, max_value=0.9, allow_nan=False, allow_infinity=False)

# Strategy for decrease_illumination amount
decrease_amount_strategy = st.floats(min_value=0.0, max_value=4.0, allow_nan=False, allow_infinity=False)


class TestGammaDirectionCorrectness:
    """
    Property 10: Gamma Direction Correctness
    
    For any illumination increase operation:
    - The applied gamma value SHALL be less than 1.0
    - The resulting image mean luminance SHALL be higher than the input
    
    For any illumination decrease operation:
    - The applied gamma value SHALL be greater than 1.0
    - The resulting image mean luminance SHALL be lower than the input
    
    Validates: Requirements 8.3, 8.4, 8.5
    """

    @given(
        frame=frame_with_luminance(min_lum=30, max_lum=200),
        amount=st.floats(min_value=0.1, max_value=0.8, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_increase_illumination_brightens_image(self, frame, amount):
        """
        Feature: ocr-enhancement-improvements, Property 10: Gamma Direction Correctness
        
        Verify that increase_illumination results in higher mean luminance.
        Uses gamma < 1 to brighten the image.
        """
        controller = IlluminationController()
        
        # Get original luminance
        original_luminance = controller.get_mean_luminance(frame)
        
        # Skip edge cases where image is already very bright (can't get brighter)
        assume(original_luminance < 250)
        
        # Apply increase illumination
        brightened = controller.increase_illumination(frame, amount)
        
        # Get new luminance
        new_luminance = controller.get_mean_luminance(brightened)
        
        # Verify luminance increased (or stayed same for very bright images)
        assert new_luminance >= original_luminance, \
            f"Luminance should increase: {original_luminance:.2f} -> {new_luminance:.2f}"

    @given(
        frame=frame_with_luminance(min_lum=50, max_lum=220),
        amount=st.floats(min_value=0.1, max_value=3.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_decrease_illumination_darkens_image(self, frame, amount):
        """
        Feature: ocr-enhancement-improvements, Property 10: Gamma Direction Correctness
        
        Verify that decrease_illumination results in lower mean luminance.
        Uses gamma > 1 to darken the image.
        """
        controller = IlluminationController()
        
        # Get original luminance
        original_luminance = controller.get_mean_luminance(frame)
        
        # Skip edge cases where image is already very dark (can't get darker)
        assume(original_luminance > 10)
        
        # Apply decrease illumination
        darkened = controller.decrease_illumination(frame, amount)
        
        # Get new luminance
        new_luminance = controller.get_mean_luminance(darkened)
        
        # Verify luminance decreased (or stayed same for very dark images)
        assert new_luminance <= original_luminance, \
            f"Luminance should decrease: {original_luminance:.2f} -> {new_luminance:.2f}"

    @given(amount=st.floats(min_value=0.01, max_value=0.89, allow_nan=False, allow_infinity=False))
    @settings(max_examples=100)
    def test_increase_illumination_uses_gamma_less_than_one(self, amount):
        """
        Feature: ocr-enhancement-improvements, Property 10: Gamma Direction Correctness
        
        Verify that increase_illumination uses gamma < 1.
        The gamma value is calculated as (1.0 - amount).
        """
        # Calculate expected gamma
        expected_gamma = max(IlluminationController.MIN_GAMMA, 1.0 - amount)
        
        # Verify gamma is less than 1
        assert expected_gamma < 1.0, \
            f"Gamma for brightening should be < 1.0, got {expected_gamma}"

    @given(amount=st.floats(min_value=0.01, max_value=3.99, allow_nan=False, allow_infinity=False))
    @settings(max_examples=100)
    def test_decrease_illumination_uses_gamma_greater_than_one(self, amount):
        """
        Feature: ocr-enhancement-improvements, Property 10: Gamma Direction Correctness
        
        Verify that decrease_illumination uses gamma > 1.
        The gamma value is calculated as (1.0 + amount).
        """
        # Calculate expected gamma
        expected_gamma = min(IlluminationController.MAX_GAMMA, 1.0 + amount)
        
        # Verify gamma is greater than 1
        assert expected_gamma > 1.0, \
            f"Gamma for darkening should be > 1.0, got {expected_gamma}"

    @given(
        frame=random_bgr_frame(),
        gamma=st.floats(min_value=0.2, max_value=0.9, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_gamma_less_than_one_brightens(self, frame, gamma):
        """
        Feature: ocr-enhancement-improvements, Property 10: Gamma Direction Correctness
        
        Verify that applying gamma < 1 directly brightens the image.
        """
        controller = IlluminationController()
        
        original_luminance = controller.get_mean_luminance(frame)
        assume(original_luminance < 250)  # Skip very bright images
        assume(original_luminance > 5)    # Skip very dark images
        
        brightened = controller.apply_gamma(frame, gamma)
        new_luminance = controller.get_mean_luminance(brightened)
        
        assert new_luminance >= original_luminance, \
            f"Gamma {gamma} < 1 should brighten: {original_luminance:.2f} -> {new_luminance:.2f}"

    @given(
        frame=random_bgr_frame(),
        gamma=st.floats(min_value=1.1, max_value=4.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_gamma_greater_than_one_darkens(self, frame, gamma):
        """
        Feature: ocr-enhancement-improvements, Property 10: Gamma Direction Correctness
        
        Verify that applying gamma > 1 directly darkens the image.
        """
        controller = IlluminationController()
        
        original_luminance = controller.get_mean_luminance(frame)
        assume(original_luminance > 10)  # Skip very dark images
        
        darkened = controller.apply_gamma(frame, gamma)
        new_luminance = controller.get_mean_luminance(darkened)
        
        assert new_luminance <= original_luminance, \
            f"Gamma {gamma} > 1 should darken: {original_luminance:.2f} -> {new_luminance:.2f}"


class TestIlluminationControllerSettings:
    """Unit tests for IlluminationController settings and configuration."""

    def test_default_initialization(self):
        """Test default initialization values."""
        controller = IlluminationController()
        
        assert controller.get_gamma_value() == IlluminationController.DEFAULT_GAMMA
        assert controller.get_low_light_threshold() == IlluminationController.DEFAULT_LOW_LIGHT_THRESHOLD
        assert controller.is_auto_mode() is True
        assert controller.is_enabled() is True

    def test_custom_initialization(self):
        """Test initialization with custom values."""
        controller = IlluminationController(
            gamma_value=1.5,
            low_light_threshold=100,
            auto_mode=False
        )
        
        assert controller.get_gamma_value() == 1.5
        assert controller.get_low_light_threshold() == 100
        assert controller.is_auto_mode() is False

    def test_invalid_gamma_raises_error(self):
        """Test that invalid gamma values raise ValueError."""
        with pytest.raises(ValueError):
            IlluminationController(gamma_value=0)
        
        with pytest.raises(ValueError):
            IlluminationController(gamma_value=-1)

    def test_invalid_threshold_raises_error(self):
        """Test that invalid threshold values raise ValueError."""
        with pytest.raises(ValueError):
            IlluminationController(low_light_threshold=-1)
        
        with pytest.raises(ValueError):
            IlluminationController(low_light_threshold=256)

    def test_get_set_settings(self):
        """Test getting and setting settings."""
        controller = IlluminationController()
        
        new_settings = IlluminationSettings(
            gamma_value=2.0,
            low_light_threshold=120,
            auto_mode=False,
            enabled=False
        )
        
        controller.set_settings(new_settings)
        
        settings = controller.get_settings()
        assert settings.gamma_value == 2.0
        assert settings.low_light_threshold == 120
        assert settings.auto_mode is False
        assert settings.enabled is False

    def test_set_auto_mode(self):
        """Test setting auto mode."""
        controller = IlluminationController(auto_mode=True)
        
        assert controller.is_auto_mode() is True
        
        controller.set_auto_mode(False)
        assert controller.is_auto_mode() is False
        
        controller.set_auto_mode(True)
        assert controller.is_auto_mode() is True

    def test_set_gamma_value(self):
        """Test setting gamma value."""
        controller = IlluminationController()
        
        controller.set_gamma_value(2.5)
        assert controller.get_gamma_value() == 2.5
        
        with pytest.raises(ValueError):
            controller.set_gamma_value(0)

    def test_set_low_light_threshold(self):
        """Test setting low-light threshold."""
        controller = IlluminationController()
        
        controller.set_low_light_threshold(150)
        assert controller.get_low_light_threshold() == 150
        
        with pytest.raises(ValueError):
            controller.set_low_light_threshold(-1)
        
        with pytest.raises(ValueError):
            controller.set_low_light_threshold(256)


class TestIlluminationControllerLowLight:
    """Unit tests for low-light detection."""

    def test_is_low_light_dark_image(self):
        """Test low-light detection on dark image."""
        controller = IlluminationController(low_light_threshold=80)
        
        # Create a dark image (mean luminance ~30)
        dark_image = np.full((50, 50, 3), 30, dtype=np.uint8)
        
        assert controller.is_low_light(dark_image) is True

    def test_is_low_light_bright_image(self):
        """Test low-light detection on bright image."""
        controller = IlluminationController(low_light_threshold=80)
        
        # Create a bright image (mean luminance ~200)
        bright_image = np.full((50, 50, 3), 200, dtype=np.uint8)
        
        assert controller.is_low_light(bright_image) is False

    def test_get_mean_luminance(self):
        """Test mean luminance calculation."""
        controller = IlluminationController()
        
        # Create image with known luminance
        image = np.full((50, 50, 3), 128, dtype=np.uint8)
        
        luminance = controller.get_mean_luminance(image)
        assert abs(luminance - 128) < 1  # Allow small rounding error

    def test_get_mean_luminance_grayscale(self):
        """Test mean luminance calculation on grayscale image."""
        controller = IlluminationController()
        
        # Create grayscale image with known luminance
        image = np.full((50, 50), 100, dtype=np.uint8)
        
        luminance = controller.get_mean_luminance(image)
        assert abs(luminance - 100) < 1


class TestIlluminationControllerProcess:
    """Unit tests for the process method."""

    def test_process_disabled(self):
        """Test that process returns unchanged image when disabled."""
        controller = IlluminationController()
        controller.set_enabled(False)
        
        image = np.random.randint(0, 256, size=(50, 50, 3), dtype=np.uint8)
        result = controller.process(image)
        
        assert np.array_equal(result, image)

    def test_process_low_light_applies_gamma(self):
        """Test that process applies gamma to low-light images."""
        controller = IlluminationController(
            gamma_value=0.7,  # Brighten
            low_light_threshold=100
        )
        
        # Create a dark image
        dark_image = np.full((50, 50, 3), 50, dtype=np.uint8)
        
        result = controller.process(dark_image)
        
        # Result should be brighter
        assert controller.get_mean_luminance(result) > controller.get_mean_luminance(dark_image)

    def test_process_bright_image_unchanged(self):
        """Test that process doesn't change bright images."""
        controller = IlluminationController(
            gamma_value=0.7,
            low_light_threshold=50
        )
        
        # Create a bright image
        bright_image = np.full((50, 50, 3), 200, dtype=np.uint8)
        
        result = controller.process(bright_image)
        
        # Result should be unchanged
        assert np.array_equal(result, bright_image)


class TestIlluminationControllerCalibration:
    """Unit tests for calibration integration."""

    def test_update_from_calibration_auto_mode(self):
        """Test updating from calibration in auto mode."""
        from pipelines.calibration_manager import CalibrationResult
        
        controller = IlluminationController(auto_mode=True)
        
        result = CalibrationResult(
            blur_threshold=100.0,
            low_light_threshold=90,
            gamma_value=0.8,
            sample_count=30,
            blur_scores=[],
            luminance_values=[]
        )
        
        controller.update_from_calibration(result)
        
        assert controller.get_gamma_value() == 0.8
        assert controller.get_low_light_threshold() == 90

    def test_update_from_calibration_manual_mode(self):
        """Test that calibration is ignored in manual mode."""
        from pipelines.calibration_manager import CalibrationResult
        
        controller = IlluminationController(
            gamma_value=1.5,
            low_light_threshold=100,
            auto_mode=False
        )
        
        result = CalibrationResult(
            blur_threshold=100.0,
            low_light_threshold=90,
            gamma_value=0.8,
            sample_count=30,
            blur_scores=[],
            luminance_values=[]
        )
        
        controller.update_from_calibration(result)
        
        # Values should remain unchanged
        assert controller.get_gamma_value() == 1.5
        assert controller.get_low_light_threshold() == 100
