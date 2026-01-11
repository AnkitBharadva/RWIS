"""
Property-based tests for MPRNet deblur wrapper module.

Feature: railway-wagon-inspection
Property 10: Deblur ROI-Only Constraint
Validates: Requirements 4.4, 4.7, 11.1, 11.6
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, assume

from pipelines.mprnet_wrapper import (
    MPRNetDeblur,
    FullFrameDeblurError,
    ModelNotLoadedError,
    NumericalInstabilityError
)


# Strategy for generating valid ROI dimensions (small images)
@st.composite
def valid_roi_dimensions(draw):
    """Generate valid ROI dimensions that are smaller than max limits."""
    max_width = MPRNetDeblur.DEFAULT_MAX_ROI_WIDTH
    max_height = MPRNetDeblur.DEFAULT_MAX_ROI_HEIGHT
    
    width = draw(st.integers(min_value=10, max_value=max_width))
    height = draw(st.integers(min_value=10, max_value=max_height))
    return width, height


# Strategy for generating full-frame dimensions (large images)
@st.composite
def full_frame_dimensions(draw):
    """Generate full-frame dimensions that exceed ROI limits."""
    # Full frames are typically 1280x720, 1920x1080, etc.
    # We generate dimensions larger than the max ROI limits
    max_width = MPRNetDeblur.DEFAULT_MAX_ROI_WIDTH
    max_height = MPRNetDeblur.DEFAULT_MAX_ROI_HEIGHT
    
    # At least one dimension must exceed the limit
    exceed_width = draw(st.booleans())
    
    if exceed_width:
        width = draw(st.integers(min_value=max_width + 1, max_value=1920))
        height = draw(st.integers(min_value=100, max_value=1080))
    else:
        width = draw(st.integers(min_value=100, max_value=1920))
        height = draw(st.integers(min_value=max_height + 1, max_value=1080))
    
    return width, height


# Strategy for generating random BGR ROI images
@st.composite
def random_roi_image(draw, max_width=None, max_height=None):
    """Generate a random BGR ROI image within size limits."""
    if max_width is None:
        max_width = MPRNetDeblur.DEFAULT_MAX_ROI_WIDTH
    if max_height is None:
        max_height = MPRNetDeblur.DEFAULT_MAX_ROI_HEIGHT
    
    width = draw(st.integers(min_value=10, max_value=max_width))
    height = draw(st.integers(min_value=10, max_value=max_height))
    
    # Generate random pixel values using a seed for reproducibility
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    return image


# Strategy for generating random full-frame images
@st.composite
def random_full_frame_image(draw):
    """Generate a random BGR full-frame image that exceeds ROI limits."""
    width, height = draw(full_frame_dimensions())
    
    # Generate random pixel values using a seed for reproducibility
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    return image


class TestDeblurROIOnlyConstraint:
    """
    Property 10: Deblur ROI-Only Constraint
    
    For any invocation of MPRNet deblurring:
    - The input image dimensions SHALL be smaller than the full frame dimensions
    - The input width SHALL be <= max_roi_width (256px default)
    - Full-frame deblurring SHALL never occur
    
    Validates: Requirements 4.4, 4.7, 11.1, 11.6
    """

    @given(full_frame=random_full_frame_image())
    @settings(max_examples=100)
    def test_full_frame_rejected(self, full_frame):
        """
        Feature: railway-wagon-inspection, Property 10: Deblur ROI-Only Constraint
        
        Verify that full-frame inputs are rejected with FullFrameDeblurError.
        MPRNet SHALL only accept ROI-sized inputs, never full frames.
        """
        deblur = MPRNetDeblur(
            model_path="models/mprnet_deblur.pth",
            device='cpu',
            use_fp16=False
        )
        # Note: We don't load the model here since validation happens before inference
        # The dimension validation should reject the input before model is needed
        deblur._model_loaded = True  # Simulate loaded state for testing validation
        
        # Full frame should be rejected
        with pytest.raises(FullFrameDeblurError):
            deblur.deblur_roi(full_frame)

    @given(
        roi_dims=valid_roi_dimensions(),
        frame_dims=full_frame_dimensions()
    )
    @settings(max_examples=100)
    def test_roi_smaller_than_frame(self, roi_dims, frame_dims):
        """
        Feature: railway-wagon-inspection, Property 10: Deblur ROI-Only Constraint
        
        Verify that ROI dimensions are always smaller than full frame dimensions.
        """
        roi_width, roi_height = roi_dims
        frame_width, frame_height = frame_dims
        
        # ROI should be smaller than full frame in at least one dimension
        # (since full frame exceeds max ROI limits by definition)
        max_roi_width = MPRNetDeblur.DEFAULT_MAX_ROI_WIDTH
        max_roi_height = MPRNetDeblur.DEFAULT_MAX_ROI_HEIGHT
        
        assert roi_width <= max_roi_width, \
            f"ROI width {roi_width} exceeds max {max_roi_width}"
        assert roi_height <= max_roi_height, \
            f"ROI height {roi_height} exceeds max {max_roi_height}"
        
        # Full frame exceeds at least one limit
        assert frame_width > max_roi_width or frame_height > max_roi_height, \
            f"Full frame ({frame_width}x{frame_height}) should exceed ROI limits"

    @given(
        width=st.integers(min_value=MPRNetDeblur.DEFAULT_MAX_ROI_WIDTH + 1, max_value=1920),
        height=st.integers(min_value=10, max_value=1080)
    )
    @settings(max_examples=100)
    def test_width_exceeds_limit_rejected(self, width, height):
        """
        Feature: railway-wagon-inspection, Property 10: Deblur ROI-Only Constraint
        
        Verify that images with width exceeding the limit are rejected.
        """
        deblur = MPRNetDeblur(
            model_path="models/mprnet_deblur.pth",
            device='cpu',
            use_fp16=False
        )
        deblur._model_loaded = True  # Simulate loaded state
        
        # Create image with width exceeding limit
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        with pytest.raises(FullFrameDeblurError):
            deblur.deblur_roi(image)

    @given(
        width=st.integers(min_value=10, max_value=MPRNetDeblur.DEFAULT_MAX_ROI_WIDTH),
        height=st.integers(min_value=MPRNetDeblur.DEFAULT_MAX_ROI_HEIGHT + 1, max_value=1080)
    )
    @settings(max_examples=100)
    def test_height_exceeds_limit_rejected(self, width, height):
        """
        Feature: railway-wagon-inspection, Property 10: Deblur ROI-Only Constraint
        
        Verify that images with height exceeding the limit are rejected.
        """
        deblur = MPRNetDeblur(
            model_path="models/mprnet_deblur.pth",
            device='cpu',
            use_fp16=False
        )
        deblur._model_loaded = True  # Simulate loaded state
        
        # Create image with height exceeding limit
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        with pytest.raises(FullFrameDeblurError):
            deblur.deblur_roi(image)

    @given(roi=random_roi_image())
    @settings(max_examples=100)
    def test_known_frame_dimensions_rejected(self, roi):
        """
        Feature: railway-wagon-inspection, Property 10: Deblur ROI-Only Constraint
        
        Verify that when full frame dimensions are set, matching inputs are rejected.
        """
        height, width = roi.shape[:2]
        
        deblur = MPRNetDeblur(
            model_path="models/mprnet_deblur.pth",
            device='cpu',
            use_fp16=False,
            # Set max ROI larger than the test image
            max_roi_width=width + 100,
            max_roi_height=height + 100
        )
        deblur._model_loaded = True  # Simulate loaded state
        
        # Set the ROI dimensions as "full frame" dimensions
        deblur.set_full_frame_dimensions(width, height)
        
        # Now the same-sized input should be rejected as a full frame
        with pytest.raises(FullFrameDeblurError):
            deblur.deblur_roi(roi)

    @given(
        roi_width=st.integers(min_value=10, max_value=MPRNetDeblur.DEFAULT_MAX_ROI_WIDTH),
        roi_height=st.integers(min_value=10, max_value=MPRNetDeblur.DEFAULT_MAX_ROI_HEIGHT)
    )
    @settings(max_examples=100)
    def test_valid_roi_dimensions_accepted(self, roi_width, roi_height):
        """
        Feature: railway-wagon-inspection, Property 10: Deblur ROI-Only Constraint
        
        Verify that valid ROI dimensions pass validation (dimensions check only).
        This test verifies the dimension validation logic without running actual inference.
        """
        deblur = MPRNetDeblur(
            model_path="models/mprnet_deblur.pth",
            device='cpu',
            use_fp16=False
        )
        
        # Create a valid ROI image
        roi = np.zeros((roi_height, roi_width, 3), dtype=np.uint8)
        
        # Validation should pass (no exception)
        # We call the internal validation method directly
        deblur._validate_roi_dimensions(roi)
        
        # If we get here, validation passed
        assert True


class TestMPRNetDeblurValidation:
    """Additional validation tests for MPRNetDeblur."""

    def test_model_not_loaded_error(self):
        """Verify that using deblur before loading model raises error."""
        deblur = MPRNetDeblur(
            model_path="models/mprnet_deblur.pth",
            device='cpu',
            use_fp16=False
        )
        
        roi = np.zeros((100, 100, 3), dtype=np.uint8)
        
        with pytest.raises(ModelNotLoadedError):
            deblur.deblur_roi(roi)

    def test_none_input_rejected(self):
        """Verify that None input is rejected."""
        deblur = MPRNetDeblur(
            model_path="models/mprnet_deblur.pth",
            device='cpu',
            use_fp16=False
        )
        deblur._model_loaded = True  # Simulate loaded state
        
        with pytest.raises(ValueError):
            deblur.deblur_roi(None)

    def test_empty_input_rejected(self):
        """Verify that empty input is rejected."""
        deblur = MPRNetDeblur(
            model_path="models/mprnet_deblur.pth",
            device='cpu',
            use_fp16=False
        )
        deblur._model_loaded = True  # Simulate loaded state
        
        empty_roi = np.array([])
        
        with pytest.raises(ValueError):
            deblur.deblur_roi(empty_roi)

    def test_invalid_max_roi_dimensions(self):
        """Verify that invalid max ROI dimensions raise ValueError."""
        with pytest.raises(ValueError):
            MPRNetDeblur(
                model_path="models/mprnet_deblur.pth",
                max_roi_width=0
            )
        
        with pytest.raises(ValueError):
            MPRNetDeblur(
                model_path="models/mprnet_deblur.pth",
                max_roi_height=-1
            )

    def test_is_loaded_state(self):
        """Verify is_loaded() returns correct state."""
        deblur = MPRNetDeblur(
            model_path="models/mprnet_deblur.pth",
            device='cpu',
            use_fp16=False
        )
        
        assert not deblur.is_loaded(), "Model should not be loaded initially"

    def test_default_max_roi_width(self):
        """Verify default max ROI width is 256 as per requirements."""
        assert MPRNetDeblur.DEFAULT_MAX_ROI_WIDTH == 256, \
            "Default max ROI width should be 256 per requirements"

    def test_fp16_default_enabled(self):
        """Verify FP16 is enabled by default."""
        deblur = MPRNetDeblur(
            model_path="models/mprnet_deblur.pth"
        )
        assert deblur.use_fp16 is True, "FP16 should be enabled by default"

    def test_fp32_fallback_default_enabled(self):
        """Verify FP32 fallback is enabled by default."""
        deblur = MPRNetDeblur(
            model_path="models/mprnet_deblur.pth"
        )
        assert deblur.fp32_fallback is True, "FP32 fallback should be enabled by default"

    def test_set_full_frame_dimensions(self):
        """Verify set_full_frame_dimensions stores dimensions correctly."""
        deblur = MPRNetDeblur(
            model_path="models/mprnet_deblur.pth",
            device='cpu'
        )
        
        deblur.set_full_frame_dimensions(1920, 1080)
        
        assert deblur._full_frame_dimensions == (1920, 1080)

    def test_memory_usage_returns_dict(self):
        """Verify get_memory_usage returns a dictionary with expected keys."""
        deblur = MPRNetDeblur(
            model_path="models/mprnet_deblur.pth",
            device='cpu'
        )
        
        memory = deblur.get_memory_usage()
        
        assert isinstance(memory, dict)
        assert 'allocated' in memory
        assert 'reserved' in memory
        assert 'max_allocated' in memory

    def test_clear_memory_cache_no_error(self):
        """Verify clear_memory_cache doesn't raise errors."""
        deblur = MPRNetDeblur(
            model_path="models/mprnet_deblur.pth",
            device='cpu'
        )
        
        # Should not raise any errors
        deblur.clear_memory_cache()

    def test_1d_input_rejected(self):
        """Verify that 1D input is rejected."""
        deblur = MPRNetDeblur(
            model_path="models/mprnet_deblur.pth",
            device='cpu',
            use_fp16=False
        )
        deblur._model_loaded = True
        
        # 1D array should be rejected
        roi_1d = np.array([1, 2, 3, 4, 5])
        
        with pytest.raises(ValueError):
            deblur.deblur_roi(roi_1d)
