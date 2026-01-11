"""
Property-based tests for NAFNet deblur wrapper module.

Feature: railway-wagon-inspection
Property 10: Deblur ROI-Only Constraint
Validates: Requirements 4.4, 4.7, 8.1, 8.6
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, assume

from pipelines.nafnet_wrapper import (
    NAFNetDeblur,
    FullFrameDeblurError,
    ModelNotLoadedError
)


# Strategy for generating valid ROI dimensions (small images)
@st.composite
def valid_roi_dimensions(draw):
    """Generate valid ROI dimensions that are smaller than max limits."""
    max_width = NAFNetDeblur.DEFAULT_MAX_ROI_WIDTH
    max_height = NAFNetDeblur.DEFAULT_MAX_ROI_HEIGHT
    
    width = draw(st.integers(min_value=10, max_value=max_width))
    height = draw(st.integers(min_value=10, max_value=max_height))
    return width, height


# Strategy for generating full-frame dimensions (large images)
@st.composite
def full_frame_dimensions(draw):
    """Generate full-frame dimensions that exceed ROI limits."""
    # Full frames are typically 1280x720, 1920x1080, etc.
    # We generate dimensions larger than the max ROI limits
    max_width = NAFNetDeblur.DEFAULT_MAX_ROI_WIDTH
    max_height = NAFNetDeblur.DEFAULT_MAX_ROI_HEIGHT
    
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
        max_width = NAFNetDeblur.DEFAULT_MAX_ROI_WIDTH
    if max_height is None:
        max_height = NAFNetDeblur.DEFAULT_MAX_ROI_HEIGHT
    
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
    
    For any invocation of NAFNet deblurring:
    - The input image dimensions SHALL be smaller than the full frame dimensions
    - The input SHALL be an OCR ROI region only
    - Full-frame deblurring SHALL never occur
    
    Validates: Requirements 4.4, 4.7, 8.1, 8.6
    """

    @given(roi=random_roi_image())
    @settings(max_examples=100)
    def test_valid_roi_accepted(self, roi):
        """
        Feature: railway-wagon-inspection, Property 10: Deblur ROI-Only Constraint
        
        Verify that valid ROI-sized inputs are accepted for deblurring.
        """
        deblur = NAFNetDeblur(
            model_path="models/nafnet_deblur.pth",
            device='cpu'
        )
        deblur.load_model()
        
        # Valid ROI should be accepted without error
        result = deblur.deblur_roi(roi)
        
        # Result should have same shape as input
        assert result.shape == roi.shape, \
            f"Output shape {result.shape} doesn't match input shape {roi.shape}"

    @given(full_frame=random_full_frame_image())
    @settings(max_examples=100)
    def test_full_frame_rejected(self, full_frame):
        """
        Feature: railway-wagon-inspection, Property 10: Deblur ROI-Only Constraint
        
        Verify that full-frame inputs are rejected with FullFrameDeblurError.
        NAFNet SHALL only accept ROI-sized inputs, never full frames.
        """
        deblur = NAFNetDeblur(
            model_path="models/nafnet_deblur.pth",
            device='cpu'
        )
        deblur.load_model()
        
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
        max_roi_width = NAFNetDeblur.DEFAULT_MAX_ROI_WIDTH
        max_roi_height = NAFNetDeblur.DEFAULT_MAX_ROI_HEIGHT
        
        assert roi_width <= max_roi_width, \
            f"ROI width {roi_width} exceeds max {max_roi_width}"
        assert roi_height <= max_roi_height, \
            f"ROI height {roi_height} exceeds max {max_roi_height}"
        
        # Full frame exceeds at least one limit
        assert frame_width > max_roi_width or frame_height > max_roi_height, \
            f"Full frame ({frame_width}x{frame_height}) should exceed ROI limits"

    @given(
        width=st.integers(min_value=NAFNetDeblur.DEFAULT_MAX_ROI_WIDTH + 1, max_value=1920),
        height=st.integers(min_value=10, max_value=1080)
    )
    @settings(max_examples=100)
    def test_width_exceeds_limit_rejected(self, width, height):
        """
        Feature: railway-wagon-inspection, Property 10: Deblur ROI-Only Constraint
        
        Verify that images with width exceeding the limit are rejected.
        """
        deblur = NAFNetDeblur(
            model_path="models/nafnet_deblur.pth",
            device='cpu'
        )
        deblur.load_model()
        
        # Create image with width exceeding limit
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        with pytest.raises(FullFrameDeblurError):
            deblur.deblur_roi(image)

    @given(
        width=st.integers(min_value=10, max_value=NAFNetDeblur.DEFAULT_MAX_ROI_WIDTH),
        height=st.integers(min_value=NAFNetDeblur.DEFAULT_MAX_ROI_HEIGHT + 1, max_value=1080)
    )
    @settings(max_examples=100)
    def test_height_exceeds_limit_rejected(self, width, height):
        """
        Feature: railway-wagon-inspection, Property 10: Deblur ROI-Only Constraint
        
        Verify that images with height exceeding the limit are rejected.
        """
        deblur = NAFNetDeblur(
            model_path="models/nafnet_deblur.pth",
            device='cpu'
        )
        deblur.load_model()
        
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
        
        deblur = NAFNetDeblur(
            model_path="models/nafnet_deblur.pth",
            device='cpu',
            # Set max ROI larger than the test image
            max_roi_width=width + 100,
            max_roi_height=height + 100
        )
        deblur.load_model()
        
        # Set the ROI dimensions as "full frame" dimensions
        deblur.set_full_frame_dimensions(width, height)
        
        # Now the same-sized input should be rejected as a full frame
        with pytest.raises(FullFrameDeblurError):
            deblur.deblur_roi(roi)


class TestNAFNetDeblurValidation:
    """Additional validation tests for NAFNetDeblur."""

    def test_model_not_loaded_error(self):
        """Verify that using deblur before loading model raises error."""
        deblur = NAFNetDeblur(
            model_path="models/nafnet_deblur.pth",
            device='cpu'
        )
        
        roi = np.zeros((100, 100, 3), dtype=np.uint8)
        
        with pytest.raises(ModelNotLoadedError):
            deblur.deblur_roi(roi)

    def test_none_input_rejected(self):
        """Verify that None input is rejected."""
        deblur = NAFNetDeblur(
            model_path="models/nafnet_deblur.pth",
            device='cpu'
        )
        deblur.load_model()
        
        with pytest.raises(ValueError):
            deblur.deblur_roi(None)

    def test_empty_input_rejected(self):
        """Verify that empty input is rejected."""
        deblur = NAFNetDeblur(
            model_path="models/nafnet_deblur.pth",
            device='cpu'
        )
        deblur.load_model()
        
        empty_roi = np.array([])
        
        with pytest.raises(ValueError):
            deblur.deblur_roi(empty_roi)

    def test_invalid_max_roi_dimensions(self):
        """Verify that invalid max ROI dimensions raise ValueError."""
        with pytest.raises(ValueError):
            NAFNetDeblur(
                model_path="models/nafnet_deblur.pth",
                max_roi_width=0
            )
        
        with pytest.raises(ValueError):
            NAFNetDeblur(
                model_path="models/nafnet_deblur.pth",
                max_roi_height=-1
            )

    @given(roi=random_roi_image())
    @settings(max_examples=100)
    def test_output_same_dtype(self, roi):
        """Verify that output has same dtype as input."""
        deblur = NAFNetDeblur(
            model_path="models/nafnet_deblur.pth",
            device='cpu'
        )
        deblur.load_model()
        
        result = deblur.deblur_roi(roi)
        
        assert result.dtype == roi.dtype, \
            f"Output dtype {result.dtype} doesn't match input dtype {roi.dtype}"

    def test_is_loaded_state(self):
        """Verify is_loaded() returns correct state."""
        deblur = NAFNetDeblur(
            model_path="models/nafnet_deblur.pth",
            device='cpu'
        )
        
        assert not deblur.is_loaded(), "Model should not be loaded initially"
        
        deblur.load_model()
        
        assert deblur.is_loaded(), "Model should be loaded after load_model()"
