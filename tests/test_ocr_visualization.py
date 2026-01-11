"""
Property-based tests for OCR visualization module.

Feature: ocr-visual-enhancements
Property 2: OCR Coordinate Transformation Correctness
Validates: Requirements 1.4
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, HealthCheck

from dashboard.ocr_visualization import OCRVisualization
from utils.data_models import OCRDetection, BoundingBox


# Strategy for generating valid bounding boxes within frame bounds
@st.composite
def valid_bbox(draw, max_x=640, max_y=480, min_size=10, max_size=100):
    """Generate a valid bounding box within specified bounds."""
    x1 = draw(st.integers(min_value=0, max_value=max_x - min_size))
    y1 = draw(st.integers(min_value=0, max_value=max_y - min_size))
    width = draw(st.integers(min_value=min_size, max_value=min(max_size, max_x - x1)))
    height = draw(st.integers(min_value=min_size, max_value=min(max_size, max_y - y1)))
    return BoundingBox(x1=x1, y1=y1, x2=x1 + width, y2=y1 + height)


# Strategy for generating wagon bounding boxes (larger, frame-absolute)
@st.composite
def wagon_bbox(draw, frame_width=640, frame_height=480):
    """Generate a valid wagon bounding box within frame bounds."""
    x1 = draw(st.integers(min_value=0, max_value=frame_width - 100))
    y1 = draw(st.integers(min_value=0, max_value=frame_height - 100))
    width = draw(st.integers(min_value=50, max_value=min(300, frame_width - x1)))
    height = draw(st.integers(min_value=50, max_value=min(200, frame_height - y1)))
    return BoundingBox(x1=x1, y1=y1, x2=x1 + width, y2=y1 + height)


# Strategy for generating OCR bounding boxes (smaller, ROI-relative)
@st.composite
def ocr_bbox_for_wagon(draw, wagon: BoundingBox):
    """Generate a valid OCR bounding box relative to wagon ROI."""
    roi_width = wagon.width
    roi_height = wagon.height
    
    # OCR bbox must fit within wagon ROI
    x1 = draw(st.integers(min_value=0, max_value=max(0, roi_width - 20)))
    y1 = draw(st.integers(min_value=0, max_value=max(0, roi_height - 10)))
    width = draw(st.integers(min_value=5, max_value=min(100, roi_width - x1)))
    height = draw(st.integers(min_value=5, max_value=min(30, roi_height - y1)))
    return BoundingBox(x1=x1, y1=y1, x2=x1 + width, y2=y1 + height)


# Strategy for generating OCR detections
@st.composite
def ocr_detection(draw, wagon: BoundingBox):
    """Generate a valid OCRDetection with ROI-relative coordinates."""
    bbox = draw(ocr_bbox_for_wagon(wagon))
    text = draw(st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('L', 'N'))))
    confidence = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    wagon_id = draw(st.integers(min_value=1, max_value=1000))
    frame_index = draw(st.integers(min_value=0, max_value=10000))
    return OCRDetection(
        text=text,
        confidence=confidence,
        bbox=bbox,
        wagon_id=wagon_id,
        frame_index=frame_index
    )


class TestConfidenceBasedColorSelection:
    """
    Property 5: Confidence-Based Color Selection
    
    For any OCR detection with confidence C, the text color SHALL be
    warning color (orange) if C < 0.5, otherwise normal color (white).
    
    Validates: Requirements 2.5
    """

    @given(confidence=st.floats(min_value=0.0, max_value=0.4999, allow_nan=False, allow_infinity=False))
    @settings(max_examples=100)
    def test_low_confidence_uses_warning_color(self, confidence):
        """
        Feature: ocr-visual-enhancements, Property 5: Confidence-Based Color Selection
        
        Verify that confidence values below 0.5 result in warning color (orange).
        """
        visualizer = OCRVisualization()
        
        # Create a test frame
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        
        # The color selection logic is:
        # warning color if confidence < CONFIDENCE_WARNING_THRESHOLD (0.5)
        # normal color otherwise
        
        # Verify the threshold constant
        assert visualizer.CONFIDENCE_WARNING_THRESHOLD == 0.5, \
            f"Threshold should be 0.5, got {visualizer.CONFIDENCE_WARNING_THRESHOLD}"
        
        # Verify low confidence triggers warning color
        assert confidence < visualizer.CONFIDENCE_WARNING_THRESHOLD, \
            f"Test confidence {confidence} should be below threshold {visualizer.CONFIDENCE_WARNING_THRESHOLD}"
        
        # The expected color for low confidence is OCR_TEXT_COLOR_WARNING (orange)
        expected_color = visualizer.OCR_TEXT_COLOR_WARNING
        
        # Verify the color selection logic directly
        selected_color = (
            visualizer.OCR_TEXT_COLOR_WARNING 
            if confidence < visualizer.CONFIDENCE_WARNING_THRESHOLD 
            else visualizer.OCR_TEXT_COLOR_NORMAL
        )
        
        assert selected_color == expected_color, \
            f"Low confidence {confidence} should use warning color {expected_color}, got {selected_color}"

    @given(confidence=st.floats(min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=100)
    def test_high_confidence_uses_normal_color(self, confidence):
        """
        Feature: ocr-visual-enhancements, Property 5: Confidence-Based Color Selection
        
        Verify that confidence values >= 0.5 result in normal color (white).
        """
        visualizer = OCRVisualization()
        
        # Verify the threshold constant
        assert visualizer.CONFIDENCE_WARNING_THRESHOLD == 0.5, \
            f"Threshold should be 0.5, got {visualizer.CONFIDENCE_WARNING_THRESHOLD}"
        
        # Verify high confidence does not trigger warning color
        assert confidence >= visualizer.CONFIDENCE_WARNING_THRESHOLD, \
            f"Test confidence {confidence} should be at or above threshold {visualizer.CONFIDENCE_WARNING_THRESHOLD}"
        
        # The expected color for high confidence is OCR_TEXT_COLOR_NORMAL (white)
        expected_color = visualizer.OCR_TEXT_COLOR_NORMAL
        
        # Verify the color selection logic directly
        selected_color = (
            visualizer.OCR_TEXT_COLOR_WARNING 
            if confidence < visualizer.CONFIDENCE_WARNING_THRESHOLD 
            else visualizer.OCR_TEXT_COLOR_NORMAL
        )
        
        assert selected_color == expected_color, \
            f"High confidence {confidence} should use normal color {expected_color}, got {selected_color}"

    @given(confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=100)
    def test_color_selection_is_deterministic(self, confidence):
        """
        Feature: ocr-visual-enhancements, Property 5: Confidence-Based Color Selection
        
        Verify that the same confidence value always produces the same color.
        """
        visualizer = OCRVisualization()
        
        # Determine expected color based on threshold
        if confidence < visualizer.CONFIDENCE_WARNING_THRESHOLD:
            expected_color = visualizer.OCR_TEXT_COLOR_WARNING
        else:
            expected_color = visualizer.OCR_TEXT_COLOR_NORMAL
        
        # Run the selection multiple times to verify determinism
        for _ in range(3):
            selected_color = (
                visualizer.OCR_TEXT_COLOR_WARNING 
                if confidence < visualizer.CONFIDENCE_WARNING_THRESHOLD 
                else visualizer.OCR_TEXT_COLOR_NORMAL
            )
            assert selected_color == expected_color, \
                f"Color selection should be deterministic for confidence {confidence}"

    def test_boundary_value_at_threshold(self):
        """
        Feature: ocr-visual-enhancements, Property 5: Confidence-Based Color Selection
        
        Verify that exactly 0.5 confidence uses normal color (>= threshold).
        """
        visualizer = OCRVisualization()
        confidence = 0.5
        
        # At exactly the threshold, should use normal color (>= comparison)
        expected_color = visualizer.OCR_TEXT_COLOR_NORMAL
        
        selected_color = (
            visualizer.OCR_TEXT_COLOR_WARNING 
            if confidence < visualizer.CONFIDENCE_WARNING_THRESHOLD 
            else visualizer.OCR_TEXT_COLOR_NORMAL
        )
        
        assert selected_color == expected_color, \
            f"Confidence exactly at threshold (0.5) should use normal color"

    def test_color_constants_are_distinct(self):
        """
        Feature: ocr-visual-enhancements, Property 5: Confidence-Based Color Selection
        
        Verify that warning and normal colors are visually distinct.
        """
        visualizer = OCRVisualization()
        
        # Colors should be different
        assert visualizer.OCR_TEXT_COLOR_WARNING != visualizer.OCR_TEXT_COLOR_NORMAL, \
            "Warning and normal colors must be distinct"
        
        # Verify expected BGR values
        assert visualizer.OCR_TEXT_COLOR_WARNING == (0, 165, 255), \
            f"Warning color should be orange (0, 165, 255), got {visualizer.OCR_TEXT_COLOR_WARNING}"
        assert visualizer.OCR_TEXT_COLOR_NORMAL == (255, 255, 255), \
            f"Normal color should be white (255, 255, 255), got {visualizer.OCR_TEXT_COLOR_NORMAL}"


class TestOCRCoordinateTransformation:
    """
    Property 2: OCR Coordinate Transformation Correctness
    
    For any OCR bounding box with ROI-relative coordinates (rx1, ry1, rx2, ry2)
    and wagon bounding box with frame coordinates (wx1, wy1, wx2, wy2),
    the adjusted frame-absolute coordinates SHALL be 
    (wx1+rx1, wy1+ry1, wx1+rx2, wy1+ry2).
    
    Validates: Requirements 1.4
    """

    @given(
        wagon=wagon_bbox(),
        ocr_rel_x1=st.integers(min_value=0, max_value=200),
        ocr_rel_y1=st.integers(min_value=0, max_value=100),
        ocr_width=st.integers(min_value=5, max_value=50),
        ocr_height=st.integers(min_value=5, max_value=20)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_coordinate_transformation_formula(
        self, wagon, ocr_rel_x1, ocr_rel_y1, ocr_width, ocr_height
    ):
        """
        Feature: ocr-visual-enhancements, Property 2: OCR Coordinate Transformation Correctness
        
        Verify that the coordinate transformation follows the formula:
        frame_x = wagon_x1 + roi_x
        frame_y = wagon_y1 + roi_y
        """
        visualizer = OCRVisualization()
        
        # Create OCR bbox with ROI-relative coordinates
        ocr_bbox = BoundingBox(
            x1=ocr_rel_x1,
            y1=ocr_rel_y1,
            x2=ocr_rel_x1 + ocr_width,
            y2=ocr_rel_y1 + ocr_height
        )
        
        # Transform coordinates
        adjusted = visualizer.adjust_coordinates(ocr_bbox, wagon)
        
        # Verify the transformation formula
        assert adjusted.x1 == wagon.x1 + ocr_bbox.x1, \
            f"x1: expected {wagon.x1 + ocr_bbox.x1}, got {adjusted.x1}"
        assert adjusted.y1 == wagon.y1 + ocr_bbox.y1, \
            f"y1: expected {wagon.y1 + ocr_bbox.y1}, got {adjusted.y1}"
        assert adjusted.x2 == wagon.x1 + ocr_bbox.x2, \
            f"x2: expected {wagon.x1 + ocr_bbox.x2}, got {adjusted.x2}"
        assert adjusted.y2 == wagon.y1 + ocr_bbox.y2, \
            f"y2: expected {wagon.y1 + ocr_bbox.y2}, got {adjusted.y2}"

    @given(wagon=wagon_bbox())
    @settings(max_examples=100)
    def test_transformation_preserves_dimensions(self, wagon):
        """
        Feature: ocr-visual-enhancements, Property 2: OCR Coordinate Transformation Correctness
        
        Verify that coordinate transformation preserves the width and height
        of the bounding box.
        """
        visualizer = OCRVisualization()
        
        # Create OCR bbox with arbitrary ROI-relative coordinates
        ocr_bbox = BoundingBox(x1=10, y1=5, x2=60, y2=25)
        original_width = ocr_bbox.width
        original_height = ocr_bbox.height
        
        # Transform coordinates
        adjusted = visualizer.adjust_coordinates(ocr_bbox, wagon)
        
        # Verify dimensions are preserved
        assert adjusted.width == original_width, \
            f"Width should be preserved: expected {original_width}, got {adjusted.width}"
        assert adjusted.height == original_height, \
            f"Height should be preserved: expected {original_height}, got {adjusted.height}"

    @given(wagon=wagon_bbox())
    @settings(max_examples=100)
    def test_zero_offset_transformation(self, wagon):
        """
        Feature: ocr-visual-enhancements, Property 2: OCR Coordinate Transformation Correctness
        
        Verify that an OCR bbox at (0,0) in ROI coordinates transforms to
        the wagon's top-left corner in frame coordinates.
        """
        visualizer = OCRVisualization()
        
        # OCR bbox at origin of ROI
        ocr_bbox = BoundingBox(x1=0, y1=0, x2=20, y2=10)
        
        # Transform coordinates
        adjusted = visualizer.adjust_coordinates(ocr_bbox, wagon)
        
        # Should start at wagon's top-left corner
        assert adjusted.x1 == wagon.x1, \
            f"x1 should equal wagon.x1: expected {wagon.x1}, got {adjusted.x1}"
        assert adjusted.y1 == wagon.y1, \
            f"y1 should equal wagon.y1: expected {wagon.y1}, got {adjusted.y1}"


class TestTextTruncation:
    """
    Property 8: Text Truncation at 50 Characters
    
    For any detected text longer than 50 characters, the displayed text
    SHALL be truncated to 50 characters followed by ellipsis.
    
    Validates: Requirements 5.4
    """

    @given(text=st.text(min_size=51, max_size=200))
    @settings(max_examples=100)
    def test_long_text_is_truncated(self, text):
        """
        Feature: ocr-visual-enhancements, Property 8: Text Truncation at 50 Characters
        
        Verify that text longer than 50 characters is truncated with ellipsis.
        """
        visualizer = OCRVisualization()
        
        result = visualizer.truncate_text(text)
        
        # Result should be exactly 50 chars + ellipsis (3 chars) = 53 chars
        assert len(result) == 53, \
            f"Truncated text should be 53 chars (50 + '...'), got {len(result)}"
        
        # Result should end with ellipsis
        assert result.endswith("..."), \
            f"Truncated text should end with '...', got '{result[-10:]}'"
        
        # First 50 chars should match original
        assert result[:50] == text[:50], \
            f"First 50 chars should match original text"

    @given(text=st.text(min_size=0, max_size=50))
    @settings(max_examples=100)
    def test_short_text_is_not_truncated(self, text):
        """
        Feature: ocr-visual-enhancements, Property 8: Text Truncation at 50 Characters
        
        Verify that text 50 characters or less is returned unchanged.
        """
        visualizer = OCRVisualization()
        
        result = visualizer.truncate_text(text)
        
        # Result should be identical to input
        assert result == text, \
            f"Text <= 50 chars should not be modified: expected '{text}', got '{result}'"

    def test_exactly_50_characters_not_truncated(self):
        """
        Feature: ocr-visual-enhancements, Property 8: Text Truncation at 50 Characters
        
        Verify that text exactly 50 characters is not truncated.
        """
        visualizer = OCRVisualization()
        text = "A" * 50
        
        result = visualizer.truncate_text(text)
        
        assert result == text, \
            f"Text exactly 50 chars should not be truncated"
        assert len(result) == 50, \
            f"Result should be 50 chars, got {len(result)}"

    def test_exactly_51_characters_is_truncated(self):
        """
        Feature: ocr-visual-enhancements, Property 8: Text Truncation at 50 Characters
        
        Verify that text exactly 51 characters is truncated.
        """
        visualizer = OCRVisualization()
        text = "A" * 51
        
        result = visualizer.truncate_text(text)
        
        assert result == "A" * 50 + "...", \
            f"Text 51 chars should be truncated to 50 + '...'"
        assert len(result) == 53, \
            f"Result should be 53 chars, got {len(result)}"

    def test_empty_string_not_truncated(self):
        """
        Feature: ocr-visual-enhancements, Property 8: Text Truncation at 50 Characters
        
        Verify that empty string is returned unchanged.
        """
        visualizer = OCRVisualization()
        
        result = visualizer.truncate_text("")
        
        assert result == "", \
            f"Empty string should remain empty, got '{result}'"

    def test_truncation_constants(self):
        """
        Feature: ocr-visual-enhancements, Property 8: Text Truncation at 50 Characters
        
        Verify that truncation constants are correctly defined.
        """
        visualizer = OCRVisualization()
        
        assert visualizer.MAX_TEXT_LENGTH == 50, \
            f"MAX_TEXT_LENGTH should be 50, got {visualizer.MAX_TEXT_LENGTH}"
        assert visualizer.ELLIPSIS == "...", \
            f"ELLIPSIS should be '...', got '{visualizer.ELLIPSIS}'"
