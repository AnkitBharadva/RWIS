"""
Property-based tests for OCR pipeline module.

Feature: railway-wagon-inspection
Validates: Requirements 4.6
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings
from unittest.mock import Mock, patch, MagicMock

from utils.data_models import OCRResult, BoundingBox
from pipelines.ocr_pipeline import OCRPipeline


# Strategy for generating valid text strings
text_strategy = st.text(
    alphabet=st.characters(whitelist_categories=('L', 'N', 'P', 'S', 'Z')),
    min_size=0,
    max_size=100
)


# Strategy for generating valid confidence scores in [0.0, 1.0]
confidence_strategy = st.floats(
    min_value=0.0,
    max_value=1.0,
    allow_nan=False,
    allow_infinity=False
)


# Strategy for generating optional bounding boxes
@st.composite
def optional_bounding_box(draw):
    """Generate an optional valid bounding box."""
    has_bbox = draw(st.booleans())
    if not has_bbox:
        return None
    
    x1 = draw(st.integers(min_value=0, max_value=1000))
    y1 = draw(st.integers(min_value=0, max_value=1000))
    width = draw(st.integers(min_value=1, max_value=500))
    height = draw(st.integers(min_value=1, max_value=500))
    return BoundingBox(x1=x1, y1=y1, x2=x1 + width, y2=y1 + height)


# Strategy for generating random OCR results
@st.composite
def random_ocr_result(draw):
    """Generate a random OCRResult object."""
    text = draw(text_strategy)
    confidence = draw(confidence_strategy)
    bbox = draw(optional_bounding_box())
    
    return OCRResult(
        text=text,
        confidence=confidence,
        bbox=bbox
    )


class TestOCROutputCompleteness:
    """
    Property 12: OCR Output Completeness
    
    For any OCR result, the output SHALL contain:
    - A text field (may be empty string if no text detected)
    - A confidence score in range [0.0, 1.0]
    
    Validates: Requirements 4.6
    """

    @given(ocr_result=random_ocr_result())
    @settings(max_examples=100)
    def test_ocr_result_has_text_field(self, ocr_result):
        """
        Feature: railway-wagon-inspection, Property 12: OCR Output Completeness
        
        Generate random OCR results.
        Verify text field exists.
        """
        # Verify text field is present
        assert hasattr(ocr_result, 'text'), "OCRResult missing text field"
        
        # Verify text is a string (may be empty)
        assert isinstance(ocr_result.text, str), \
            f"text must be a string, got {type(ocr_result.text)}"

    @given(ocr_result=random_ocr_result())
    @settings(max_examples=100)
    def test_ocr_result_confidence_in_valid_range(self, ocr_result):
        """
        Feature: railway-wagon-inspection, Property 12: OCR Output Completeness
        
        Generate random OCR results.
        Verify confidence is in [0.0, 1.0].
        """
        # Verify confidence field is present
        assert hasattr(ocr_result, 'confidence'), "OCRResult missing confidence field"
        
        # Verify confidence is numeric
        assert isinstance(ocr_result.confidence, (int, float)), \
            f"confidence must be numeric, got {type(ocr_result.confidence)}"
        
        # Verify confidence is in valid range [0.0, 1.0]
        assert 0.0 <= ocr_result.confidence <= 1.0, \
            f"confidence must be in [0.0, 1.0], got {ocr_result.confidence}"

    @given(ocr_result=random_ocr_result())
    @settings(max_examples=100)
    def test_ocr_result_completeness(self, ocr_result):
        """
        Feature: railway-wagon-inspection, Property 12: OCR Output Completeness
        
        Generate random OCR results.
        Verify all required fields are present and valid.
        """
        # Verify text field exists and is a string
        assert hasattr(ocr_result, 'text'), "OCRResult missing text field"
        assert isinstance(ocr_result.text, str), \
            f"text must be a string, got {type(ocr_result.text)}"
        
        # Verify confidence field exists and is in valid range
        assert hasattr(ocr_result, 'confidence'), "OCRResult missing confidence field"
        assert isinstance(ocr_result.confidence, (int, float)), \
            f"confidence must be numeric, got {type(ocr_result.confidence)}"
        assert 0.0 <= ocr_result.confidence <= 1.0, \
            f"confidence must be in [0.0, 1.0], got {ocr_result.confidence}"
        
        # Verify bbox field exists (may be None)
        assert hasattr(ocr_result, 'bbox'), "OCRResult missing bbox field"
        
        # If bbox is present, verify it's valid
        if ocr_result.bbox is not None:
            assert isinstance(ocr_result.bbox, BoundingBox), \
                f"bbox must be BoundingBox or None, got {type(ocr_result.bbox)}"
            assert ocr_result.bbox.width > 0, \
                f"bbox width must be positive, got {ocr_result.bbox.width}"
            assert ocr_result.bbox.height > 0, \
                f"bbox height must be positive, got {ocr_result.bbox.height}"

    @given(
        text=text_strategy,
        confidence=confidence_strategy,
        bbox=optional_bounding_box()
    )
    @settings(max_examples=100)
    def test_ocr_result_construction(self, text, confidence, bbox):
        """
        Feature: railway-wagon-inspection, Property 12: OCR Output Completeness
        
        Verify that OCRResult can be constructed with valid parameters
        and all fields are accessible.
        """
        # Construct OCR result
        ocr_result = OCRResult(
            text=text,
            confidence=confidence,
            bbox=bbox
        )
        
        # Verify all fields match input
        assert ocr_result.text == text, \
            "text not preserved during construction"
        assert ocr_result.confidence == confidence, \
            "confidence not preserved during construction"
        assert ocr_result.bbox == bbox, \
            "bbox not preserved during construction"

    @given(results=st.lists(random_ocr_result(), min_size=1, max_size=10))
    @settings(max_examples=100)
    def test_multiple_ocr_results_completeness(self, results):
        """
        Feature: railway-wagon-inspection, Property 12: OCR Output Completeness
        
        Verify that multiple OCR results all have complete and valid fields.
        """
        for i, ocr_result in enumerate(results):
            # Verify text field
            assert hasattr(ocr_result, 'text'), \
                f"OCRResult {i} missing text field"
            assert isinstance(ocr_result.text, str), \
                f"OCRResult {i} text has wrong type"
            
            # Verify confidence field
            assert hasattr(ocr_result, 'confidence'), \
                f"OCRResult {i} missing confidence field"
            assert isinstance(ocr_result.confidence, (int, float)), \
                f"OCRResult {i} confidence has wrong type"
            assert 0.0 <= ocr_result.confidence <= 1.0, \
                f"OCRResult {i} confidence out of range: {ocr_result.confidence}"

    @given(confidence=st.floats(allow_nan=True, allow_infinity=True))
    @settings(max_examples=100)
    def test_confidence_boundary_validation(self, confidence):
        """
        Feature: railway-wagon-inspection, Property 12: OCR Output Completeness
        
        Test that confidence values outside [0.0, 1.0] or invalid floats
        can be detected.
        """
        # Create OCR result with potentially invalid confidence
        ocr_result = OCRResult(text="test", confidence=confidence, bbox=None)
        
        # Check if confidence is valid
        is_valid = (
            isinstance(ocr_result.confidence, (int, float)) and
            not np.isnan(ocr_result.confidence) and
            not np.isinf(ocr_result.confidence) and
            0.0 <= ocr_result.confidence <= 1.0
        )
        
        # If confidence is invalid, we should be able to detect it
        if np.isnan(confidence) or np.isinf(confidence):
            assert not is_valid, "NaN/Inf confidence should be detected as invalid"
        elif confidence < 0.0 or confidence > 1.0:
            assert not is_valid, "Out-of-range confidence should be detected as invalid"



class TestOCRPipelineInitialization:
    """Tests for OCRPipeline initialization and configuration."""
    
    def test_default_initialization(self):
        """Test OCRPipeline initializes with default parameters."""
        pipeline = OCRPipeline()
        
        assert pipeline.gpu_enabled == True
        assert pipeline.language == 'en'
        assert pipeline.low_light_threshold == 80
        assert pipeline.gamma_value == 1.5
        assert not pipeline.is_initialized()
    
    def test_custom_initialization(self):
        """Test OCRPipeline initializes with custom parameters."""
        pipeline = OCRPipeline(
            gpu_enabled=False,
            language='ch',
            low_light_threshold=100,
            gamma_value=2.0
        )
        
        assert pipeline.gpu_enabled == False
        assert pipeline.language == 'ch'
        assert pipeline.low_light_threshold == 100
        assert pipeline.gamma_value == 2.0
    
    def test_invalid_low_light_threshold(self):
        """Test that invalid low_light_threshold raises ValueError."""
        with pytest.raises(ValueError):
            OCRPipeline(low_light_threshold=-1)
        
        with pytest.raises(ValueError):
            OCRPipeline(low_light_threshold=256)
    
    def test_invalid_gamma_value(self):
        """Test that invalid gamma_value raises ValueError."""
        with pytest.raises(ValueError):
            OCRPipeline(gamma_value=0)
        
        with pytest.raises(ValueError):
            OCRPipeline(gamma_value=-1)


class TestOCRPipelineLowLightDetection:
    """Tests for low-light detection functionality."""
    
    def test_dark_image_detected_as_low_light(self):
        """Test that dark images are detected as low-light."""
        pipeline = OCRPipeline(low_light_threshold=80)
        
        # Create a dark image (mean value ~30)
        dark_roi = np.full((100, 100, 3), 30, dtype=np.uint8)
        
        assert pipeline._is_low_light(dark_roi) == True
    
    def test_bright_image_not_detected_as_low_light(self):
        """Test that bright images are not detected as low-light."""
        pipeline = OCRPipeline(low_light_threshold=80)
        
        # Create a bright image (mean value ~200)
        bright_roi = np.full((100, 100, 3), 200, dtype=np.uint8)
        
        assert pipeline._is_low_light(bright_roi) == False
    
    def test_threshold_boundary(self):
        """Test low-light detection at threshold boundary."""
        pipeline = OCRPipeline(low_light_threshold=80)
        
        # Image exactly at threshold
        at_threshold = np.full((100, 100, 3), 80, dtype=np.uint8)
        assert pipeline._is_low_light(at_threshold) == False
        
        # Image just below threshold
        below_threshold = np.full((100, 100, 3), 79, dtype=np.uint8)
        assert pipeline._is_low_light(below_threshold) == True
    
    @given(threshold=st.integers(min_value=0, max_value=255))
    @settings(max_examples=50)
    def test_threshold_setter(self, threshold):
        """Test that threshold can be set dynamically."""
        pipeline = OCRPipeline()
        pipeline.set_low_light_threshold(threshold)
        assert pipeline.get_low_light_threshold() == threshold


class TestOCRPipelineGammaCorrection:
    """Tests for gamma correction functionality."""
    
    def test_gamma_brightens_dark_image(self):
        """Test that gamma correction brightens dark images."""
        pipeline = OCRPipeline(gamma_value=2.0)
        
        # Create a dark image
        dark_roi = np.full((100, 100, 3), 50, dtype=np.uint8)
        
        # Apply gamma correction
        corrected = pipeline._apply_adaptive_gamma(dark_roi)
        
        # Corrected image should be brighter
        assert np.mean(corrected) > np.mean(dark_roi)
    
    def test_gamma_preserves_shape(self):
        """Test that gamma correction preserves image shape."""
        pipeline = OCRPipeline(gamma_value=1.5)
        
        roi = np.random.randint(0, 256, (100, 150, 3), dtype=np.uint8)
        corrected = pipeline._apply_adaptive_gamma(roi)
        
        assert corrected.shape == roi.shape
    
    def test_gamma_preserves_dtype(self):
        """Test that gamma correction preserves data type."""
        pipeline = OCRPipeline(gamma_value=1.5)
        
        roi = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        corrected = pipeline._apply_adaptive_gamma(roi)
        
        assert corrected.dtype == np.uint8
    
    @given(gamma=st.floats(min_value=0.1, max_value=5.0))
    @settings(max_examples=50)
    def test_gamma_setter(self, gamma):
        """Test that gamma can be set dynamically."""
        pipeline = OCRPipeline()
        pipeline.set_gamma_value(gamma)
        assert pipeline.get_gamma_value() == gamma
    
    def test_gamma_empty_roi_raises_error(self):
        """Test that empty ROI raises ValueError."""
        pipeline = OCRPipeline()
        
        with pytest.raises(ValueError):
            pipeline._apply_adaptive_gamma(np.array([]))
        
        with pytest.raises(ValueError):
            pipeline._apply_adaptive_gamma(None)


class TestOCRPipelineExtraction:
    """Tests for text extraction functionality."""
    
    def test_extract_text_validates_input(self):
        """Test that extract_text validates input ROI."""
        pipeline = OCRPipeline()
        
        with pytest.raises(ValueError):
            pipeline.extract_text(None)
        
        with pytest.raises(ValueError):
            pipeline.extract_text(np.array([]))
    
    def test_extract_text_returns_ocr_result(self):
        """Test that extract_text returns OCRResult object."""
        pipeline = OCRPipeline()
        
        # Create a test ROI
        roi = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)
        
        # Extract text (will use simulation mode if PaddleOCR not installed)
        result = pipeline.extract_text(roi)
        
        assert isinstance(result, OCRResult)
        assert hasattr(result, 'text')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'bbox')
    
    def test_extract_text_with_low_light_override(self):
        """Test extract_text with explicit low_light parameter."""
        pipeline = OCRPipeline()
        
        roi = np.full((100, 100, 3), 200, dtype=np.uint8)  # Bright image
        
        # Force low-light enhancement
        result = pipeline.extract_text(roi, is_low_light=True)
        assert isinstance(result, OCRResult)
        
        # Disable low-light enhancement
        result = pipeline.extract_text(roi, is_low_light=False)
        assert isinstance(result, OCRResult)
    
    @given(
        height=st.integers(min_value=10, max_value=500),
        width=st.integers(min_value=10, max_value=500)
    )
    @settings(max_examples=50, deadline=None)
    def test_extract_text_various_sizes(self, height, width):
        """Test extract_text with various ROI sizes."""
        pipeline = OCRPipeline()
        
        roi = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        result = pipeline.extract_text(roi)
        
        assert isinstance(result, OCRResult)
        assert isinstance(result.text, str)
        assert 0.0 <= result.confidence <= 1.0


class TestOCRPipelineWithMockedEasyOCR:
    """Tests using mocked EasyOCR for controlled testing."""
    
    def test_successful_ocr_extraction(self):
        """Test successful OCR extraction with mocked EasyOCR."""
        pipeline = OCRPipeline()
        
        # Mock EasyOCR result format: [(bbox_points, text, confidence), ...]
        mock_result = [
            ([[0, 0], [100, 0], [100, 30], [0, 30]], 'WAGON123', 0.95),
            ([[0, 40], [80, 40], [80, 70], [0, 70]], 'TEST', 0.88)
        ]
        
        # Create a proper mock object
        mock_ocr = MagicMock()
        mock_ocr.readtext.return_value = mock_result
        
        pipeline._ocr_initialized = True
        pipeline._ocr = mock_ocr
        
        roi = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)
        result = pipeline._run_ocr(roi)
        
        assert 'WAGON123' in result.text
        assert 'TEST' in result.text
        assert result.confidence > 0
    
    def test_empty_ocr_result(self):
        """Test handling of empty OCR result."""
        pipeline = OCRPipeline()
        
        # Create a proper mock object with empty result
        mock_ocr = MagicMock()
        mock_ocr.readtext.return_value = []
        
        pipeline._ocr_initialized = True
        pipeline._ocr = mock_ocr
        
        roi = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)
        result = pipeline._run_ocr(roi)
        
        assert result.text == ""
        assert result.confidence == 0.0
    
    def test_ocr_exception_handling(self):
        """Test that OCR exceptions are handled gracefully."""
        pipeline = OCRPipeline()
        
        # Create a proper mock object that raises exception
        mock_ocr = MagicMock()
        mock_ocr.readtext.side_effect = Exception("OCR failed")
        
        pipeline._ocr_initialized = True
        pipeline._ocr = mock_ocr
        
        roi = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)
        result = pipeline._run_ocr(roi)
        
        # Should return empty result, not raise exception
        assert result.text == ""
        assert result.confidence == 0.0


class TestOCRPipelinePreprocessing:
    """Tests for ROI preprocessing functionality."""
    
    def test_preprocess_converts_bgr_to_rgb(self):
        """Test that preprocessing converts BGR to RGB."""
        pipeline = OCRPipeline()
        
        # Create BGR image with distinct channel values
        bgr_roi = np.zeros((100, 100, 3), dtype=np.uint8)
        bgr_roi[:, :, 0] = 255  # Blue channel
        bgr_roi[:, :, 1] = 128  # Green channel
        bgr_roi[:, :, 2] = 64   # Red channel
        
        rgb_roi = pipeline._preprocess_roi(bgr_roi)
        
        # After conversion, red should be in channel 0
        assert rgb_roi[:, :, 0].mean() == 64   # Was red (channel 2)
        assert rgb_roi[:, :, 1].mean() == 128  # Green stays same
        assert rgb_roi[:, :, 2].mean() == 255  # Was blue (channel 0)
    
    def test_preprocess_handles_grayscale(self):
        """Test that preprocessing handles grayscale images."""
        pipeline = OCRPipeline()
        
        gray_roi = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        rgb_roi = pipeline._preprocess_roi(gray_roi)
        
        assert rgb_roi.shape == (100, 100, 3)
        assert rgb_roi.dtype == np.uint8
    
    def test_preprocess_ensures_uint8(self):
        """Test that preprocessing ensures uint8 dtype."""
        pipeline = OCRPipeline()
        
        # Create float image
        float_roi = np.random.rand(100, 100, 3) * 255
        uint8_roi = pipeline._preprocess_roi(float_roi.astype(np.float32))
        
        assert uint8_roi.dtype == np.uint8



class TestOCRResultCompletenessProperty:
    """
    Property 1: OCR Result Completeness
    
    For any ROI passed to the OCR pipeline, the returned OCRResult SHALL contain:
    - A text field (string, may be empty)
    - A confidence score in range [0.0, 1.0]
    - An optional bounding box (may be None if no text detected)
    
    Feature: ocr-enhancement-improvements, Property 1: OCR Result Completeness
    Validates: Requirements 1.5
    """
    
    @st.composite
    def valid_roi_strategy(draw):
        """Generate valid ROI images for testing."""
        height = draw(st.integers(min_value=10, max_value=200))
        width = draw(st.integers(min_value=10, max_value=200))
        channels = draw(st.sampled_from([1, 3]))  # Grayscale or BGR
        
        if channels == 1:
            return np.random.randint(0, 256, (height, width), dtype=np.uint8)
        else:
            return np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
    
    @given(
        height=st.integers(min_value=10, max_value=200),
        width=st.integers(min_value=10, max_value=200)
    )
    @settings(max_examples=100, deadline=None)
    def test_ocr_result_has_text_field_for_any_roi(self, height, width):
        """
        Feature: ocr-enhancement-improvements, Property 1: OCR Result Completeness
        
        For any valid ROI, the OCRResult must have a text field that is a string.
        Validates: Requirements 1.5
        """
        pipeline = OCRPipeline()
        
        # Generate random ROI
        roi = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        
        # Extract text
        result = pipeline.extract_text(roi)
        
        # Verify text field exists and is a string
        assert hasattr(result, 'text'), "OCRResult must have text field"
        assert isinstance(result.text, str), f"text must be string, got {type(result.text)}"
    
    @given(
        height=st.integers(min_value=10, max_value=200),
        width=st.integers(min_value=10, max_value=200)
    )
    @settings(max_examples=100, deadline=None)
    def test_ocr_result_confidence_in_valid_range_for_any_roi(self, height, width):
        """
        Feature: ocr-enhancement-improvements, Property 1: OCR Result Completeness
        
        For any valid ROI, the OCRResult confidence must be in range [0.0, 1.0].
        Validates: Requirements 1.5
        """
        pipeline = OCRPipeline()
        
        # Generate random ROI
        roi = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        
        # Extract text
        result = pipeline.extract_text(roi)
        
        # Verify confidence field exists and is in valid range
        assert hasattr(result, 'confidence'), "OCRResult must have confidence field"
        assert isinstance(result.confidence, (int, float)), \
            f"confidence must be numeric, got {type(result.confidence)}"
        assert 0.0 <= result.confidence <= 1.0, \
            f"confidence must be in [0.0, 1.0], got {result.confidence}"
    
    @given(
        height=st.integers(min_value=10, max_value=200),
        width=st.integers(min_value=10, max_value=200)
    )
    @settings(max_examples=100, deadline=None)
    def test_ocr_result_bbox_is_optional_for_any_roi(self, height, width):
        """
        Feature: ocr-enhancement-improvements, Property 1: OCR Result Completeness
        
        For any valid ROI, the OCRResult bbox must be either None or a valid BoundingBox.
        Validates: Requirements 1.5
        """
        pipeline = OCRPipeline()
        
        # Generate random ROI
        roi = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        
        # Extract text
        result = pipeline.extract_text(roi)
        
        # Verify bbox field exists
        assert hasattr(result, 'bbox'), "OCRResult must have bbox field"
        
        # Verify bbox is either None or a valid BoundingBox
        if result.bbox is not None:
            assert isinstance(result.bbox, BoundingBox), \
                f"bbox must be BoundingBox or None, got {type(result.bbox)}"
            # If bbox exists, verify it has valid dimensions
            assert result.bbox.width > 0, f"bbox width must be positive, got {result.bbox.width}"
            assert result.bbox.height > 0, f"bbox height must be positive, got {result.bbox.height}"
    
    @given(
        height=st.integers(min_value=10, max_value=200),
        width=st.integers(min_value=10, max_value=200),
        is_grayscale=st.booleans()
    )
    @settings(max_examples=100, deadline=None)
    def test_ocr_result_completeness_for_any_roi(self, height, width, is_grayscale):
        """
        Feature: ocr-enhancement-improvements, Property 1: OCR Result Completeness
        
        For any valid ROI (grayscale or color), the OCRResult must be complete:
        - text field (string)
        - confidence in [0.0, 1.0]
        - bbox is None or valid BoundingBox
        
        Validates: Requirements 1.5
        """
        pipeline = OCRPipeline()
        
        # Generate random ROI (grayscale or color)
        if is_grayscale:
            roi = np.random.randint(0, 256, (height, width), dtype=np.uint8)
        else:
            roi = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        
        # Extract text
        result = pipeline.extract_text(roi)
        
        # Verify result is OCRResult
        assert isinstance(result, OCRResult), f"Result must be OCRResult, got {type(result)}"
        
        # Verify text field
        assert hasattr(result, 'text'), "OCRResult must have text field"
        assert isinstance(result.text, str), f"text must be string, got {type(result.text)}"
        
        # Verify confidence field
        assert hasattr(result, 'confidence'), "OCRResult must have confidence field"
        assert isinstance(result.confidence, (int, float)), \
            f"confidence must be numeric, got {type(result.confidence)}"
        assert 0.0 <= result.confidence <= 1.0, \
            f"confidence must be in [0.0, 1.0], got {result.confidence}"
        
        # Verify bbox field
        assert hasattr(result, 'bbox'), "OCRResult must have bbox field"
        if result.bbox is not None:
            assert isinstance(result.bbox, BoundingBox), \
                f"bbox must be BoundingBox or None, got {type(result.bbox)}"



class TestEasyOCRInitialization:
    """
    Unit tests for EasyOCR pipeline initialization.
    
    Tests initialization with GPU enabled/disabled, language configuration,
    and empty/invalid ROI handling.
    
    Validates: Requirements 1.2, 1.3, 1.6
    """
    
    def test_initialization_with_gpu_enabled(self):
        """
        Test OCRPipeline initializes with GPU enabled.
        Validates: Requirements 1.2
        """
        pipeline = OCRPipeline(gpu_enabled=True)
        
        assert pipeline.gpu_enabled == True
        assert not pipeline.is_initialized()  # Lazy initialization
    
    def test_initialization_with_gpu_disabled(self):
        """
        Test OCRPipeline initializes with GPU disabled.
        Validates: Requirements 1.3
        """
        pipeline = OCRPipeline(gpu_enabled=False)
        
        assert pipeline.gpu_enabled == False
        assert not pipeline.is_initialized()  # Lazy initialization
    
    def test_initialization_with_default_language(self):
        """
        Test OCRPipeline initializes with default English language.
        Validates: Requirements 1.4
        """
        pipeline = OCRPipeline()
        
        assert pipeline.language == 'en'
    
    def test_initialization_with_custom_language(self):
        """
        Test OCRPipeline initializes with custom language.
        Validates: Requirements 1.4
        """
        # Test with Chinese
        pipeline_ch = OCRPipeline(language='ch_sim')
        assert pipeline_ch.language == 'ch_sim'
        
        # Test with French
        pipeline_fr = OCRPipeline(language='fr')
        assert pipeline_fr.language == 'fr'
        
        # Test with German
        pipeline_de = OCRPipeline(language='de')
        assert pipeline_de.language == 'de'
    
    def test_lazy_initialization(self):
        """
        Test that OCR is lazily initialized only when needed.
        """
        pipeline = OCRPipeline()
        
        # Should not be initialized yet
        assert not pipeline.is_initialized()
        assert pipeline._ocr is None
        
        # After calling is_available, should be initialized
        pipeline.is_available()
        assert pipeline.is_initialized()
    
    def test_initialization_error_message_when_easyocr_not_available(self):
        """
        Test that initialization error message is set when EasyOCR fails.
        Validates: Requirements 1.7
        """
        pipeline = OCRPipeline()
        
        # Mock the import to fail
        with patch.dict('sys.modules', {'easyocr': None}):
            # Force re-initialization
            pipeline._ocr_initialized = False
            pipeline._ocr = None
            pipeline._initialization_error = None
            
            # This should handle the import error gracefully
            is_available = pipeline.is_available()
            
            # If EasyOCR is not installed, should return False
            # and have an error message
            if not is_available:
                error = pipeline.get_initialization_error()
                assert error is not None
                assert "EasyOCR" in error or "not installed" in error.lower()


class TestEasyOCREmptyInvalidROIHandling:
    """
    Unit tests for empty/invalid ROI handling.
    
    Validates: Requirements 1.6
    """
    
    def test_extract_text_with_none_roi_raises_error(self):
        """
        Test that extract_text raises ValueError for None ROI.
        Validates: Requirements 1.6
        """
        pipeline = OCRPipeline()
        
        with pytest.raises(ValueError, match="ROI cannot be None or empty"):
            pipeline.extract_text(None)
    
    def test_extract_text_with_empty_roi_raises_error(self):
        """
        Test that extract_text raises ValueError for empty ROI.
        Validates: Requirements 1.6
        """
        pipeline = OCRPipeline()
        
        with pytest.raises(ValueError, match="ROI cannot be None or empty"):
            pipeline.extract_text(np.array([]))
    
    def test_extract_text_with_zero_size_roi_raises_error(self):
        """
        Test that extract_text raises ValueError for zero-size ROI.
        Validates: Requirements 1.6
        """
        pipeline = OCRPipeline()
        
        # Create zero-size array
        zero_roi = np.zeros((0, 0, 3), dtype=np.uint8)
        
        with pytest.raises(ValueError, match="ROI cannot be None or empty"):
            pipeline.extract_text(zero_roi)
    
    def test_extract_text_with_valid_roi_returns_ocr_result(self):
        """
        Test that extract_text returns OCRResult for valid ROI.
        Validates: Requirements 1.5, 1.6
        """
        pipeline = OCRPipeline()
        
        # Create valid ROI
        roi = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)
        
        result = pipeline.extract_text(roi)
        
        assert isinstance(result, OCRResult)
        assert hasattr(result, 'text')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'bbox')
    
    def test_extract_text_with_grayscale_roi(self):
        """
        Test that extract_text handles grayscale ROI correctly.
        Validates: Requirements 1.6
        """
        pipeline = OCRPipeline()
        
        # Create grayscale ROI
        roi = np.random.randint(0, 256, (100, 200), dtype=np.uint8)
        
        result = pipeline.extract_text(roi)
        
        assert isinstance(result, OCRResult)
        assert isinstance(result.text, str)
        assert 0.0 <= result.confidence <= 1.0
    
    def test_extract_text_with_float_roi(self):
        """
        Test that extract_text handles float ROI by converting to uint8.
        Validates: Requirements 1.6
        """
        pipeline = OCRPipeline()
        
        # Create float ROI
        roi = np.random.rand(100, 200, 3).astype(np.float32) * 255
        
        result = pipeline.extract_text(roi)
        
        assert isinstance(result, OCRResult)
        assert isinstance(result.text, str)


class TestEasyOCRResultParsing:
    """
    Unit tests for EasyOCR result parsing.
    """
    
    def test_parse_easyocr_result_with_valid_results(self):
        """
        Test parsing valid EasyOCR results.
        """
        pipeline = OCRPipeline()
        
        # Mock EasyOCR result format
        results = [
            ([[0, 0], [100, 0], [100, 30], [0, 30]], 'WAGON123', 0.95),
            ([[0, 40], [80, 40], [80, 70], [0, 70]], 'TEST', 0.88)
        ]
        
        texts, confidences, boxes = pipeline._parse_easyocr_result(results, min_confidence=0.5)
        
        assert len(texts) == 2
        assert 'WAGON123' in texts
        assert 'TEST' in texts
        assert len(confidences) == 2
        assert all(0.0 <= c <= 1.0 for c in confidences)
        assert len(boxes) == 2
    
    def test_parse_easyocr_result_with_confidence_filter(self):
        """
        Test that results below min_confidence are filtered out.
        """
        pipeline = OCRPipeline()
        
        results = [
            ([[0, 0], [100, 0], [100, 30], [0, 30]], 'HIGH', 0.95),
            ([[0, 40], [80, 40], [80, 70], [0, 70]], 'LOW', 0.30)
        ]
        
        texts, confidences, boxes = pipeline._parse_easyocr_result(results, min_confidence=0.5)
        
        assert len(texts) == 1
        assert 'HIGH' in texts
        assert 'LOW' not in texts
    
    def test_parse_easyocr_result_with_empty_text(self):
        """
        Test that empty text results are filtered out.
        """
        pipeline = OCRPipeline()
        
        results = [
            ([[0, 0], [100, 0], [100, 30], [0, 30]], 'VALID', 0.95),
            ([[0, 40], [80, 40], [80, 70], [0, 70]], '', 0.88),
            ([[0, 80], [80, 80], [80, 110], [0, 110]], '   ', 0.90)
        ]
        
        texts, confidences, boxes = pipeline._parse_easyocr_result(results, min_confidence=0.5)
        
        assert len(texts) == 1
        assert 'VALID' in texts
    
    def test_parse_easyocr_result_with_none_input(self):
        """
        Test parsing None input returns empty lists.
        """
        pipeline = OCRPipeline()
        
        texts, confidences, boxes = pipeline._parse_easyocr_result(None)
        
        assert texts == []
        assert confidences == []
        assert boxes == []
    
    def test_parse_easyocr_result_with_empty_list(self):
        """
        Test parsing empty list returns empty lists.
        """
        pipeline = OCRPipeline()
        
        texts, confidences, boxes = pipeline._parse_easyocr_result([])
        
        assert texts == []
        assert confidences == []
        assert boxes == []
    
    def test_parse_easyocr_result_bounding_box_conversion(self):
        """
        Test that bounding boxes are correctly converted from polygon to BoundingBox.
        """
        pipeline = OCRPipeline()
        
        # Polygon points: top-left, top-right, bottom-right, bottom-left
        results = [
            ([[10, 20], [110, 20], [110, 50], [10, 50]], 'TEXT', 0.95)
        ]
        
        texts, confidences, boxes = pipeline._parse_easyocr_result(results)
        
        assert len(boxes) == 1
        bbox = boxes[0]
        assert bbox.x1 == 10
        assert bbox.y1 == 20
        assert bbox.x2 == 110
        assert bbox.y2 == 50
        assert bbox.width == 100
        assert bbox.height == 30
