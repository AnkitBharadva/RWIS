"""OCR pipeline for the Railway Wagon Inspection Pipeline using EasyOCR."""
from typing import Optional, List, Tuple, Any
import os
import logging
import cv2
import numpy as np
from utils.data_models import OCRResult, BoundingBox

logger = logging.getLogger(__name__)


class OCRInitializationError(Exception):
    """Raised when OCR initialization fails."""
    pass


class OCRPipeline:
    """Pipeline for extracting text from images using EasyOCR.
    
    This pipeline uses EasyOCR for text extraction with GPU acceleration
    when available, falling back to CPU processing when GPU is unavailable.
    """
    
    DEFAULT_LOW_LIGHT_THRESHOLD = 80
    DEFAULT_GAMMA_VALUE = 1.5
    
    def __init__(
        self,
        gpu_enabled: bool = True,
        language: str = 'en',
        low_light_threshold: int = 80,
        gamma_value: float = 1.5
    ):
        """Initialize EasyOCR with configurable settings.
        
        Args:
            gpu_enabled: Whether to use GPU acceleration
            language: Language code for OCR (default: 'en')
            low_light_threshold: Luminance threshold for low-light detection (0-255)
            gamma_value: Gamma correction value for low-light enhancement (must be positive)
            
        Raises:
            ValueError: If low_light_threshold is not in range 0-255 or gamma_value is not positive
        """
        if low_light_threshold < 0 or low_light_threshold > 255:
            raise ValueError("low_light_threshold must be 0-255")
        if gamma_value <= 0:
            raise ValueError("gamma_value must be positive")
        
        self.gpu_enabled = gpu_enabled
        self.language = language
        self.low_light_threshold = low_light_threshold
        self.gamma_value = gamma_value
        self._ocr = None
        self._ocr_initialized = False
        self._initialization_error = None
        self._gamma_table = self._build_gamma_table(gamma_value)
    
    def _build_gamma_table(self, gamma: float) -> np.ndarray:
        """Build lookup table for gamma correction.
        
        Args:
            gamma: Gamma value for correction
            
        Returns:
            Lookup table for gamma correction
        """
        inv_gamma = 1.0 / gamma
        return np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
    
    def _initialize_ocr(self) -> None:
        """Lazy initialization of EasyOCR reader.
        
        Attempts to initialize EasyOCR with GPU support if enabled,
        falling back to CPU if GPU is unavailable.
        """
        if self._ocr_initialized:
            return
        
        try:
            import easyocr
            
            # Determine GPU availability
            use_gpu = self.gpu_enabled
            if use_gpu:
                try:
                    import torch
                    if not torch.cuda.is_available():
                        logger.warning("GPU requested but CUDA not available, falling back to CPU")
                        use_gpu = False
                except ImportError:
                    logger.warning("PyTorch not available, falling back to CPU for EasyOCR")
                    use_gpu = False
            
            # Initialize EasyOCR Reader
            # EasyOCR expects language as a list
            lang_list = [self.language] if isinstance(self.language, str) else self.language
            
            self._ocr = easyocr.Reader(
                lang_list,
                gpu=use_gpu,
                verbose=False
            )
            self._ocr_initialized = True
            logger.info(f"EasyOCR initialized with language={lang_list}, gpu={use_gpu}")
            
        except ImportError as e:
            self._initialization_error = (
                "EasyOCR not installed. Please install it with: pip install easyocr"
            )
            self._ocr = None
            self._ocr_initialized = True
            logger.error(self._initialization_error)
            
        except Exception as e:
            self._initialization_error = f"EasyOCR initialization failed: {str(e)}"
            self._ocr = None
            self._ocr_initialized = True
            logger.error(self._initialization_error)
    
    def is_initialized(self) -> bool:
        """Check if OCR has been initialized (attempted).
        
        Returns:
            True if initialization has been attempted
        """
        return self._ocr_initialized
    
    def is_available(self) -> bool:
        """Check if EasyOCR is properly initialized and available.
        
        Returns:
            True if EasyOCR is ready for use
        """
        self._initialize_ocr()
        return self._ocr is not None
    
    def get_initialization_error(self) -> Optional[str]:
        """Get the initialization error message if any.
        
        Returns:
            Error message string or None if no error
        """
        return self._initialization_error
    
    def _is_low_light(self, image: np.ndarray) -> bool:
        """Check if image is low-light based on mean luminance.
        
        Args:
            image: Input BGR or grayscale image
            
        Returns:
            True if image mean luminance is below threshold
        """
        if image is None or image.size == 0:
            return False
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        return np.mean(gray) < self.low_light_threshold
    
    def _apply_gamma_correction(self, image: np.ndarray) -> np.ndarray:
        """Apply gamma correction to image using lookup table.
        
        Args:
            image: Input image
            
        Returns:
            Gamma-corrected image
        """
        return cv2.LUT(image, self._gamma_table)
    
    def _apply_adaptive_gamma(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive gamma correction to image.
        
        Args:
            image: Input image
            
        Returns:
            Gamma-corrected image
            
        Raises:
            ValueError: If image is None or empty
        """
        if image is None or image.size == 0:
            raise ValueError("Image cannot be None or empty")
        return cv2.LUT(image, self._gamma_table)
    
    def _preprocess_roi(self, roi: np.ndarray) -> np.ndarray:
        """Preprocess ROI for OCR extraction.
        
        Converts image to RGB format and ensures uint8 dtype.
        
        Args:
            roi: Input ROI image
            
        Returns:
            Preprocessed RGB image
            
        Raises:
            ValueError: If ROI is None
        """
        if roi is None:
            raise ValueError("ROI cannot be None")
        
        if roi.dtype != np.uint8:
            roi = np.clip(roi, 0, 255).astype(np.uint8)
        
        if len(roi.shape) == 2:
            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
        elif roi.shape[2] == 3:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        
        return roi
    
    def get_low_light_threshold(self) -> int:
        """Get current low-light threshold.
        
        Returns:
            Current low-light threshold value (0-255)
        """
        return self.low_light_threshold
    
    def set_low_light_threshold(self, threshold: int) -> None:
        """Set low-light threshold.
        
        Args:
            threshold: New threshold value (0-255)
            
        Raises:
            ValueError: If threshold is not in range 0-255
        """
        if threshold < 0 or threshold > 255:
            raise ValueError("low_light_threshold must be 0-255")
        self.low_light_threshold = threshold
    
    def get_gamma_value(self) -> float:
        """Get current gamma value.
        
        Returns:
            Current gamma value
        """
        return self.gamma_value
    
    def set_gamma_value(self, gamma: float) -> None:
        """Set gamma value and rebuild lookup table.
        
        Args:
            gamma: New gamma value (must be positive)
            
        Raises:
            ValueError: If gamma is not positive
        """
        if gamma <= 0:
            raise ValueError("gamma_value must be positive")
        self.gamma_value = gamma
        self._gamma_table = self._build_gamma_table(gamma)
    
    def extract_text(
        self,
        roi: np.ndarray,
        min_confidence: float = 0.5,
        is_low_light: Optional[bool] = None
    ) -> OCRResult:
        """Extract text from ROI using EasyOCR.
        
        Args:
            roi: Input ROI image (BGR format)
            min_confidence: Minimum confidence threshold for text detection
            is_low_light: Override for low-light detection (None = auto-detect)
            
        Returns:
            OCRResult with text, confidence, and bounding box
            
        Raises:
            ValueError: If ROI is None or empty
        """
        if roi is None or (hasattr(roi, 'size') and roi.size == 0):
            raise ValueError("ROI cannot be None or empty")
        
        self._initialize_ocr()
        
        if self._ocr is None:
            return OCRResult(text="", confidence=0.0, bbox=None)
        
        try:
            processed = roi.copy()
            
            # Apply gamma correction for low-light images
            apply_gamma = is_low_light if is_low_light is not None else self._is_low_light(processed)
            if apply_gamma:
                processed = self._apply_gamma_correction(processed)
            
            # EasyOCR expects RGB or grayscale
            if len(processed.shape) == 3 and processed.shape[2] == 3:
                # Convert BGR to RGB for EasyOCR
                processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            
            # Run EasyOCR
            results = self._ocr.readtext(processed)
            
            # Parse results
            texts, confidences, boxes = self._parse_easyocr_result(results, min_confidence)
            
            if not texts:
                return OCRResult(text="", confidence=0.0, bbox=None)
            
            combined_text = " ".join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            first_bbox = boxes[0] if boxes else None
            
            return OCRResult(text=combined_text, confidence=avg_confidence, bbox=first_bbox)
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return OCRResult(text="", confidence=0.0, bbox=None)
    
    def extract_text_batch(
        self,
        rois: List[np.ndarray],
        min_confidence: float = 0.5
    ) -> List[OCRResult]:
        """Extract text from multiple ROIs.
        
        Args:
            rois: List of ROI images
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of OCRResult objects
        """
        return [self.extract_text(roi, min_confidence) for roi in rois]
    
    def _parse_easyocr_result(
        self,
        results: List,
        min_confidence: float = 0.0
    ) -> Tuple[List[str], List[float], List[BoundingBox]]:
        """Parse EasyOCR results into structured format.
        
        EasyOCR returns results in format:
        [([x1,y1], [x2,y2], [x3,y3], [x4,y4]), text, confidence]
        
        Args:
            results: Raw EasyOCR results
            min_confidence: Minimum confidence threshold
            
        Returns:
            Tuple of (texts, confidences, bounding_boxes)
        """
        texts, confidences, boxes = [], [], []
        
        if results is None:
            return texts, confidences, boxes
        
        try:
            for result in results:
                if result is None or len(result) < 3:
                    continue
                
                bbox_points, text, confidence = result
                
                # Filter by confidence
                if confidence < min_confidence:
                    continue
                
                # Filter empty text
                text = str(text).strip()
                if not text:
                    continue
                
                texts.append(text)
                confidences.append(float(confidence))
                
                # Parse bounding box from polygon points
                if bbox_points and len(bbox_points) >= 4:
                    try:
                        x_coords = [p[0] for p in bbox_points]
                        y_coords = [p[1] for p in bbox_points]
                        boxes.append(BoundingBox(
                            x1=int(min(x_coords)),
                            y1=int(min(y_coords)),
                            x2=int(max(x_coords)),
                            y2=int(max(y_coords))
                        ))
                    except (TypeError, IndexError, ValueError) as e:
                        logger.warning(f"Failed to parse bounding box: {e}")
                        
        except Exception as e:
            logger.warning(f"Failed to parse EasyOCR result: {e}")
        
        return texts, confidences, boxes
    
    def _parse_ocr_result(
        self,
        result: Any,
        min_confidence: float = 0.0
    ) -> Tuple[List[str], List[float], List[BoundingBox]]:
        """Parse OCR result (compatibility method).
        
        This method provides backward compatibility with the old PaddleOCR
        result format while also supporting EasyOCR format.
        
        Args:
            result: Raw OCR results
            min_confidence: Minimum confidence threshold
            
        Returns:
            Tuple of (texts, confidences, bounding_boxes)
        """
        # If it's EasyOCR format (list of tuples)
        if isinstance(result, list) and result and isinstance(result[0], tuple):
            return self._parse_easyocr_result(result, min_confidence)
        
        # Legacy PaddleOCR format handling
        texts, confidences, boxes = [], [], []
        
        if result is None:
            return texts, confidences, boxes
        
        try:
            if isinstance(result, dict):
                rec_texts = result.get('rec_texts', [])
                rec_scores = result.get('rec_scores', [])
                dt_polys = result.get('dt_polys', [])
                
                for i, (text, score) in enumerate(zip(rec_texts, rec_scores)):
                    if score >= min_confidence and text.strip():
                        texts.append(text)
                        confidences.append(float(score))
                        if i < len(dt_polys) and dt_polys[i] is not None:
                            try:
                                poly = dt_polys[i]
                                if len(poly) >= 4:
                                    x_coords = [p[0] for p in poly]
                                    y_coords = [p[1] for p in poly]
                                    boxes.append(BoundingBox(
                                        x1=int(min(x_coords)),
                                        y1=int(min(y_coords)),
                                        x2=int(max(x_coords)),
                                        y2=int(max(y_coords))
                                    ))
                            except:
                                pass
                                
            elif isinstance(result, list):
                for item in result:
                    if item is None:
                        continue
                    for line in item:
                        if line is None or len(line) < 2:
                            continue
                        box_points, text_info = line[0], line[1]
                        if text_info and len(text_info) >= 2:
                            text = str(text_info[0]).strip()
                            score = float(text_info[1])
                            if score >= min_confidence and text:
                                texts.append(text)
                                confidences.append(score)
                                if box_points and len(box_points) >= 4:
                                    try:
                                        x_coords = [p[0] for p in box_points]
                                        y_coords = [p[1] for p in box_points]
                                        boxes.append(BoundingBox(
                                            x1=int(min(x_coords)),
                                            y1=int(min(y_coords)),
                                            x2=int(max(x_coords)),
                                            y2=int(max(y_coords))
                                        ))
                                    except:
                                        pass
                                        
        except Exception as e:
            logger.warning(f"Failed to parse OCR result: {e}")
        
        return texts, confidences, boxes
    
    def _run_ocr(self, roi: np.ndarray) -> OCRResult:
        """Run OCR on ROI (internal method).
        
        Args:
            roi: Input ROI image
            
        Returns:
            OCRResult with extracted text
        """
        self._initialize_ocr()
        
        if self._ocr is None:
            return OCRResult(text="", confidence=0.0, bbox=None)
        
        try:
            # Convert BGR to RGB for EasyOCR
            if len(roi.shape) == 3 and roi.shape[2] == 3:
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            else:
                roi_rgb = roi
            
            # Run EasyOCR
            results = self._ocr.readtext(roi_rgb)
            
            texts, confidences, boxes = self._parse_easyocr_result(results, 0.0)
            
            if not texts:
                return OCRResult(text="", confidence=0.0, bbox=None)
            
            combined_text = " ".join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            first_bbox = boxes[0] if boxes else None
            
            return OCRResult(text=combined_text, confidence=avg_confidence, bbox=first_bbox)
            
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return OCRResult(text="", confidence=0.0, bbox=None)
