"""NAFNet deblurring wrapper for the Railway Wagon Inspection Pipeline.

This module provides a wrapper for the NAFNet-Deblur model that processes
only small ROI images for OCR enhancement. Full-frame deblurring is
explicitly prohibited to maintain detection accuracy.

Requirements: 4.3, 4.4, 4.7, 8.1, 8.6
"""

import numpy as np
from typing import Tuple, Optional


class FullFrameDeblurError(Exception):
    """Raised when attempting to deblur a full-frame image instead of ROI."""
    pass


class ModelNotLoadedError(Exception):
    """Raised when attempting to use the model before it's loaded."""
    pass


class NAFNetDeblur:
    """NAFNet deblurring wrapper for ROI-only processing.
    
    This class wraps the NAFNet-Deblur model and enforces strict constraints:
    - Only small ROI images can be processed (not full frames)
    - GPU memory is managed to stay within safe limits
    - Input dimensions are validated before processing
    
    The deblurring is intended ONLY for OCR ROI regions, never for
    full frames that would be fed to YOLO detection models.
    
    Attributes:
        model_path: Path to the NAFNet model weights
        device: Device to run inference on ('cuda' or 'cpu')
        max_roi_width: Maximum allowed ROI width (rejects larger inputs)
        max_roi_height: Maximum allowed ROI height (rejects larger inputs)
        model: The loaded NAFNet model (None until loaded)
    """
    
    # Default maximum ROI dimensions - anything larger is likely a full frame
    DEFAULT_MAX_ROI_WIDTH = 640
    DEFAULT_MAX_ROI_HEIGHT = 480
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        max_roi_width: int = DEFAULT_MAX_ROI_WIDTH,
        max_roi_height: int = DEFAULT_MAX_ROI_HEIGHT
    ):
        """Initialize NAFNetDeblur with model path and constraints.
        
        Args:
            model_path: Path to the NAFNet-Deblur model weights file
            device: Device for inference ('cuda' or 'cpu'). Defaults to 'cuda'.
            max_roi_width: Maximum allowed ROI width. Inputs wider than this
                          are rejected as potential full frames.
            max_roi_height: Maximum allowed ROI height. Inputs taller than this
                           are rejected as potential full frames.
        
        Raises:
            ValueError: If max_roi_width or max_roi_height are not positive
        """
        if max_roi_width <= 0:
            raise ValueError(f"max_roi_width must be positive, got {max_roi_width}")
        if max_roi_height <= 0:
            raise ValueError(f"max_roi_height must be positive, got {max_roi_height}")
        
        self.model_path = model_path
        self.device = device
        self.max_roi_width = max_roi_width
        self.max_roi_height = max_roi_height
        self.model = None
        self._model_loaded = False
        
        # Track frame dimensions for full-frame rejection
        self._full_frame_dimensions: Optional[Tuple[int, int]] = None
    
    def set_full_frame_dimensions(self, width: int, height: int) -> None:
        """Set the full frame dimensions for rejection validation.
        
        When set, any input matching these dimensions will be rejected
        as a full-frame deblur attempt.
        
        Args:
            width: Full frame width in pixels
            height: Full frame height in pixels
        """
        self._full_frame_dimensions = (width, height)
    
    def load_model(self) -> None:
        """Load the NAFNet model from disk.
        
        This method loads the model weights and prepares for inference.
        In a real implementation, this would load the actual NAFNet model.
        For this implementation, we simulate the model loading.
        
        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails
        """
        # In a real implementation, this would load the actual NAFNet model:
        # import torch
        # from nafnet import NAFNet
        # self.model = NAFNet.load(self.model_path)
        # self.model.to(self.device)
        # self.model.eval()
        
        # For now, we mark the model as loaded (simulated)
        # The actual model loading would happen here
        self._model_loaded = True
    
    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready for inference.
        
        Returns:
            True if model is loaded, False otherwise
        """
        return self._model_loaded
    
    def _validate_roi_dimensions(self, roi: np.ndarray) -> None:
        """Validate that input is a valid ROI, not a full frame.
        
        Args:
            roi: Input image as numpy array
            
        Raises:
            ValueError: If roi is None, empty, or has invalid shape
            FullFrameDeblurError: If roi dimensions suggest a full frame
        """
        if roi is None:
            raise ValueError("ROI cannot be None")
        
        if roi.size == 0:
            raise ValueError("ROI cannot be empty")
        
        if len(roi.shape) < 2:
            raise ValueError(f"ROI must be at least 2D, got shape {roi.shape}")
        
        height, width = roi.shape[:2]
        
        # Check against maximum ROI dimensions
        if width > self.max_roi_width or height > self.max_roi_height:
            raise FullFrameDeblurError(
                f"Input dimensions ({width}x{height}) exceed maximum ROI size "
                f"({self.max_roi_width}x{self.max_roi_height}). "
                f"NAFNet deblurring is only allowed for small OCR ROI regions, "
                f"not full frames."
            )
        
        # Check against known full-frame dimensions if set
        if self._full_frame_dimensions is not None:
            full_width, full_height = self._full_frame_dimensions
            if width == full_width and height == full_height:
                raise FullFrameDeblurError(
                    f"Input dimensions ({width}x{height}) match full frame dimensions. "
                    f"NAFNet deblurring is only allowed for small OCR ROI regions, "
                    f"not full frames."
                )
    
    def deblur_roi(self, roi: np.ndarray) -> np.ndarray:
        """Deblur a small ROI image.
        
        CRITICAL: This method only accepts small OCR ROI regions.
        Full-frame inputs will be rejected with FullFrameDeblurError.
        
        The deblurring process:
        1. Validate input dimensions (reject full frames)
        2. Preprocess ROI for model input
        3. Run NAFNet inference
        4. Postprocess and return deblurred ROI
        
        Args:
            roi: Input ROI image as BGR numpy array (H, W, 3) or grayscale (H, W)
            
        Returns:
            Deblurred ROI as numpy array with same shape as input
            
        Raises:
            ValueError: If roi is None, empty, or has invalid shape
            FullFrameDeblurError: If roi dimensions suggest a full frame
            ModelNotLoadedError: If model hasn't been loaded yet
        """
        # Validate ROI dimensions first
        self._validate_roi_dimensions(roi)
        
        # Check if model is loaded
        if not self._model_loaded:
            raise ModelNotLoadedError(
                "Model not loaded. Call load_model() before deblur_roi()."
            )
        
        # In a real implementation, this would run NAFNet inference:
        # 1. Convert BGR to RGB
        # 2. Normalize to [0, 1]
        # 3. Convert to tensor and add batch dimension
        # 4. Run model inference
        # 5. Convert back to numpy BGR
        
        # For now, return a simulated deblurred result
        # The actual deblurring would happen here with the real model
        return self._simulate_deblur(roi)
    
    def _simulate_deblur(self, roi: np.ndarray) -> np.ndarray:
        """Simulate deblurring for testing purposes.
        
        In a real implementation, this would be replaced by actual
        NAFNet inference. This simulation applies a slight sharpening
        effect to demonstrate the interface.
        
        Args:
            roi: Input ROI image
            
        Returns:
            Simulated deblurred ROI (same shape as input)
        """
        import cv2
        
        # Apply a simple sharpening kernel as simulation
        # Real NAFNet would produce much better results
        if len(roi.shape) == 3:
            # BGR image - apply sharpening
            kernel = np.array([
                [0, -0.5, 0],
                [-0.5, 3, -0.5],
                [0, -0.5, 0]
            ])
            sharpened = cv2.filter2D(roi, -1, kernel)
            # Clip to valid range
            return np.clip(sharpened, 0, 255).astype(np.uint8)
        else:
            # Grayscale image
            kernel = np.array([
                [0, -0.5, 0],
                [-0.5, 3, -0.5],
                [0, -0.5, 0]
            ])
            sharpened = cv2.filter2D(roi, -1, kernel)
            return np.clip(sharpened, 0, 255).astype(np.uint8)
    
    def get_memory_usage(self) -> dict:
        """Get current GPU memory usage information.
        
        Returns:
            Dictionary with memory usage statistics:
            - allocated: Currently allocated memory in bytes
            - reserved: Reserved memory in bytes
            - max_allocated: Peak allocated memory in bytes
        """
        try:
            import torch
            if torch.cuda.is_available() and self.device == 'cuda':
                return {
                    'allocated': torch.cuda.memory_allocated(),
                    'reserved': torch.cuda.memory_reserved(),
                    'max_allocated': torch.cuda.max_memory_allocated()
                }
        except ImportError:
            pass
        
        # Return zeros if CUDA not available or not using GPU
        return {
            'allocated': 0,
            'reserved': 0,
            'max_allocated': 0
        }
    
    def clear_memory_cache(self) -> None:
        """Clear GPU memory cache to free unused memory.
        
        This should be called periodically during processing to
        prevent GPU memory from accumulating.
        """
        try:
            import torch
            if torch.cuda.is_available() and self.device == 'cuda':
                torch.cuda.empty_cache()
        except ImportError:
            pass
