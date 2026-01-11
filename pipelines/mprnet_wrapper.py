"""MPRNet deblurring wrapper for the Railway Wagon Inspection Pipeline.

This module provides a wrapper for the MPRNet GoPro-trained deblurring model
that processes only small ROI images for OCR enhancement. Full-frame deblurring
is explicitly prohibited to maintain detection accuracy.

Key features:
- ROI-only processing (max 256px width by default)
- FP16 inference with FP32 fallback for numerical stability
- Batch size always 1, no tiling
- Dimension validation to reject full-frame inputs

Requirements: 4.3, 4.4, 4.7, 7.1, 7.2, 7.3, 9.2, 9.3, 11.1, 11.6
"""

from __future__ import annotations

import numpy as np
import os
import sys
from typing import Tuple, Optional, TYPE_CHECKING
from collections import OrderedDict

# Lazy import torch to allow testing without GPU
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None

if TYPE_CHECKING:
    import torch


class FullFrameDeblurError(Exception):
    """Raised when attempting to deblur a full-frame image instead of ROI."""
    pass


class ModelNotLoadedError(Exception):
    """Raised when attempting to use the model before it's loaded."""
    pass


class NumericalInstabilityError(Exception):
    """Raised when numerical instability (NaN/Inf) is detected in output."""
    pass


class MPRNetDeblur:
    """MPRNet deblurring wrapper for ROI-only processing.
    
    This class wraps the MPRNet GoPro-trained deblurring model and enforces
    strict constraints:
    - Only small ROI images can be processed (not full frames)
    - FP16 inference by default with FP32 fallback
    - Batch size is always 1, no tiling
    - Input dimensions are validated before processing
    
    The deblurring is intended ONLY for OCR ROI regions, never for
    full frames that would be fed to YOLO detection models.
    
    Attributes:
        model_path: Path to the MPRNet model weights
        device: Device to run inference on ('cuda' or 'cpu')
        use_fp16: Whether to use FP16 (half precision) inference
        fp32_fallback: Whether to fall back to FP32 on numerical instability
        max_roi_width: Maximum allowed ROI width (rejects larger inputs)
        max_roi_height: Maximum allowed ROI height (rejects larger inputs)
        model: The loaded MPRNet model (None until loaded)
    """
    
    # Default maximum ROI dimensions - anything larger is likely a full frame
    DEFAULT_MAX_ROI_WIDTH = 256
    DEFAULT_MAX_ROI_HEIGHT = 256
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        use_fp16: bool = True,
        fp32_fallback: bool = True,
        max_roi_width: int = DEFAULT_MAX_ROI_WIDTH,
        max_roi_height: int = DEFAULT_MAX_ROI_HEIGHT
    ):
        """Initialize MPRNetDeblur with model path and constraints.
        
        Args:
            model_path: Path to the MPRNet GoPro-trained model weights file
            device: Device for inference ('cuda' or 'cpu'). Defaults to 'cuda'.
            use_fp16: Whether to use FP16 (half precision) inference. Defaults to True.
            fp32_fallback: Whether to fall back to FP32 on numerical instability.
                          Defaults to True.
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
        self.use_fp16 = use_fp16
        self.fp32_fallback = fp32_fallback
        self.max_roi_width = max_roi_width
        self.max_roi_height = max_roi_height
        self.model = None
        self._model_loaded = False
        self._using_fp32_fallback = False
        
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
        """Load the MPRNet model from disk.
        
        This method loads the model weights and prepares for inference.
        If use_fp16 is True and CUDA is available, the model is converted
        to half precision.
        
        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails
            ImportError: If torch is not available
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for MPRNet inference but is not installed")
        
        # Add MPRNet to path for importing
        mprnet_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'MPRNet', 'Deblurring')
        if mprnet_path not in sys.path:
            sys.path.insert(0, mprnet_path)
        
        try:
            from MPRNet import MPRNet
        except ImportError:
            # Fallback: try importing from the full path
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "MPRNet", 
                os.path.join(mprnet_path, "MPRNet.py")
            )
            mprnet_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mprnet_module)
            MPRNet = mprnet_module.MPRNet
        
        # Check if model file exists
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Create model instance
        self.model = MPRNet()
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # Handle different checkpoint formats
        # Some checkpoints have 'state_dict' key, others store state dict directly
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and any(k.startswith(('encoder', 'decoder', 'stage')) for k in checkpoint.keys()):
            # State dict stored directly (common for MPRNet pretrained models)
            state_dict = checkpoint
        elif isinstance(checkpoint, OrderedDict):
            state_dict = checkpoint
        else:
            raise RuntimeError(f"Unknown checkpoint format. Keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint)}")
        
        try:
            self.model.load_state_dict(state_dict)
        except RuntimeError:
            # Handle DataParallel saved models (keys start with 'module.')
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k  # remove `module.`
                new_state_dict[name] = v
            self.model.load_state_dict(new_state_dict)
        
        # Move to device
        if self.device == 'cuda' and torch.cuda.is_available():
            self.model = self.model.cuda()
            
            # Convert to FP16 if requested
            if self.use_fp16:
                self.model = self.model.half()
        else:
            self.device = 'cpu'
            self.use_fp16 = False  # FP16 not well supported on CPU
        
        self.model.eval()
        self._model_loaded = True
    
    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready for inference.
        
        Returns:
            True if model is loaded, False otherwise
        """
        return self._model_loaded
    
    def is_using_fp32_fallback(self) -> bool:
        """Check if the model is currently using FP32 fallback.
        
        Returns:
            True if using FP32 fallback, False otherwise
        """
        return self._using_fp32_fallback
    
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
                f"MPRNet deblurring is only allowed for small OCR ROI regions, "
                f"not full frames."
            )
        
        # Check against known full-frame dimensions if set
        if self._full_frame_dimensions is not None:
            full_width, full_height = self._full_frame_dimensions
            if width == full_width and height == full_height:
                raise FullFrameDeblurError(
                    f"Input dimensions ({width}x{height}) match full frame dimensions. "
                    f"MPRNet deblurring is only allowed for small OCR ROI regions, "
                    f"not full frames."
                )
    
    def _check_numerical_stability(self, output: "torch.Tensor") -> bool:
        """Check for NaN/Inf values in the output tensor.
        
        Args:
            output: Output tensor from model inference
            
        Returns:
            True if output is numerically stable (no NaN/Inf), False otherwise
        """
        has_nan = torch.isnan(output).any().item()
        has_inf = torch.isinf(output).any().item()
        return not (has_nan or has_inf)
    
    def _preprocess(self, roi: np.ndarray, use_fp16: bool) -> "torch.Tensor":
        """Preprocess ROI for model input.
        
        Args:
            roi: Input ROI image as BGR numpy array (H, W, 3)
            use_fp16: Whether to use FP16 precision
            
        Returns:
            Preprocessed tensor ready for model input
        """
        import cv2
        
        # Convert BGR to RGB
        if len(roi.shape) == 3 and roi.shape[2] == 3:
            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        else:
            # Grayscale - convert to 3 channel
            rgb = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
        
        # Normalize to [0, 1]
        rgb = rgb.astype(np.float32) / 255.0
        
        # Convert to tensor: (H, W, C) -> (C, H, W)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1)
        
        # Add batch dimension: (C, H, W) -> (1, C, H, W)
        tensor = tensor.unsqueeze(0)
        
        # Move to device
        if self.device == 'cuda' and torch.cuda.is_available():
            tensor = tensor.cuda()
        
        # Convert to FP16 if requested
        if use_fp16:
            tensor = tensor.half()
        
        return tensor
    
    def _postprocess(self, output: "torch.Tensor") -> np.ndarray:
        """Postprocess model output to numpy array.
        
        Args:
            output: Model output tensor
            
        Returns:
            Deblurred image as BGR numpy array (H, W, 3)
        """
        import cv2
        
        # Clamp to [0, 1]
        output = torch.clamp(output, 0, 1)
        
        # Remove batch dimension and convert to numpy: (1, C, H, W) -> (H, W, C)
        output = output.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        
        # Convert to uint8
        output = (output * 255).astype(np.uint8)
        
        # Convert RGB to BGR
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        
        return output
    
    def _pad_to_multiple(self, tensor: "torch.Tensor", factor: int = 8) -> Tuple["torch.Tensor", int, int]:
        """Pad tensor to be a multiple of factor.
        
        MPRNet requires input dimensions to be multiples of 8.
        
        Args:
            tensor: Input tensor (1, C, H, W)
            factor: Factor to pad to (default 8)
            
        Returns:
            Tuple of (padded_tensor, pad_h, pad_w)
        """
        _, _, h, w = tensor.shape
        pad_h = (factor - h % factor) % factor
        pad_w = (factor - w % factor) % factor
        
        if pad_h > 0 or pad_w > 0:
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')
        
        return tensor, pad_h, pad_w
    
    def _run_inference(self, tensor: "torch.Tensor", use_fp16: bool) -> "torch.Tensor":
        """Run model inference with optional FP16.
        
        Args:
            tensor: Preprocessed input tensor
            use_fp16: Whether to use FP16 precision
            
        Returns:
            Model output tensor
        """
        # Ensure model is in correct precision
        if use_fp16 and self.device == 'cuda':
            if not next(self.model.parameters()).dtype == torch.float16:
                self.model = self.model.half()
            tensor = tensor.half()
        else:
            if next(self.model.parameters()).dtype == torch.float16:
                self.model = self.model.float()
            tensor = tensor.float()
        
        with torch.no_grad():
            # Pad to multiple of 8
            tensor, pad_h, pad_w = self._pad_to_multiple(tensor)
            
            # Run inference - MPRNet returns [stage3, stage2, stage1]
            outputs = self.model(tensor)
            
            # Use stage3 output (best quality)
            output = outputs[0]
            
            # Remove padding
            if pad_h > 0 or pad_w > 0:
                output = output[:, :, :output.shape[2]-pad_h, :output.shape[3]-pad_w]
        
        return output
    
    def deblur_roi(self, roi: np.ndarray) -> np.ndarray:
        """Deblur a small ROI image.
        
        CRITICAL: This method only accepts small OCR ROI regions.
        Full-frame inputs will be rejected with FullFrameDeblurError.
        
        The deblurring process:
        1. Validate input dimensions (reject full frames)
        2. Preprocess ROI for model input
        3. Run MPRNet inference (FP16 with FP32 fallback)
        4. Check numerical stability
        5. Postprocess and return deblurred ROI
        
        Args:
            roi: Input ROI image as BGR numpy array (H, W, 3) or grayscale (H, W)
            
        Returns:
            Deblurred ROI as numpy array with same shape as input
            
        Raises:
            ValueError: If roi is None, empty, or has invalid shape
            FullFrameDeblurError: If roi dimensions suggest a full frame
            ModelNotLoadedError: If model hasn't been loaded yet
            NumericalInstabilityError: If FP32 fallback is disabled and instability occurs
        """
        # Validate ROI dimensions first
        self._validate_roi_dimensions(roi)
        
        # Check if model is loaded
        if not self._model_loaded:
            raise ModelNotLoadedError(
                "Model not loaded. Call load_model() before deblur_roi()."
            )
        
        # Store original shape for grayscale handling
        original_shape = roi.shape
        is_grayscale = len(roi.shape) == 2
        
        # Try FP16 first if enabled
        use_fp16 = self.use_fp16 and self.device == 'cuda'
        
        # Preprocess
        tensor = self._preprocess(roi, use_fp16)
        
        # Run inference
        output = self._run_inference(tensor, use_fp16)
        
        # Check numerical stability
        if not self._check_numerical_stability(output):
            if self.fp32_fallback and use_fp16:
                # Fall back to FP32
                self._using_fp32_fallback = True
                tensor = self._preprocess(roi, use_fp16=False)
                output = self._run_inference(tensor, use_fp16=False)
                
                # Check stability again
                if not self._check_numerical_stability(output):
                    raise NumericalInstabilityError(
                        "Numerical instability detected even with FP32 fallback"
                    )
            else:
                raise NumericalInstabilityError(
                    "Numerical instability (NaN/Inf) detected in output. "
                    "Enable fp32_fallback to automatically retry with FP32."
                )
        
        # Postprocess
        result = self._postprocess(output)
        
        # Convert back to grayscale if input was grayscale
        if is_grayscale:
            import cv2
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        
        return result
    
    def get_memory_usage(self) -> dict:
        """Get current GPU memory usage information.
        
        Returns:
            Dictionary with memory usage statistics:
            - allocated: Currently allocated memory in bytes
            - reserved: Reserved memory in bytes
            - max_allocated: Peak allocated memory in bytes
        """
        if TORCH_AVAILABLE and torch.cuda.is_available() and self.device == 'cuda':
            return {
                'allocated': torch.cuda.memory_allocated(),
                'reserved': torch.cuda.memory_reserved(),
                'max_allocated': torch.cuda.max_memory_allocated()
            }
        
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
        if TORCH_AVAILABLE and torch.cuda.is_available() and self.device == 'cuda':
            torch.cuda.empty_cache()
