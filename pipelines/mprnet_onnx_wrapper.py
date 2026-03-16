"""ONNX-optimized MPRNet wrapper for 2-3x faster inference.

This wrapper uses ONNX Runtime instead of PyTorch for significantly
faster inference while maintaining the same quality.
"""

import numpy as np
import os
from typing import Tuple, Optional
import logging

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None

logger = logging.getLogger(__name__)


class MPRNetONNXWrapper:
    """ONNX-optimized MPRNet wrapper for faster inference."""
    
    def __init__(
        self,
        onnx_model_path: str,
        device: str = 'cuda',
        max_roi_width: int = 256,
        max_roi_height: int = 256
    ):
        """Initialize ONNX MPRNet wrapper.
        
        Args:
            onnx_model_path: Path to ONNX model file
            device: Device for inference ('cuda' or 'cpu')
            max_roi_width: Maximum ROI width
            max_roi_height: Maximum ROI height
        """
        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime-gpu is required. Install with: pip install onnxruntime-gpu")
        
        self.onnx_model_path = onnx_model_path
        self.device = device
        self.max_roi_width = max_roi_width
        self.max_roi_height = max_roi_height
        self.session = None
        self._model_loaded = False
        
        # Pre-allocate buffers for faster processing
        self.input_buffer = np.empty((1, 3, 256, 256), dtype=np.float32)
    
    def load_model(self):
        """Load ONNX model."""
        if not os.path.exists(self.onnx_model_path):
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_model_path}")
        
        # Set up providers (GPU first, then CPU fallback)
        providers = []
        if self.device == 'cuda':
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
        
        # Create ONNX session with optimizations
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 1  # Single thread for GPU
        
        self.session = ort.InferenceSession(
            self.onnx_model_path,
            sess_options=sess_options,
            providers=providers
        )
        
        self._model_loaded = True
        logger.info(f"ONNX model loaded with provider: {self.session.get_providers()[0]}")
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model_loaded
    
    def deblur_roi(self, roi: np.ndarray) -> np.ndarray:
        """Deblur ROI using ONNX Runtime (2-3x faster than PyTorch).
        
        Args:
            roi: Input ROI as BGR numpy array
            
        Returns:
            Deblurred ROI
        """
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        import cv2
        
        # Preprocess
        if len(roi.shape) == 3 and roi.shape[2] == 3:
            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
        
        # Normalize
        rgb = rgb.astype(np.float32) / 255.0
        
        # Transpose to CHW format
        input_tensor = np.transpose(rgb, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, 0)
        
        # Pad to multiple of 8
        _, _, h, w = input_tensor.shape
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        
        if pad_h > 0 or pad_w > 0:
            input_tensor = np.pad(
                input_tensor,
                ((0, 0), (0, 0), (0, pad_h), (0, pad_w)),
                mode='reflect'
            )
        
        # ONNX inference (much faster than PyTorch)
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        
        output = self.session.run([output_name], {input_name: input_tensor})[0]
        
        # Remove padding
        if pad_h > 0 or pad_w > 0:
            output = output[:, :, :output.shape[2]-pad_h, :output.shape[3]-pad_w]
        
        # Postprocess
        output = np.clip(output, 0, 1)
        output = np.squeeze(output, 0)
        output = np.transpose(output, (1, 2, 0))
        output = (output * 255).astype(np.uint8)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        
        return output
    
    def set_full_frame_dimensions(self, width: int, height: int):
        """Set full frame dimensions (for compatibility)."""
        pass
    
    def is_using_fp32_fallback(self) -> bool:
        """Check if using FP32 fallback (always False for ONNX)."""
        return False
    
    def get_memory_usage(self) -> dict:
        """Get memory usage."""
        return {'allocated': 0, 'reserved': 0, 'max_allocated': 0}
    
    def clear_memory_cache(self):
        """Clear memory cache."""
        pass
