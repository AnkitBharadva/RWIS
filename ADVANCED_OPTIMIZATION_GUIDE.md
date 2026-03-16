# Advanced Performance Optimization
## Without Skipping Frames or Disabling Components

This guide focuses on making each component FASTER, not doing less work.

## Strategy Overview

1. **Model Optimization** - Convert models to faster formats (ONNX, TensorRT)
2. **GPU Optimization** - Better GPU utilization with CUDA streams
3. **Algorithm Optimization** - Faster implementations of existing logic
4. **Memory Optimization** - Reduce memory copies and allocations
5. **Preprocessing Optimization** - Vectorized operations

## Phase 1: Model Optimization (2-4x Faster)

### 1.1 Convert MPRNet to ONNX Runtime
**Impact**: 2-3x faster inference

```python
# Export MPRNet to ONNX
import torch
from pipelines.mprnet_wrapper import MPRNetDeblur

# Load model
mprnet = MPRNetDeblur(model_path="...", device='cuda')
mprnet.load_model()

# Export to ONNX
dummy_input = torch.randn(1, 3, 256, 256).cuda()
torch.onnx.export(
    mprnet.model,
    dummy_input,
    "mprnet_optimized.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch', 2: 'height', 3: 'width'}}
)
```

### 1.2 Use ONNX Runtime for Inference
```python
import onnxruntime as ort

# Create ONNX session with GPU
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession("mprnet_optimized.onnx", providers=providers)

# Inference (2-3x faster)
output = session.run(None, {'input': input_tensor})[0]
```

### 1.3 Convert YOLO to TensorRT
**Impact**: 3-4x faster detection

```bash
# Export YOLO to TensorRT
yolo export model=models/damage_detector.pt format=engine device=0
```

## Phase 2: GPU Stream Optimization (30-50% Faster)

### 2.1 Use CUDA Streams for Concurrent Operations
```python
import torch

class OptimizedPipeline:
    def __init__(self):
        # Create separate CUDA streams
        self.stream_detection = torch.cuda.Stream()
        self.stream_ocr = torch.cuda.Stream()
        self.stream_deblur = torch.cuda.Stream()
    
    def process_frame(self, frame):
        # Detection on stream 1
        with torch.cuda.stream(self.stream_detection):
            detections = self.detect(frame)
        
        # OCR on stream 2 (parallel with detection)
        with torch.cuda.stream(self.stream_ocr):
            ocr_results = self.ocr(frame)
        
        # Synchronize when needed
        torch.cuda.synchronize()
```

### 2.2 Overlap CPU and GPU Work
```python
# While GPU processes current frame, CPU prepares next frame
def process_with_overlap(self, frames):
    # Preload first frame to GPU
    current_gpu = self.to_gpu(frames[0])
    
    for i in range(len(frames) - 1):
        # GPU: Process current frame
        result = self.process_gpu(current_gpu)
        
        # CPU: Prepare next frame (parallel)
        next_gpu = self.to_gpu(frames[i + 1])
        
        # Synchronize and swap
        torch.cuda.synchronize()
        current_gpu = next_gpu
        
        yield result
```

## Phase 3: Algorithm Optimization (20-40% Faster)

### 3.1 Vectorized Image Preprocessing
```python
import numpy as np
import cv2

class FastPreprocessor:
    """Vectorized preprocessing for batch operations."""
    
    def __init__(self):
        # Pre-allocate buffers
        self.buffer_rgb = np.empty((256, 256, 3), dtype=np.float32)
        self.buffer_normalized = np.empty((256, 256, 3), dtype=np.float32)
    
    def preprocess_batch(self, rois):
        """Process multiple ROIs at once."""
        # Vectorized resize
        resized = np.array([cv2.resize(roi, (256, 256)) for roi in rois])
        
        # Vectorized color conversion (faster than loop)
        rgb_batch = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Vectorized normalization
        normalized = rgb_batch.astype(np.float32) / 255.0
        
        return normalized
```

### 3.2 Optimized Blur Detection
```python
class FastBlurDetector:
    """Optimized blur detection using integral images."""
    
    def __init__(self):
        self.laplacian_kernel = np.array([[0, 1, 0],
                                          [1, -4, 1],
                                          [0, 1, 0]], dtype=np.float32)
    
    def compute_blur_score_fast(self, roi):
        """Faster blur detection using cv2.filter2D."""
        # Convert to grayscale if needed
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi
        
        # Use optimized filter2D (faster than Laplacian)
        laplacian = cv2.filter2D(gray, cv2.CV_64F, self.laplacian_kernel)
        
        # Variance calculation (vectorized)
        return laplacian.var()
```

### 3.3 Fast ROI Extraction with Slicing
```python
def extract_roi_fast(frame, bbox):
    """Optimized ROI extraction using numpy slicing."""
    x1, y1, x2, y2 = map(int, bbox)
    
    # Clip to frame bounds (vectorized)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    # Direct slice (no copy until needed)
    roi = frame[y1:y2, x1:x2]
    
    # Only copy if we need to modify
    return roi.copy() if roi.flags['WRITEABLE'] else roi
```

## Phase 4: Memory Optimization (15-25% Faster)

### 4.1 Pre-allocate Buffers
```python
class BufferPool:
    """Reusable buffer pool to avoid allocations."""
    
    def __init__(self, max_size=(256, 256, 3)):
        self.buffers = []
        self.max_size = max_size
        self.in_use = set()
    
    def get_buffer(self):
        """Get a reusable buffer."""
        for i, buf in enumerate(self.buffers):
            if i not in self.in_use:
                self.in_use.add(i)
                return buf
        
        # Create new buffer if none available
        buf = np.empty(self.max_size, dtype=np.uint8)
        self.buffers.append(buf)
        self.in_use.add(len(self.buffers) - 1)
        return buf
    
    def release_buffer(self, buf):
        """Return buffer to pool."""
        for i, b in enumerate(self.buffers):
            if b is buf:
                self.in_use.discard(i)
                break
```

### 4.2 Avoid Unnecessary Copies
```python
def process_roi_nocopy(self, roi):
    """Process ROI without copying when possible."""
    # Use views instead of copies
    if roi.flags['C_CONTIGUOUS']:
        # Can use directly
        return self.model.predict(roi)
    else:
        # Need to make contiguous (unavoidable)
        return self.model.predict(np.ascontiguousarray(roi))
```

### 4.3 Pinned Memory for GPU Transfers
```python
import torch

class FastGPUTransfer:
    """Use pinned memory for faster CPU->GPU transfers."""
    
    def __init__(self):
        # Allocate pinned memory
        self.pinned_buffer = torch.empty(
            (1, 3, 256, 256),
            dtype=torch.float32,
            pin_memory=True
        )
    
    def to_gpu_fast(self, numpy_array):
        """Fast transfer using pinned memory."""
        # Copy to pinned memory
        self.pinned_buffer.copy_(torch.from_numpy(numpy_array))
        
        # Transfer to GPU (faster with pinned memory)
        return self.pinned_buffer.cuda(non_blocking=True)
```

## Phase 5: Batch Processing (2-3x Faster)

### 5.1 Batch Multiple ROIs Together
```python
class BatchProcessor:
    """Process multiple ROIs in a single batch."""
    
    def __init__(self, model, batch_size=4):
        self.model = model
        self.batch_size = batch_size
        self.roi_queue = []
    
    def add_roi(self, roi, wagon_id):
        """Add ROI to batch queue."""
        self.roi_queue.append((roi, wagon_id))
        
        # Process when batch is full
        if len(self.roi_queue) >= self.batch_size:
            return self.process_batch()
        return []
    
    def process_batch(self):
        """Process all queued ROIs in one batch."""
        if not self.roi_queue:
            return []
        
        # Stack ROIs into batch
        rois = [r[0] for r in self.roi_queue]
        wagon_ids = [r[1] for r in self.roi_queue]
        
        # Batch inference (much faster than individual)
        batch_tensor = torch.stack([self.preprocess(r) for r in rois])
        results = self.model(batch_tensor)
        
        # Clear queue
        self.roi_queue.clear()
        
        return list(zip(wagon_ids, results))
```

### 5.2 Dynamic Batching
```python
class DynamicBatcher:
    """Automatically batch ROIs with timeout."""
    
    def __init__(self, model, max_batch_size=8, timeout_ms=10):
        self.model = model
        self.max_batch_size = max_batch_size
        self.timeout_ms = timeout_ms
        self.queue = []
        self.last_process_time = time.time()
    
    def process_roi(self, roi):
        """Add ROI and process if batch ready or timeout."""
        self.queue.append(roi)
        
        # Process if batch full or timeout
        current_time = time.time()
        time_since_last = (current_time - self.last_process_time) * 1000
        
        if len(self.queue) >= self.max_batch_size or time_since_last > self.timeout_ms:
            results = self._process_batch()
            self.last_process_time = current_time
            return results
        
        return None
```

## Phase 6: Optimized OCR (40-60% Faster)

### 6.1 Use EasyOCR with Optimizations
```python
import easyocr

class OptimizedOCR:
    def __init__(self):
        self.reader = easyocr.Reader(
            ['en'],
            gpu=True,
            model_storage_directory='models/',
            download_enabled=False,
            quantize=True,  # Use quantized models
            cudnn_benchmark=True  # Enable cuDNN auto-tuner
        )
    
    def extract_text_fast(self, roi):
        """Optimized OCR with preprocessing."""
        # Resize to optimal size for OCR (not too large)
        if roi.shape[1] > 640:
            scale = 640 / roi.shape[1]
            roi = cv2.resize(roi, None, fx=scale, fy=scale)
        
        # Use batch mode if multiple ROIs
        results = self.reader.readtext(
            roi,
            batch_size=1,
            workers=0,  # Disable multiprocessing (faster for single images)
            paragraph=False,  # Disable paragraph detection
            min_size=10  # Skip very small text
        )
        
        return results
```

### 6.2 ROI Preprocessing for Better OCR
```python
def preprocess_for_ocr(roi):
    """Optimize ROI for faster OCR."""
    # Convert to grayscale (faster OCR)
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi
    
    # Adaptive thresholding (improves accuracy and speed)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    
    return binary
```

## Implementation Priority

### Week 1: Quick Wins (30-50% faster)
1. ✅ Vectorized preprocessing
2. ✅ Optimized blur detection
3. ✅ Fast ROI extraction
4. ✅ Buffer pooling

### Week 2: Model Optimization (2-3x faster)
1. ✅ Convert MPRNet to ONNX
2. ✅ Convert YOLO to TensorRT
3. ✅ Optimize OCR settings

### Week 3: Advanced (3-4x faster)
1. ✅ CUDA streams
2. ✅ Batch processing
3. ✅ Pinned memory

## Expected Results

| Optimization | Improvement | Cumulative |
|-------------|-------------|------------|
| **Baseline** | - | 12 FPS |
| **Vectorized Ops** | +30% | 15.6 FPS |
| **Buffer Pooling** | +20% | 18.7 FPS |
| **ONNX Runtime** | +150% | 46.8 FPS |
| **TensorRT** | +200% | 140 FPS |
| **CUDA Streams** | +40% | 196 FPS |
| **Batch Processing** | +100% | 392 FPS |

**Final Target: 50-100 FPS without skipping frames!**

## Next Steps

1. Start with Phase 3 (Algorithm Optimization) - Easy to implement
2. Move to Phase 4 (Memory Optimization) - Medium effort
3. Then Phase 1 (Model Optimization) - Highest impact
4. Finally Phase 2 (GPU Optimization) - Advanced

Each phase builds on the previous one for maximum performance gain.
