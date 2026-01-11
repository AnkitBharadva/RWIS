# Design Document: Railway Wagon Inspection Pipeline

## Overview

This document describes the technical design for an industrial AI video processing pipeline that performs railway wagon counting, damage detection, and OCR logging. The system processes video streams in real-time on a Windows 11 machine with an NVIDIA RTX 3050 GPU (6 GB VRAM).

The pipeline follows a strict multi-stage architecture where:
- Stage 1: Blur assessment determines processing path
- Stage 2: Primary YOLO model detects and tracks wagons
- Stage 3: Secondary YOLO model detects damage on wagon ROIs
- Stage 4: OCR extracts wagon identification numbers
- Stage 5: All results are logged to CSV/JSON

Key design principles:
- YOLO models receive only RAW or CLAHE-enhanced frames (never deblurred)
- Deblurring (NAFNet) is applied only to small OCR ROIs
- GPU memory is carefully managed to stay within 6 GB VRAM
- Counting uses line-crossing logic with tracking to prevent duplicates

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MAIN PIPELINE CONTROLLER                          │
│                              (main.py)                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FRAME CAPTURE & BLUR DETECTION                      │
│                                                                             │
│  ┌──────────────┐    ┌──────────────────┐    ┌─────────────────────────┐   │
│  │ Video Source │───▶│ Frame Capture    │───▶│ Blur Detector           │   │
│  │ (file/stream)│    │ (OpenCV)         │    │ (Laplacian Variance)    │   │
│  └──────────────┘    └──────────────────┘    └─────────────────────────┘   │
│                                                        │                    │
│                                              blur_score + decision          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
              blur < T1      T1 ≤ blur < T2    blur ≥ T2
           (skip deblur)    (ROI deblur)    (no deblur)
                    │               │               │
                    └───────────────┼───────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      STAGE 2: PRIMARY DETECTION (YOLO #1)                   │
│                                                                             │
│  ┌──────────────┐    ┌──────────────────┐    ┌─────────────────────────┐   │
│  │ RAW Frame    │───▶│ CLAHE Enhancement│───▶│ Wagon Detector          │   │
│  │ (optional)   │    │ (L-channel only) │    │ (YOLOv11n)              │   │
│  └──────────────┘    └──────────────────┘    └─────────────────────────┘   │
│                                                        │                    │
│                                              wagon detections + ROIs        │
│                                                        │                    │
│                                                        ▼                    │
│                                              ┌─────────────────────────┐   │
│                                              │ Tracker (ByteTrack)     │   │
│                                              │ - Assign wagon IDs      │   │
│                                              │ - Line-crossing count   │   │
│                                              └─────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                          tracked wagons + ROIs
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
┌─────────────────────────────────┐  ┌─────────────────────────────────────┐
│ STAGE 3: DAMAGE DETECTION       │  │ STAGE 4: OCR PIPELINE               │
│                                 │  │                                     │
│ ┌─────────────────────────────┐ │  │ ┌─────────────────────────────────┐ │
│ │ Damage Detector (YOLOv11n)  │ │  │ │ OCR ROI Extraction              │ │
│ │ - Door damage               │ │  │ └─────────────────────────────────┘ │
│ │ - Floor damage              │ │  │                 │                   │
│ │ - Structural cracks         │ │  │                 ▼                   │
│ │ - Deformation               │ │  │ ┌─────────────────────────────────┐ │
│ └─────────────────────────────┘ │  │ │ Low-light Enhancement           │ │
│              │                  │  │ │ (Adaptive Gamma)                │ │
│              ▼                  │  │ └─────────────────────────────────┘ │
│ ┌─────────────────────────────┐ │  │                 │                   │
│ │ Output:                     │ │  │                 ▼                   │
│ │ - damage_class              │ │  │ ┌─────────────────────────────────┐ │
│ │ - bbox                      │ │  │ │ NAFNet Deblur (if needed)       │ │
│ │ - confidence                │ │  │ │ (ROI only, blur_score < T2)     │ │
│ │ - wagon_id                  │ │  │ └─────────────────────────────────┘ │
│ └─────────────────────────────┘ │  │                 │                   │
└─────────────────────────────────┘  │                 ▼                   │
                    │                │ ┌─────────────────────────────────┐ │
                    │                │ │ PaddleOCR Inference             │ │
                    │                │ │ (GPU accelerated)               │ │
                    │                │ └─────────────────────────────────┘ │
                    │                │                 │                   │
                    │                │    text + confidence               │
                    │                └─────────────────────────────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          STAGE 5: LOGGING & OUTPUT                          │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ Logger                                                                │  │
│  │ - CSV output (logs.csv)                                              │  │
│  │ - JSON output (logs.json)                                            │  │
│  │ - Debug frames (optional)                                            │  │
│  │                                                                       │  │
│  │ Fields: timestamp, wagon_id, count_index, blur_score, damage_detected│  │
│  │         damage_classes, ocr_text, ocr_confidence, frame_index        │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Components and Interfaces

### 1. Configuration Module (config.py)

```python
@dataclass
class PipelineConfig:
    # Video input
    video_source: str
    
    # Blur thresholds
    blur_threshold_t1: float  # Below this: skip deblur
    blur_threshold_t2: float  # Above this: no deblur (too blurry)
    
    # Model paths
    wagon_model_path: str
    damage_model_path: str
    nafnet_model_path: str
    
    # Detection settings
    wagon_confidence_threshold: float
    damage_confidence_threshold: float
    
    # Counting line position (y-coordinate as fraction of frame height)
    counting_line_position: float
    
    # OCR settings
    ocr_gpu_enabled: bool
    ocr_language: str
    
    # Performance settings
    max_batch_size: int
    enable_threading: bool
    
    # Output settings
    output_dir: str
    enable_debug_frames: bool
    log_format: List[str]  # ['csv', 'json']
```

### 2. Blur Detector Interface (pipelines/blur_detector.py)

```python
class BlurDetector:
    def __init__(self, t1: float, t2: float):
        """Initialize with configurable thresholds."""
        
    def compute_blur_score(self, frame: np.ndarray) -> float:
        """Compute Laplacian variance blur score."""
        
    def get_blur_decision(self, blur_score: float) -> BlurDecision:
        """
        Returns:
            BlurDecision.SKIP_DEBLUR if blur_score < T1
            BlurDecision.ROI_DEBLUR if T1 <= blur_score < T2
            BlurDecision.NO_DEBLUR if blur_score >= T2
        """
```

### 3. Wagon Detector Interface (pipelines/wagon_detector.py)

```python
class WagonDetector:
    def __init__(self, model_path: str, confidence_threshold: float):
        """Load YOLOv11n wagon detection model."""
        
    def detect(self, frame: np.ndarray) -> List[WagonDetection]:
        """
        Detect wagons in frame.
        Input: RAW or CLAHE-enhanced frame only.
        Returns: List of WagonDetection with bbox, confidence, class_id
        """
        
    def extract_roi(self, frame: np.ndarray, detection: WagonDetection) -> np.ndarray:
        """Extract wagon ROI from frame."""
```

### 4. Tracker Interface (tracking/tracker.py)

```python
class WagonTracker:
    def __init__(self, counting_line_y: float):
        """Initialize ByteTrack tracker with counting line."""
        
    def update(self, detections: List[WagonDetection], frame_shape: Tuple) -> List[TrackedWagon]:
        """
        Update tracker with new detections.
        Returns: List of TrackedWagon with track_id, bbox, crossed_line flag
        """
        
    def get_wagon_count(self) -> int:
        """Return total count of wagons that crossed the line."""
        
    def has_crossed_line(self, track_id: int) -> bool:
        """Check if wagon has already been counted."""
```

### 5. Damage Detector Interface (pipelines/damage_detector.py)

```python
class DamageDetector:
    def __init__(self, model_path: str, confidence_threshold: float):
        """Load YOLOv11n damage detection model."""
        
    def detect(self, wagon_roi: np.ndarray, wagon_id: int) -> List[DamageDetection]:
        """
        Detect damage in wagon ROI.
        Input: RAW or CLAHE-enhanced ROI only.
        Returns: List of DamageDetection with damage_class, bbox, confidence, wagon_id
        """
```

### 6. NAFNet Wrapper Interface (pipelines/nafnet_wrapper.py)

```python
class NAFNetDeblur:
    def __init__(self, model_path: str, device: str = 'cuda'):
        """Load NAFNet deblur model."""
        
    def deblur_roi(self, roi: np.ndarray) -> np.ndarray:
        """
        Deblur a small ROI image.
        CRITICAL: Only call this on OCR ROI, never on full frames.
        """
```

### 7. OCR Pipeline Interface (pipelines/ocr_pipeline.py)

```python
class OCRPipeline:
    def __init__(self, gpu_enabled: bool, language: str):
        """Initialize PaddleOCR with GPU support."""
        
    def extract_text(
        self, 
        wagon_roi: np.ndarray, 
        blur_decision: BlurDecision,
        nafnet: NAFNetDeblur
    ) -> OCRResult:
        """
        Extract text from wagon ROI.
        Processing order:
        1. Extract OCR region from wagon ROI
        2. Apply low-light enhancement if needed
        3. Apply NAFNet deblur if blur_decision == ROI_DEBLUR
        4. Run PaddleOCR
        Returns: OCRResult with text, confidence
        """
```

### 8. CLAHE Enhancement (utils/clahe.py)

```python
class CLAHEEnhancer:
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple = (8, 8)):
        """Initialize CLAHE with parameters."""
        
    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE to L-channel only in LAB color space.
        Returns: Enhanced frame in BGR format.
        """
```

### 9. Logger Interface (utils/logger.py)

```python
class InspectionLogger:
    def __init__(self, output_dir: str, formats: List[str], enable_debug: bool):
        """Initialize logger with output configuration."""
        
    def log_wagon(self, record: WagonRecord) -> None:
        """Log a complete wagon inspection record."""
        
    def save_debug_frame(self, frame: np.ndarray, frame_index: int, annotations: List) -> None:
        """Save annotated debug frame if enabled."""
        
    def flush(self) -> None:
        """Flush all pending writes to disk."""
```

## Data Models

```python
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

class BlurDecision(Enum):
    SKIP_DEBLUR = "skip_deblur"      # blur_score < T1
    ROI_DEBLUR = "roi_deblur"        # T1 <= blur_score < T2
    NO_DEBLUR = "no_deblur"          # blur_score >= T2

class DamageClass(Enum):
    DOOR_DAMAGE = "door_damage"
    FLOOR_DAMAGE = "floor_damage"
    STRUCTURAL_CRACK = "structural_crack"
    DEFORMATION = "deformation"

@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        return self.y2 - self.y1
    
    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

@dataclass
class WagonDetection:
    bbox: BoundingBox
    confidence: float
    class_id: int

@dataclass
class TrackedWagon:
    track_id: int
    bbox: BoundingBox
    confidence: float
    crossed_line: bool
    count_index: Optional[int]  # Assigned when wagon crosses line

@dataclass
class DamageDetection:
    damage_class: DamageClass
    bbox: BoundingBox
    confidence: float
    wagon_id: int

@dataclass
class OCRResult:
    text: str
    confidence: float
    bbox: Optional[BoundingBox]

@dataclass
class WagonRecord:
    timestamp: str
    wagon_id: int
    count_index: int
    blur_score: float
    frame_index: int
    damage_detected: bool
    damage_classes: List[str]
    damage_bboxes: List[BoundingBox]
    ocr_text: str
    ocr_confidence: float
```



## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Blur Decision Logic Consistency

*For any* frame with a computed blur_score and configured thresholds T1 and T2 (where T1 < T2):
- If blur_score < T1, the blur decision SHALL be SKIP_DEBLUR
- If T1 <= blur_score < T2, the blur decision SHALL be ROI_DEBLUR
- If blur_score >= T2, the blur decision SHALL be NO_DEBLUR

**Validates: Requirements 1.3, 1.4, 1.5**

### Property 2: Blur Score Computation Determinism

*For any* input frame, computing the blur score twice SHALL produce identical results. The Laplacian variance computation must be deterministic.

**Validates: Requirements 1.2**

### Property 3: YOLO Input Integrity

*For any* frame passed to the Wagon_Detector or Damage_Detector, the frame SHALL NOT have been processed by NAFNet deblurring. The input must be either RAW or CLAHE-enhanced only.

**Validates: Requirements 2.2, 2.3, 3.6, 8.3**

### Property 4: CLAHE L-Channel Isolation

*For any* frame processed by CLAHE enhancement, converting to LAB color space before and after enhancement SHALL show:
- The L channel is modified
- The A and B channels remain unchanged (within floating-point tolerance)

**Validates: Requirements 2.4, 8.5**

### Property 5: Wagon Counting Accuracy

*For any* tracked wagon that crosses the counting line:
- The wagon count SHALL increment by exactly 1 when the wagon first crosses
- Subsequent frames showing the same wagon SHALL NOT increment the count again
- The total count SHALL equal the number of unique wagons that crossed the line

**Validates: Requirements 2.6, 2.7**

### Property 6: Unique Wagon ID Assignment

*For any* sequence of wagon detections processed by the tracker, each tracked wagon SHALL have a unique track_id. No two wagons in the same tracking session SHALL share the same ID.

**Validates: Requirements 2.5**

### Property 7: Valid ROI Generation

*For any* wagon detection, the generated ROI coordinates SHALL:
- Have x1 < x2 and y1 < y2 (valid bounding box)
- Be within frame boundaries (0 <= x1, x2 <= frame_width; 0 <= y1, y2 <= frame_height)
- Have non-zero width and height

**Validates: Requirements 2.8**

### Property 8: Damage Detection Output Completeness

*For any* damage detection result, the output SHALL contain:
- A valid damage_class from the defined enum
- A valid bounding box with positive dimensions
- A confidence score in range [0.0, 1.0]
- A wagon_id matching the input wagon's track_id

**Validates: Requirements 3.4, 3.5**

### Property 9: Damage Classes Within Expected Set

*For any* damage detection output, the damage_class SHALL be one of: DOOR_DAMAGE, FLOOR_DAMAGE, STRUCTURAL_CRACK, or DEFORMATION. No other damage classes are valid.

**Validates: Requirements 3.3**

### Property 10: Deblur ROI-Only Constraint

*For any* invocation of NAFNet deblurring:
- The input image dimensions SHALL be smaller than the full frame dimensions
- The input SHALL be an OCR ROI region only
- Full-frame deblurring SHALL never occur

**Validates: Requirements 4.4, 4.7, 8.1, 8.6**

### Property 11: OCR Processes ROI Only

*For any* OCR inference call, the input image SHALL be a cropped ROI region, not a full frame. The input dimensions SHALL be significantly smaller than the source frame dimensions.

**Validates: Requirements 4.2**

### Property 12: OCR Output Completeness

*For any* OCR result, the output SHALL contain:
- A text field (may be empty string if no text detected)
- A confidence score in range [0.0, 1.0]

**Validates: Requirements 4.6**

### Property 13: Log Record Completeness

*For any* wagon record written to the log, the record SHALL contain all required fields:
- timestamp (non-empty string)
- wagon_id (positive integer)
- count_index (positive integer)
- blur_score (float)
- frame_index (non-negative integer)
- damage_detected (boolean)
- damage_classes (list, may be empty)
- ocr_text (string, may be empty)
- ocr_confidence (float in [0.0, 1.0])

**Validates: Requirements 5.1, 5.2, 5.3, 5.4**

### Property 14: Dual Format Logging Consistency

*For any* wagon record logged, both CSV and JSON outputs SHALL contain the same data. Parsing the CSV row and the corresponding JSON entry SHALL produce equivalent WagonRecord objects.

**Validates: Requirements 5.5**

### Property 15: Configuration Loading

*For any* configuration file, all threshold values (T1, T2, confidence thresholds, counting line position) SHALL be loaded and applied to the pipeline. Changing config values SHALL change pipeline behavior accordingly.

**Validates: Requirements 7.1**

### Property 16: Debug Frame Toggle

*For any* pipeline run:
- If enable_debug_frames is True, debug frames SHALL be saved to outputs/debug_frames/
- If enable_debug_frames is False, no debug frames SHALL be created

**Validates: Requirements 7.5**

## Error Handling

### Model Loading Errors

| Error Condition | Handling Strategy |
|----------------|-------------------|
| Wagon model file missing | Raise `ModelNotFoundError` with clear message indicating expected path |
| Damage model file missing | Raise `ModelNotFoundError` with clear message indicating expected path |
| NAFNet model file missing | Raise `ModelNotFoundError` with clear message indicating expected path |
| Model file corrupted | Raise `ModelLoadError` with details about the corruption |
| CUDA not available | Fall back to CPU with warning, or raise error if GPU required |

### Video Input Errors

| Error Condition | Handling Strategy |
|----------------|-------------------|
| Video file not found | Raise `VideoSourceError` with path information |
| Stream URL unreachable | Retry with exponential backoff, then raise `StreamConnectionError` |
| Corrupted video frame | Skip frame, log warning, continue processing |
| End of video stream | Gracefully terminate pipeline, flush logs |

### Runtime Errors

| Error Condition | Handling Strategy |
|----------------|-------------------|
| GPU out of memory | Queue operations, reduce batch size, log warning |
| Invalid ROI coordinates | Skip ROI processing, log warning with frame index |
| OCR timeout | Return empty result with zero confidence, log warning |
| Tracker state corruption | Reset tracker, log error, continue with new tracking session |

### Configuration Errors

| Error Condition | Handling Strategy |
|----------------|-------------------|
| Invalid threshold values (T1 >= T2) | Raise `ConfigurationError` with explanation |
| Missing required config fields | Raise `ConfigurationError` listing missing fields |
| Invalid model paths | Raise `ConfigurationError` with path validation details |

## Testing Strategy

### Unit Tests

Unit tests verify specific examples and edge cases:

1. **Blur Detection Tests**
   - Test Laplacian variance computation on known sharp/blurry images
   - Test threshold boundary conditions (blur_score exactly at T1, T2)
   - Test with synthetic images of known blur levels

2. **CLAHE Enhancement Tests**
   - Test L-channel modification verification
   - Test A/B channel preservation
   - Test with various lighting conditions

3. **ROI Extraction Tests**
   - Test boundary clipping when detection near frame edge
   - Test valid ROI generation for various detection sizes
   - Test empty detection list handling

4. **Tracker Tests**
   - Test ID assignment for new detections
   - Test line-crossing detection at various positions
   - Test tracker reset behavior

5. **Logger Tests**
   - Test CSV format correctness
   - Test JSON format correctness
   - Test file creation and append behavior

### Property-Based Tests

Property-based tests verify universal properties across many generated inputs. Each test runs minimum 100 iterations.

**Testing Framework**: pytest with hypothesis library

1. **Property Test: Blur Decision Logic**
   - Generate random blur_score values and T1/T2 thresholds
   - Verify decision matches expected logic for all combinations
   - **Tag: Feature: railway-wagon-inspection, Property 1: Blur Decision Logic Consistency**

2. **Property Test: Blur Score Determinism**
   - Generate random frames
   - Verify computing blur_score twice produces identical results
   - **Tag: Feature: railway-wagon-inspection, Property 2: Blur Score Computation Determinism**

3. **Property Test: CLAHE L-Channel Isolation**
   - Generate random BGR images
   - Apply CLAHE and verify A/B channels unchanged
   - **Tag: Feature: railway-wagon-inspection, Property 4: CLAHE L-Channel Isolation**

4. **Property Test: Wagon Counting Accuracy**
   - Generate sequences of wagon positions crossing a line
   - Verify count equals unique wagons that crossed
   - **Tag: Feature: railway-wagon-inspection, Property 5: Wagon Counting Accuracy**

5. **Property Test: Unique Wagon IDs**
   - Generate random detection sequences
   - Verify all assigned IDs are unique
   - **Tag: Feature: railway-wagon-inspection, Property 6: Unique Wagon ID Assignment**

6. **Property Test: Valid ROI Generation**
   - Generate random detections within frame bounds
   - Verify all ROIs have valid coordinates
   - **Tag: Feature: railway-wagon-inspection, Property 7: Valid ROI Generation**

7. **Property Test: Damage Detection Output Completeness**
   - Generate random damage detections
   - Verify all required fields present and valid
   - **Tag: Feature: railway-wagon-inspection, Property 8: Damage Detection Output Completeness**

8. **Property Test: Log Record Completeness**
   - Generate random wagon records
   - Verify all required fields present in logged output
   - **Tag: Feature: railway-wagon-inspection, Property 13: Log Record Completeness**

9. **Property Test: Dual Format Logging Consistency**
   - Generate random wagon records
   - Log to both formats and verify equivalence
   - **Tag: Feature: railway-wagon-inspection, Property 14: Dual Format Logging Consistency**

10. **Property Test: Configuration Loading**
    - Generate random valid configurations
    - Verify all values are correctly loaded and applied
    - **Tag: Feature: railway-wagon-inspection, Property 15: Configuration Loading**

### Integration Tests

1. **End-to-End Pipeline Test**
   - Process a short test video with known wagon count
   - Verify final count matches expected
   - Verify log files are created with correct format

2. **GPU Memory Test**
   - Process video while monitoring GPU memory
   - Verify memory stays within 6 GB limit

3. **Performance Test**
   - Measure FPS on test video
   - Verify near-real-time performance achieved
