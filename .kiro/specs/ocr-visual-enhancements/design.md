# Design Document: OCR Visual Enhancements

## Overview

This design document describes the implementation of enhanced OCR visualization features for the Railway Wagon Inspection System. The enhancements include OCR bounding box visualization, text overlays on video frames, frame saving with metadata, OCR frame interval control, and processing status indicators in the main dashboard.

## Architecture

The implementation extends the existing dashboard architecture with new visualization and control components:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MissionControlDashboard                            │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Enhanced Metrics Row                              │   │
│  │  FPS | Latency | Objects | Wagons | Damage | Illum | Deblur | OCR   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Dual Video Display                                │   │
│  │  ┌─────────────────────┐    ┌─────────────────────┐                 │   │
│  │  │    Raw Input        │    │  Processed Output   │                 │   │
│  │  │                     │    │  + OCR Bounding Boxes│                 │   │
│  │  │                     │    │  + Text Overlays     │                 │   │
│  │  └─────────────────────┘    └─────────────────────┘                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Detection Log / OCR Log                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Components and Interfaces

### 1. OCRVisualization Class

New class responsible for drawing OCR bounding boxes and text overlays on frames.

```python
class OCRVisualization:
    """Handles OCR visualization on video frames."""
    
    # Color constants (BGR format)
    OCR_BOX_COLOR_NORMAL = (255, 255, 0)      # Cyan
    OCR_BOX_COLOR_DEBLURRED = (255, 200, 0)   # Bright cyan
    OCR_TEXT_COLOR_NORMAL = (255, 255, 255)   # White
    OCR_TEXT_COLOR_WARNING = (0, 165, 255)    # Orange (low confidence)
    CONFIDENCE_WARNING_THRESHOLD = 0.5
    
    def draw_ocr_boxes(
        self,
        frame: np.ndarray,
        ocr_results: List[OCRDetection],
        wagon_bbox: BoundingBox,
        deblur_applied: bool = False
    ) -> np.ndarray:
        """Draw OCR bounding boxes on frame with ROI-to-frame coordinate adjustment."""
        pass
    
    def draw_text_overlay(
        self,
        frame: np.ndarray,
        text: str,
        confidence: float,
        position: Tuple[int, int],
        deblur_applied: bool = False
    ) -> np.ndarray:
        """Draw text overlay with semi-transparent background."""
        pass
    
    def adjust_coordinates(
        self,
        ocr_bbox: BoundingBox,
        wagon_bbox: BoundingBox
    ) -> BoundingBox:
        """Convert ROI-relative coordinates to frame-absolute coordinates."""
        pass
```

### 2. OCRDetection Data Model

New data model for OCR detection results with bounding box information.

```python
@dataclass
class OCRDetection:
    """Represents a single OCR detection with position and confidence."""
    text: str
    confidence: float
    bbox: BoundingBox  # ROI-relative coordinates
    wagon_id: int
    frame_index: int
```

### 3. OCRFrameSaver Class

Extended frame saver for OCR-specific saving with metadata.

```python
class OCRFrameSaver:
    """Saves OCR frames with JSON metadata."""
    
    def save_ocr_frame(
        self,
        frame: np.ndarray,
        ocr_detections: List[OCRDetection],
        metadata: OCRFrameMetadata
    ) -> Tuple[str, str]:
        """Save frame and metadata, returns (frame_path, metadata_path)."""
        pass
    
    def generate_filename(
        self,
        timestamp: datetime,
        frame_index: int,
        wagon_id: int
    ) -> str:
        """Generate filename with timestamp, frame index, and wagon ID."""
        pass
```

### 4. OCRFrameMetadata Data Model

```python
@dataclass
class OCRFrameMetadata:
    """Metadata saved alongside OCR frames."""
    timestamp: str  # ISO format
    frame_index: int
    wagon_id: int
    detections: List[Dict]  # text, confidence, bbox for each detection
    deblur_applied: bool
    illumination_applied: bool
    blur_score: Optional[float]
    luminance_level: Optional[float]
```

### 5. ProcessingStatusIndicator Class

New class for rendering processing status indicators in the metrics row.

```python
class ProcessingStatusIndicator:
    """Renders processing status indicators with color coding."""
    
    # Status colors
    STATUS_COLORS = {
        "APPLIED": "#28a745",   # Green
        "ACTIVE": "#28a745",    # Green
        "NORMAL": "#6c757d",    # Gray
        "SKIPPED": "#ffc107",   # Yellow
        "OFF": "#dc3545",       # Red
        "ERROR": "#dc3545"      # Red
    }
    
    def render_indicator(
        self,
        label: str,
        status: str,
        tooltip: Optional[str] = None
    ) -> None:
        """Render a single status indicator with color coding."""
        pass
    
    def render_metrics_row(
        self,
        fps: float,
        latency_ms: float,
        object_count: int,
        wagon_count: int,
        damage_count: int,
        illumination_status: str,
        deblur_status: str,
        ocr_status: str,
        tooltips: Dict[str, str]
    ) -> None:
        """Render the complete enhanced metrics row."""
        pass
```

### 6. OCRIntervalController

Manages OCR frame interval logic.

```python
class OCRIntervalController:
    """Controls OCR execution frequency based on frame interval."""
    
    def __init__(self, interval: int = 5):
        self.interval = interval
        self._frame_counter = 0
    
    def should_run_ocr(self, frame_index: int) -> bool:
        """Returns True if OCR should run on this frame."""
        return frame_index % self.interval == 0
    
    def get_status_text(self, frame_index: int) -> str:
        """Returns status text like 'ACTIVE' or 'SKIPPED (frame 3 of 5)'."""
        pass
```

## Data Models

### Extended Session State Fields

```python
# New session state fields for OCR visual enhancements
session_state_defaults = {
    # OCR interval control
    "ocr_frame_interval": 5,
    "ocr_status": "ACTIVE",
    "ocr_last_text": "",
    "ocr_last_confidence": 0.0,
    
    # Processing status indicators
    "illumination_applied_this_frame": False,
    "deblur_applied_this_frame": False,
    "ocr_applied_this_frame": False,
    
    # OCR detections for current frame
    "current_ocr_detections": [],
}
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: OCR Bounding Box Count Matches Detection Count

*For any* list of OCR detections returned by the OCR pipeline, the number of bounding boxes drawn on the frame SHALL equal the number of detections.

**Validates: Requirements 1.1, 1.3**

### Property 2: OCR Coordinate Transformation Correctness

*For any* OCR bounding box with ROI-relative coordinates (rx1, ry1, rx2, ry2) and wagon bounding box with frame coordinates (wx1, wy1, wx2, wy2), the adjusted frame-absolute coordinates SHALL be (wx1+rx1, wy1+ry1, wx1+rx2, wy1+ry2).

**Validates: Requirements 1.4**

### Property 3: Text Overlay Contains Detection Information

*For any* OCR detection with text T and confidence C, the rendered overlay SHALL contain T and C formatted as a percentage.

**Validates: Requirements 2.1, 2.2**

### Property 4: Text Overlay Position Non-Overlapping

*For any* OCR bounding box, the text overlay position SHALL not overlap with the interior of the bounding box.

**Validates: Requirements 2.4**

### Property 5: Confidence-Based Color Selection

*For any* OCR detection with confidence C, the text color SHALL be warning color (orange) if C < 0.5, otherwise normal color (white).

**Validates: Requirements 2.5**

### Property 6: OCR Frame Interval Execution

*For any* frame index F and OCR interval N, OCR SHALL execute if and only if F % N == 0.

**Validates: Requirements 4.4, 4.6**

### Property 7: OCR Status Reflects Execution State

*For any* frame where OCR executes, status SHALL be "ACTIVE". For any frame where OCR is skipped due to interval, status SHALL contain "SKIPPED".

**Validates: Requirements 5.2, 5.3**

### Property 8: Text Truncation at 50 Characters

*For any* detected text longer than 50 characters, the displayed text SHALL be truncated to 50 characters followed by ellipsis.

**Validates: Requirements 5.4**

### Property 9: Metadata File Contains Required Fields

*For any* saved OCR frame, the accompanying JSON metadata file SHALL contain: timestamp, frame_index, wagon_id, detections array, and deblur_applied boolean.

**Validates: Requirements 3.3, 6.2**

### Property 10: Processing Indicators Reflect Current State

*For any* processed frame, the illumination indicator SHALL show "APPLIED" if CLAHE/gamma was applied, "NORMAL" if not needed, or "OFF" if disabled. The deblur indicator SHALL show "APPLIED" if deblurring was applied, "SKIPPED" if not needed, or "OFF" if disabled.

**Validates: Requirements 7.2, 7.3, 7.4**

### Property 11: Settings Persistence Round-Trip

*For any* OCR interval value V saved to settings, loading settings SHALL return the same value V.

**Validates: Requirements 4.7**

## Error Handling

### OCR Pipeline Errors

- If EasyOCR fails to initialize, display "OCR: ERROR" indicator and log the error
- If OCR extraction fails on a frame, skip OCR for that frame and continue processing
- If frame saving fails, log error but don't interrupt video processing

### Visualization Errors

- If bounding box coordinates are invalid (negative or out of bounds), clip to frame boundaries
- If text overlay would be outside frame, adjust position to stay within bounds

### Settings Errors

- If settings file is corrupted, reset to defaults and log warning
- If interval value is out of range (< 1 or > 30), clamp to valid range

## Testing Strategy

### Unit Tests

Unit tests will verify specific examples and edge cases:

1. **OCR Visualization Tests**
   - Test bounding box drawing with single detection
   - Test bounding box drawing with multiple detections
   - Test coordinate transformation with various ROI positions
   - Test text overlay positioning
   - Test color selection based on confidence threshold

2. **Frame Saving Tests**
   - Test filename generation format
   - Test metadata JSON structure
   - Test directory creation when missing

3. **Interval Controller Tests**
   - Test should_run_ocr with various frame indices and intervals
   - Test status text generation

4. **Processing Indicator Tests**
   - Test indicator rendering with each status value
   - Test color mapping for each status

### Property-Based Tests

Property-based tests will use Hypothesis to verify universal properties:

1. **Property 1 Test**: Generate random lists of OCR detections, verify box count matches
2. **Property 2 Test**: Generate random ROI and wagon bboxes, verify coordinate transformation
3. **Property 5 Test**: Generate random confidence values, verify color selection
4. **Property 6 Test**: Generate random frame indices and intervals, verify OCR execution logic
5. **Property 8 Test**: Generate random strings, verify truncation behavior
6. **Property 11 Test**: Generate random interval values, verify save/load round-trip

### Integration Tests

1. Test full OCR pipeline with visualization enabled
2. Test frame saving with actual OCR detections
3. Test metrics row rendering with all indicators
