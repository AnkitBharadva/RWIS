# Design Document: Dashboard Enhancements

## Overview

This document describes the technical design for enhancing the Mission Control Streamlit dashboard with additional monitoring capabilities. The enhancements build upon the existing dashboard architecture to add dual video display, wagon tracking visualization, comprehensive metrics, frame saving, and OCR logging.

The design follows these principles:
- Minimal changes to existing architecture
- Efficient updates using st.empty() placeholders
- Consistent dark-themed industrial aesthetic
- Modular components for easy maintenance

## Architecture

The enhanced dashboard extends the existing architecture with new components:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Enhanced Streamlit Dashboard                          │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────────────────────────────────────────┐  │
│  │   Sidebar   │  │              Main Content                        │  │
│  │             │  │  ┌─────────────────────────────────────────────┐│  │
│  │ - Source    │  │  │         Metrics Row (5 cols)                ││  │
│  │ - Settings  │  │  │  FPS | Latency | Objects | Wagons | Damage  ││  │
│  │ - Controls  │  │  └─────────────────────────────────────────────┘│  │
│  │ - Frame     │  │  ┌──────────────────┬──────────────────────────┐│  │
│  │   Saving    │  │  │   Raw Input      │   Processed Output       ││  │
│  │             │  │  │   (no overlays)  │   (with Track IDs,       ││  │
│  │             │  │  │                  │    bboxes, annotations)  ││  │
│  │             │  │  └──────────────────┴──────────────────────────┘│  │
│  │             │  │  ┌──────────────────┬──────────────────────────┐│  │
│  │             │  │  │  Detection Log   │      OCR Log             ││  │
│  │             │  │  └──────────────────┴──────────────────────────┘│  │
│  └─────────────┘  └─────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

## Components and Interfaces

### 1. Enhanced Video Display Component

```python
class DualVideoDisplay:
    """Manages side-by-side raw and processed video display."""
    
    def __init__(self):
        """Initialize dual video display placeholders."""
        self.raw_placeholder: st.empty = None
        self.processed_placeholder: st.empty = None
        self.current_frame_index: int = 0
    
    def render(
        self,
        raw_frame: Optional[np.ndarray],
        processed_frame: Optional[np.ndarray],
        frame_index: int
    ) -> None:
        """
        Render both video frames side by side.
        
        Args:
            raw_frame: Original unprocessed BGR frame
            processed_frame: Frame with bounding boxes and annotations
            frame_index: Current frame index for synchronization
        """
        pass
    
    def render_placeholders(self) -> None:
        """Render placeholder messages when video is stopped."""
        pass
```

### 2. Track ID Overlay Renderer

```python
class TrackIDRenderer:
    """Renders track IDs on video frames."""
    
    LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
    LABEL_SCALE = 0.7
    LABEL_THICKNESS = 2
    LABEL_COLOR = (255, 255, 0)  # Cyan for contrast
    BACKGROUND_COLOR = (0, 0, 0)  # Black background
    
    def draw_track_ids(
        self,
        frame: np.ndarray,
        tracked_wagons: List[TrackedWagon]
    ) -> np.ndarray:
        """
        Draw track IDs on frame for all tracked wagons.
        
        Args:
            frame: BGR frame to annotate
            tracked_wagons: List of tracked wagons with IDs
            
        Returns:
            Annotated frame with track IDs
        """
        pass
    
    def _calculate_label_position(
        self,
        bbox: BoundingBox
    ) -> Tuple[int, int]:
        """Calculate optimal label position near bounding box."""
        pass
```

### 3. Enhanced Metrics Panel

```python
@dataclass
class EnhancedMetrics:
    """Extended metrics including wagon count and latency."""
    fps: float = 0.0
    processing_latency_ms: float = 0.0
    objects_detected: int = 0
    total_wagon_count: int = 0
    damage_count: int = 0
    damage_detected: bool = False

class EnhancedMetricsPanel:
    """Renders the enhanced 5-column metrics panel."""
    
    LATENCY_WARNING_THRESHOLD_MS = 100.0
    
    def __init__(self):
        """Initialize metrics panel with smoothing."""
        self.latency_history: List[float] = []
        self.smoothing_window: int = 10
    
    def render(
        self,
        placeholder: st.empty,
        metrics: EnhancedMetrics
    ) -> None:
        """Render all metrics in a 5-column layout."""
        pass
    
    def smooth_latency(self, latency_ms: float) -> float:
        """Apply moving average smoothing to latency."""
        pass
    
    def get_latency_indicator(self, latency_ms: float) -> str:
        """Get visual indicator based on latency threshold."""
        pass
```

### 4. Frame Saver Component

```python
@dataclass
class FrameSaveConfig:
    """Configuration for frame saving."""
    enabled: bool = False
    save_on_deblur: bool = True
    save_on_illumination: bool = True
    save_on_ocr: bool = True
    output_directory: str = "outputs/saved_frames"

@dataclass
class FrameMetadata:
    """Metadata for saved frames."""
    timestamp: datetime
    frame_index: int
    processing_applied: List[str]  # ["deblur", "clahe", "ocr"]
    wagon_id: Optional[int] = None

class FrameSaver:
    """Manages automatic saving of processed frames."""
    
    def __init__(self, config: FrameSaveConfig):
        """Initialize frame saver with configuration."""
        self.config = config
        self._ensure_output_directory()
    
    def should_save(
        self,
        deblur_applied: bool,
        illumination_applied: bool,
        ocr_performed: bool
    ) -> bool:
        """Determine if frame should be saved based on config."""
        pass
    
    def save_frame(
        self,
        frame: np.ndarray,
        metadata: FrameMetadata
    ) -> Optional[str]:
        """
        Save frame to disk with metadata.
        
        Returns:
            Path to saved file, or None if saving disabled
        """
        pass
    
    def _generate_filename(self, metadata: FrameMetadata) -> str:
        """Generate filename with timestamp and processing info."""
        pass
```

### 5. OCR Log Display Component

```python
@dataclass
class OCRLogEntry:
    """Single entry in the OCR log."""
    timestamp: datetime
    wagon_id: int
    extracted_text: str
    confidence: float
    frame_index: int

class OCRLogDisplay:
    """Manages OCR log display and storage."""
    
    MAX_LOG_ENTRIES = 500
    
    def __init__(self):
        """Initialize OCR log storage."""
        self.entries: List[OCRLogEntry] = []
    
    def append_entry(self, entry: OCRLogEntry) -> None:
        """Append new OCR entry maintaining chronological order."""
        pass
    
    def render(self, placeholder: st.empty) -> None:
        """Render OCR log in collapsible expander."""
        pass
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert log entries to pandas DataFrame."""
        pass
```

## Data Models

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from enum import Enum

class ProcessingType(Enum):
    """Types of processing that can be applied to frames."""
    DEBLUR = "deblur"
    CLAHE = "clahe"
    GAMMA = "gamma"
    OCR = "ocr"

@dataclass
class DualFrameState:
    """State for dual video display."""
    raw_frame: Optional[np.ndarray] = None
    processed_frame: Optional[np.ndarray] = None
    frame_index: int = 0
    synchronized: bool = True

@dataclass
class WagonTrackingState:
    """State for wagon tracking display."""
    tracked_wagons: List[TrackedWagon] = field(default_factory=list)
    total_count: int = 0
    crossed_ids: set = field(default_factory=set)

@dataclass
class LatencyMetrics:
    """Latency measurement state."""
    current_ms: float = 0.0
    smoothed_ms: float = 0.0
    history: List[float] = field(default_factory=list)
    warning_active: bool = False

@dataclass
class EnhancedSessionState:
    """Extended session state for enhanced dashboard."""
    # Existing state
    is_running: bool = False
    connection_status: str = "Disconnected"
    detection_log: List[DetectionLogEntry] = field(default_factory=list)
    
    # New state for enhancements
    dual_frame: DualFrameState = field(default_factory=DualFrameState)
    wagon_tracking: WagonTrackingState = field(default_factory=WagonTrackingState)
    latency: LatencyMetrics = field(default_factory=LatencyMetrics)
    ocr_log: List[OCRLogEntry] = field(default_factory=list)
    frame_save_config: FrameSaveConfig = field(default_factory=FrameSaveConfig)
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system - essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Dual Frame Synchronization

*For any* frame update cycle, both the raw frame display and processed frame display SHALL show frames from the same frame index. The raw frame SHALL contain no overlays or modifications, while the processed frame SHALL contain all detection annotations.

**Validates: Requirements 1.2, 1.3, 1.4**

### Property 2: Track ID Overlay Completeness

*For any* set of tracked wagons in a frame, the processed frame overlay SHALL display a Track_ID label for each wagon. The label position SHALL be within or adjacent to the wagon's bounding box coordinates.

**Validates: Requirements 2.1, 2.2, 2.4, 2.5**

### Property 3: Wagon Counting Correctness

*For any* sequence of wagon detections crossing the counting line:
- The total wagon count SHALL equal the number of unique wagon IDs that crossed
- Each wagon SHALL be counted exactly once (no duplicates)
- The count SHALL increment by exactly 1 for each new crossing
- The count SHALL persist across frame updates

**Validates: Requirements 3.2, 3.3, 3.4, 3.5**

### Property 4: Latency Measurement Accuracy

*For any* processed frame:
- The latency measurement SHALL be >= 0 milliseconds
- The smoothed latency SHALL be the moving average of the last N measurements
- When latency exceeds the warning threshold, visual feedback SHALL be displayed

**Validates: Requirements 4.2, 4.3, 4.4, 4.5**

### Property 5: Conditional Frame Saving

*For any* frame processing event:
- If frame saving is disabled, no files SHALL be created
- If deblur is applied AND save_on_deblur is True, the frame SHALL be saved
- If illumination is applied AND save_on_illumination is True, the frame SHALL be saved
- If OCR is performed AND save_on_ocr is True, the frame SHALL be saved
- Saved files SHALL include metadata indicating processing type

**Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 8.4**

### Property 6: OCR Log Entry Completeness

*For any* OCR extraction event:
- A log entry SHALL be appended with timestamp, wagon_id, text, and confidence
- The log SHALL maintain chronological order (entries sorted by timestamp)
- All required fields SHALL be present and valid

**Validates: Requirements 6.2, 6.4, 6.6**

### Property 7: Configuration Persistence

*For any* configuration change in the sidebar:
- The configuration SHALL persist in session state across reruns
- Reloading the dashboard SHALL restore the previous configuration values

**Validates: Requirements 8.5**

## Error Handling

### Video Display Errors

| Error Condition | Handling Strategy |
|----------------|-------------------|
| Raw frame is None | Display placeholder in raw frame container |
| Processed frame is None | Display placeholder in processed frame container |
| Frame conversion failure | Log warning, display last valid frame |
| Synchronization mismatch | Reset both displays, log warning |

### Frame Saving Errors

| Error Condition | Handling Strategy |
|----------------|-------------------|
| Output directory not writable | Display st.error(), disable saving |
| Disk full | Display st.warning(), skip save, continue processing |
| Invalid frame data | Log warning, skip save |
| Metadata serialization failure | Save frame without metadata, log warning |

### OCR Log Errors

| Error Condition | Handling Strategy |
|----------------|-------------------|
| Log exceeds MAX_ENTRIES | Remove oldest entries to maintain limit |
| Invalid OCR result | Skip logging, continue processing |
| DataFrame conversion failure | Display raw list, log warning |

## Testing Strategy

### Unit Tests

Unit tests verify specific examples and edge cases:

1. **DualVideoDisplay Tests**
   - Test placeholder rendering when frames are None
   - Test frame synchronization with matching indices
   - Test BGR to RGB conversion for display

2. **TrackIDRenderer Tests**
   - Test label position calculation for various bbox positions
   - Test rendering with empty wagon list
   - Test rendering with multiple overlapping wagons

3. **FrameSaver Tests**
   - Test should_save() with various config combinations
   - Test filename generation format
   - Test metadata serialization

4. **OCRLogDisplay Tests**
   - Test append_entry() maintains chronological order
   - Test MAX_LOG_ENTRIES limit enforcement
   - Test DataFrame conversion

### Property-Based Tests

Property-based tests verify universal properties across many generated inputs. Each test runs minimum 100 iterations.

**Testing Framework**: pytest with hypothesis library

1. **Property Test: Dual Frame Synchronization**
   - Generate random frame pairs with indices
   - Verify both displays show same frame index
   - **Tag: Feature: dashboard-enhancements, Property 1: Dual Frame Synchronization**

2. **Property Test: Track ID Overlay Completeness**
   - Generate random sets of tracked wagons
   - Verify all Track_IDs are rendered on frame
   - **Tag: Feature: dashboard-enhancements, Property 2: Track ID Overlay Completeness**

3. **Property Test: Wagon Counting Correctness**
   - Generate random sequences of wagon crossings
   - Verify count equals unique crossings, no duplicates
   - **Tag: Feature: dashboard-enhancements, Property 3: Wagon Counting Correctness**

4. **Property Test: Latency Measurement Accuracy**
   - Generate random latency sequences
   - Verify smoothing calculation and threshold detection
   - **Tag: Feature: dashboard-enhancements, Property 4: Latency Measurement Accuracy**

5. **Property Test: Conditional Frame Saving**
   - Generate random processing events and configs
   - Verify correct save/no-save decisions
   - **Tag: Feature: dashboard-enhancements, Property 5: Conditional Frame Saving**

6. **Property Test: OCR Log Entry Completeness**
   - Generate random OCR results
   - Verify all entries have required fields and chronological order
   - **Tag: Feature: dashboard-enhancements, Property 6: OCR Log Entry Completeness**

7. **Property Test: Configuration Persistence**
   - Generate random config values
   - Verify persistence across simulated reruns
   - **Tag: Feature: dashboard-enhancements, Property 7: Configuration Persistence**

### Integration Tests

1. **End-to-End Dual Display Test**
   - Start dashboard with test video
   - Verify both frames display correctly
   - Verify Track IDs appear on processed frame

2. **Frame Saving Integration Test**
   - Enable frame saving with all triggers
   - Process video with deblur/CLAHE/OCR
   - Verify files created with correct metadata

3. **OCR Log Integration Test**
   - Process video with OCR-enabled wagons
   - Verify OCR log populates correctly
   - Verify scrolling and expander functionality

