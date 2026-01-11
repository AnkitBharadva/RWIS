# Design Document: Streamlit Mission Control Dashboard

## Overview

This document describes the technical design for a "Mission Control" style Streamlit dashboard for the High-Speed Railway Wagon Inspection System. The dashboard provides a professional industrial monitoring interface with real-time video display, live metrics, and detection logging.

The dashboard follows these design principles:
- Dark-themed, high-contrast industrial aesthetic
- Efficient frame updates using st.empty() to prevent flickering
- Frame-skipping (N=3) to maintain UI responsiveness
- Graceful resource management for video capture
- Clear visual feedback for damage detection status

## Architecture

The dashboard consists of the following layers:

1. **UI Layer**: Streamlit components with custom CSS styling
2. **State Management**: Streamlit session state for persistent data
3. **Video Processing**: OpenCV VideoCapture with frame-skipping
4. **Integration Layer**: Connection to existing pipeline components

```
┌─────────────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────────────────────────────────┐  │
│  │   Sidebar   │  │              Main Content               │  │
│  │             │  │  ┌─────────────────────────────────────┐│  │
│  │ - Source    │  │  │         Metrics Row (4 cols)       ││  │
│  │ - Settings  │  │  │  FPS | Latency | Objects | Damage  ││  │
│  │ - Controls  │  │  └─────────────────────────────────────┘│  │
│  │             │  │  ┌─────────────────────────────────────┐│  │
│  │             │  │  │                                     ││  │
│  │             │  │  │         Live Video Feed             ││  │
│  │             │  │  │                                     ││  │
│  │             │  │  └─────────────────────────────────────┘│  │
│  │             │  │  ┌─────────────────────────────────────┐│  │
│  │             │  │  │         Detection Log               ││  │
│  │             │  │  └─────────────────────────────────────┘│  │
│  └─────────────┘  └─────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Components and Interfaces

### 1. Main Dashboard Module (dashboard.py)

```python
import streamlit as st
import cv2
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class DashboardState:
    is_running: bool = False
    video_source: str = ""
    frame_count: int = 0
    fps: float = 0.0
    inference_ms: float = 0.0
    object_count: int = 0
    damage_count: int = 0
    detection_log: list = None
    
    def __post_init__(self):
        if self.detection_log is None:
            self.detection_log = []

class MissionControlDashboard:
    def __init__(self):
        """Initialize dashboard with page config and CSS."""
        
    def inject_custom_css(self) -> None:
        """Inject dark-themed CSS styling."""
        
    def render_sidebar(self) -> Dict[str, Any]:
        """Render sidebar controls and return settings."""
        
    def render_metrics_row(
        self, 
        metrics_placeholder: st.empty,
        fps: float,
        inference_ms: float,
        object_count: int,
        damage_count: int,
        damage_detected: bool
    ) -> None:
        """Render the top metrics row."""
        
    def render_video_feed(
        self,
        video_placeholder: st.empty,
        frame: Optional[np.ndarray]
    ) -> None:
        """Render the live video feed."""
        
    def render_detection_log(
        self,
        log_placeholder: st.empty,
        detection_log: list
    ) -> None:
        """Render the detection log table."""
        
    def run(self) -> None:
        """Main dashboard loop."""
```

### 2. Video Capture Manager (video_manager.py)

```python
class VideoManager:
    def __init__(self, frame_skip: int = 3):
        """Initialize video manager with frame skip interval."""
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_skip = frame_skip
        self.frame_count = 0
        
    def connect(self, source: str) -> bool:
        """Connect to video source (RTSP URL or file path)."""
        
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read frame with frame-skipping logic."""
        
    def release(self) -> None:
        """Gracefully release video capture resources."""
        
    def is_connected(self) -> bool:
        """Check if video source is connected."""
```

### 3. Metrics Calculator (metrics.py)

```python
class MetricsCalculator:
    def __init__(self):
        """Initialize metrics tracking."""
        self.last_time: float = 0.0
        self.fps_history: list = []
        self.inference_history: list = []
        
    def start_frame(self) -> None:
        """Mark start of frame processing."""
        
    def end_frame(self) -> None:
        """Mark end of frame processing and calculate metrics."""
        
    def get_fps(self) -> float:
        """Get smoothed FPS value."""
        
    def get_inference_ms(self) -> float:
        """Get smoothed inference latency in ms."""
        
    def record_inference_time(self, duration_ms: float) -> None:
        """Record inference duration for averaging."""
```

### 4. CSS Injection Module (styles.py)

```python
CUSTOM_CSS = """
<style>
    /* Reduce top padding */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
    
    /* Dark metric card styling */
    div[data-testid="metric-container"] {
        background-color: #1E1E1E;
        border: 1px solid #3D3D3D;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Large bold metric values */
    div[data-testid="metric-container"] > div > div > div {
        font-size: 2rem;
        font-weight: 700;
    }
    
    /* Metric label styling */
    div[data-testid="metric-container"] label {
        font-size: 0.9rem;
        color: #B0B0B0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Status indicator colors */
    .status-normal {
        color: #00FF00;
        font-weight: bold;
    }
    
    .status-alert {
        color: #FF4444;
        font-weight: bold;
    }
    
    /* Video container styling */
    .video-container {
        border: 2px solid #3D3D3D;
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Detection log styling */
    .detection-log {
        background-color: #1E1E1E;
        border-radius: 8px;
        padding: 10px;
    }
</style>
"""

def inject_css() -> None:
    """Inject custom CSS into Streamlit app."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
```

## Data Models

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from enum import Enum

class ConnectionStatus(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"

@dataclass
class DetectionLogEntry:
    timestamp: datetime
    wagon_id: int
    damage_type: str
    confidence: float
    frame_index: int

@dataclass
class DashboardMetrics:
    fps: float = 0.0
    inference_ms: float = 0.0
    object_count: int = 0
    damage_count: int = 0
    damage_detected: bool = False

@dataclass
class SidebarSettings:
    video_source: str = ""
    confidence_threshold: float = 0.5
    frame_skip: int = 3
    enable_damage_detection: bool = True

@dataclass
class SessionState:
    is_running: bool = False
    connection_status: ConnectionStatus = ConnectionStatus.DISCONNECTED
    metrics: DashboardMetrics = field(default_factory=DashboardMetrics)
    settings: SidebarSettings = field(default_factory=SidebarSettings)
    detection_log: List[DetectionLogEntry] = field(default_factory=list)
    last_frame: Optional[np.ndarray] = None
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system - essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Frame Skip Interval Enforcement

*For any* video processing session with configured frame_skip interval N, the dashboard SHALL process exactly 1 frame for every N frames captured. The frame_count modulo N determines which frames are processed.

**Validates: Requirements 7.1**

### Property 2: Video Resource Cleanup

*For any* video capture session, when the Stop button is pressed or the dashboard is closed, the cv2.VideoCapture resource SHALL be released. After release, is_connected() SHALL return False.

**Validates: Requirements 2.5, 7.5**

### Property 3: Metrics Value Ranges

*For any* metrics update:
- FPS SHALL be >= 0.0
- Inference latency SHALL be >= 0.0 milliseconds
- Object count SHALL be >= 0
- Damage count SHALL be >= 0

**Validates: Requirements 3.2, 3.3, 3.4, 3.5**

### Property 4: Detection Log Append-Only

*For any* detection log, new entries SHALL only be appended to the end. The log SHALL maintain chronological order by timestamp. Existing entries SHALL NOT be modified or removed during a session.

**Validates: Requirements 5.3, 5.5**

### Property 5: Visual Feedback Consistency

*For any* damage detection state:
- If damage_detected is True, the visual indicator SHALL be red/alert
- If damage_detected is False, the visual indicator SHALL be green/normal
- The indicator state SHALL match the current damage_detected value

**Validates: Requirements 6.1, 6.2, 6.4**

### Property 6: UI Update Isolation

*For any* frame update cycle, only the designated placeholder containers (metrics, video, log) SHALL be updated. The sidebar and page header SHALL NOT re-render during the update loop.

**Validates: Requirements 3.6, 4.4**

## Error Handling

### Video Source Errors

| Error Condition | Handling Strategy |
|----------------|-------------------|
| Invalid RTSP URL | Display st.error() with message, set status to ERROR |
| File not found | Display st.error() with path, set status to ERROR |
| Connection timeout | Retry 3 times with 1s delay, then display error |
| Stream disconnected | Set status to DISCONNECTED, display reconnect option |
| Corrupted frame | Skip frame, continue processing, log warning |

### Resource Management Errors

| Error Condition | Handling Strategy |
|----------------|-------------------|
| VideoCapture already released | Check is_connected() before operations |
| Memory overflow | Limit detection_log to last 1000 entries |
| Session state corruption | Reset to default state, display warning |

### UI Errors

| Error Condition | Handling Strategy |
|----------------|-------------------|
| CSS injection failure | Continue without custom styling, log warning |
| Placeholder update failure | Skip update cycle, continue processing |
| Invalid metric values | Clamp to valid ranges, display warning |

## Testing Strategy

### Unit Tests

Unit tests verify specific examples and edge cases:

1. **CSS Injection Tests**
   - Test CSS string is valid and contains required selectors
   - Test injection does not raise exceptions

2. **Video Manager Tests**
   - Test connect() with valid file path
   - Test connect() with invalid source
   - Test release() properly cleans up resources
   - Test frame_skip logic counts correctly

3. **Metrics Calculator Tests**
   - Test FPS calculation accuracy
   - Test inference time averaging
   - Test edge cases (zero time, very high FPS)

4. **Session State Tests**
   - Test default initialization
   - Test state persistence across reruns
   - Test detection log append behavior

### Property-Based Tests

Property-based tests verify universal properties across many generated inputs. Each test runs minimum 100 iterations.

**Testing Framework**: pytest with hypothesis library

1. **Property Test: Frame Skip Interval**
   - Generate random frame counts and skip intervals
   - Verify correct frames are processed
   - **Tag: Feature: streamlit-dashboard, Property 1: Frame Skip Interval Enforcement**

2. **Property Test: Metrics Value Ranges**
   - Generate random metric values
   - Verify all values are clamped to valid ranges
   - **Tag: Feature: streamlit-dashboard, Property 3: Metrics Value Ranges**

3. **Property Test: Detection Log Ordering**
   - Generate random detection entries
   - Verify chronological ordering is maintained
   - **Tag: Feature: streamlit-dashboard, Property 4: Detection Log Append-Only**

4. **Property Test: Visual Feedback Consistency**
   - Generate random damage_detected states
   - Verify indicator color matches state
   - **Tag: Feature: streamlit-dashboard, Property 5: Visual Feedback Consistency**

### Integration Tests

1. **End-to-End Dashboard Test**
   - Start dashboard with test video file
   - Verify metrics update correctly
   - Verify video displays without errors
   - Verify detection log populates

2. **Start/Stop Cycle Test**
   - Start processing, verify running state
   - Stop processing, verify resources released
   - Restart processing, verify clean state

3. **UI Responsiveness Test**
   - Process video and measure UI update latency
   - Verify no flickering in sidebar/header
