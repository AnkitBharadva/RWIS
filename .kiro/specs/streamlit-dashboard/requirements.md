# Requirements Document

## Introduction

This document specifies the requirements for a "Mission Control" style Streamlit dashboard for the High-Speed Railway Wagon Inspection System. The dashboard provides a professional industrial monitoring interface with dark-themed, high-contrast visuals for real-time video monitoring, live metrics display, and detection logging.

## Glossary

- **Dashboard**: The Streamlit-based web interface for monitoring the inspection pipeline
- **Live_Feed**: The real-time video stream display component
- **Metrics_Panel**: The row of live statistics cards showing FPS, inference time, objects, and damage count
- **Detection_Log**: Historical table of detected damage events with timestamps
- **Frame_Skip**: Technique of processing every Nth frame to maintain UI responsiveness
- **RTSP**: Real-Time Streaming Protocol for camera input
- **CSS_Injection**: Custom styling applied via st.markdown to override Streamlit defaults

## Requirements

### Requirement 1: Page Layout Configuration

**User Story:** As a railway inspector, I want a wide, professional-looking dashboard, so that I can monitor the inspection system effectively on large displays.

#### Acceptance Criteria

1. THE Dashboard SHALL use Streamlit's wide layout mode via st.set_page_config(layout="wide")
2. THE Dashboard SHALL apply custom CSS to reduce top padding for maximum screen utilization
3. THE Dashboard SHALL use a dark-themed color scheme with high-contrast elements
4. THE Dashboard SHALL style metric cards with borders and subtle dark grey/slate backgrounds
5. THE Dashboard SHALL use large, bold fonts for metric values for at-a-glance reading

### Requirement 2: Sidebar Controls

**User Story:** As a system operator, I want sidebar controls for connection settings and pipeline parameters, so that I can configure the system without cluttering the main view.

#### Acceptance Criteria

1. THE Sidebar SHALL contain video source input (RTSP URL or file path)
2. THE Sidebar SHALL contain model sensitivity/confidence threshold sliders
3. THE Sidebar SHALL contain Start and Stop control buttons
4. WHEN the Start button is clicked, THE Dashboard SHALL begin video capture and processing
5. WHEN the Stop button is clicked, THE Dashboard SHALL gracefully release the video capture resource
6. THE Sidebar SHALL display current connection status (Connected/Disconnected)

### Requirement 3: Live Metrics Display

**User Story:** As a railway inspector, I want to see live performance metrics at a glance, so that I can monitor system health and detection activity.

#### Acceptance Criteria

1. THE Metrics_Panel SHALL display four distinct metric cards in a top row
2. THE Metrics_Panel SHALL show current FPS (frames per second)
3. THE Metrics_Panel SHALL show inference latency in milliseconds
4. THE Metrics_Panel SHALL show current object count (wagons detected)
5. THE Metrics_Panel SHALL show damage detection count
6. WHEN metrics are updated, THE Dashboard SHALL refresh only the metric containers without flickering

### Requirement 4: Live Video Feed Display

**User Story:** As a railway inspector, I want to see the live video feed with detection overlays, so that I can visually monitor the inspection process.

#### Acceptance Criteria

1. THE Live_Feed SHALL be displayed in a large, centered container in the middle row
2. THE Live_Feed SHALL use st.empty() for efficient frame updates
3. THE Live_Feed SHALL display detection bounding boxes overlaid on the video
4. THE Live_Feed SHALL update without causing sidebar or header flickering
5. WHEN video capture is stopped, THE Live_Feed SHALL display a placeholder or last frame

### Requirement 5: Detection Log Display

**User Story:** As a railway inspector, I want to see a historical log of damage detections, so that I can review past events and track inspection results.

#### Acceptance Criteria

1. THE Detection_Log SHALL be displayed in the bottom row of the dashboard
2. THE Detection_Log SHALL show timestamp, wagon ID, damage type, and confidence for each detection
3. THE Detection_Log SHALL support scrolling for viewing historical entries
4. THE Detection_Log MAY use st.expander for collapsible view
5. WHEN new damage is detected, THE Detection_Log SHALL append the entry to the log

### Requirement 6: Visual Feedback for Damage Status

**User Story:** As a railway inspector, I want clear visual indicators for damage status, so that I can immediately notice when damage is detected.

#### Acceptance Criteria

1. WHEN damage is detected, THE Dashboard SHALL display a red visual indicator or st.error() alert
2. WHEN no damage is detected (normal status), THE Dashboard SHALL display a green indicator
3. THE damage status indicator SHALL be prominently visible in the metrics area
4. THE visual feedback SHALL update in real-time as detection status changes

### Requirement 7: Frame Processing Performance

**User Story:** As a system operator, I want the dashboard to maintain UI responsiveness, so that the interface remains usable during video processing.

#### Acceptance Criteria

1. THE Dashboard SHALL implement frame-skipping with configurable interval (default N=3)
2. THE Dashboard SHALL use time.perf_counter() for high-precision latency measurement
3. THE Dashboard SHALL use OpenCV (cv2.VideoCapture) for frame grabbing
4. THE Dashboard SHALL process frames in a while loop that refreshes only necessary containers
5. WHEN the Stop button is pressed, THE Dashboard SHALL gracefully release cv2.VideoCapture resources

### Requirement 8: CSS Styling Requirements

**User Story:** As a UI designer, I want consistent professional styling throughout the dashboard, so that it looks like a proper industrial monitoring tool.

#### Acceptance Criteria

1. THE Dashboard SHALL inject custom CSS via st.markdown with unsafe_allow_html=True
2. THE CSS SHALL style st.metric components with dark grey/slate backgrounds
3. THE CSS SHALL add subtle borders to metric cards
4. THE CSS SHALL reduce default Streamlit padding at the top of the page
5. THE CSS SHALL ensure high contrast between text and backgrounds for readability
