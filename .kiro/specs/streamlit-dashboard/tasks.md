
# Implementation Plan: Streamlit Mission Control Dashboard

## Overview

This implementation plan breaks down the Streamlit Mission Control Dashboard into discrete, incremental coding tasks. Each task builds on previous work, ensuring no orphaned code. The implementation uses Python 3.10+ with Streamlit, OpenCV, and integrates with the existing railway wagon inspection pipeline.

## Tasks

- [x] 1. Set up dashboard project structure and dependencies
  - [x] 1.1 Update requirements.txt with Streamlit dependencies
    - Add streamlit>=1.28.0 to requirements.txt
    - Add any additional UI dependencies if needed
    - _Requirements: 1.1_

  - [x] 1.2 Create dashboard module structure
    - Create dashboard/ directory
    - Create dashboard/__init__.py
    - Create dashboard/styles.py for CSS injection
    - Create dashboard/video_manager.py for video capture
    - Create dashboard/metrics.py for metrics calculation
    - Create dashboard/app.py for main dashboard
    - _Requirements: 1.1, 8.1_

- [x] 2. Implement CSS injection module
  - [x] 2.1 Implement styles.py with custom CSS
    - Define CUSTOM_CSS constant with dark theme styling
    - Style metric containers with dark grey/slate backgrounds (#1E1E1E)
    - Add subtle borders (1px solid #3D3D3D) to metric cards
    - Reduce top padding (.block-container padding-top: 1rem)
    - Style metric values with large bold fonts (2rem, 700 weight)
    - Add status indicator classes (.status-normal green, .status-alert red)
    - Implement inject_css() function using st.markdown with unsafe_allow_html=True
    - _Requirements: 1.2, 1.3, 1.4, 1.5, 8.1, 8.2, 8.3, 8.4, 8.5_

  - [x] 2.2 Write unit tests for CSS injection
    - Test CSS string contains required selectors
    - Test inject_css() does not raise exceptions
    - _Requirements: 8.1_

- [x] 3. Implement video manager module
  - [x] 3.1 Implement VideoManager class in video_manager.py
    - Implement __init__ with frame_skip parameter (default 3)
    - Implement connect(source: str) -> bool for RTSP/file connection
    - Implement read_frame() -> Tuple[bool, Optional[np.ndarray]] with frame-skipping
    - Implement release() to gracefully release cv2.VideoCapture
    - Implement is_connected() -> bool to check connection status
    - Track frame_count for skip logic
    - _Requirements: 2.1, 2.4, 2.5, 7.1, 7.3, 7.5_

  - [x] 3.2 Write property test for frame skip interval
    - **Property 1: Frame Skip Interval Enforcement**
    - Generate random frame counts and skip intervals
    - Verify exactly 1 frame processed per N frames
    - **Validates: Requirements 7.1**

  - [x] 3.3 Write property test for video resource cleanup
    - **Property 2: Video Resource Cleanup**
    - Generate random video sessions
    - Verify release() properly cleans up, is_connected() returns False after
    - **Validates: Requirements 2.5, 7.5**

- [x] 4. Implement metrics calculator module
  - [x] 4.1 Implement MetricsCalculator class in metrics.py
    - Implement __init__ with history buffers for smoothing
    - Implement start_frame() using time.perf_counter()
    - Implement end_frame() to calculate frame duration
    - Implement get_fps() returning smoothed FPS value
    - Implement get_inference_ms() returning smoothed latency
    - Implement record_inference_time(duration_ms: float)
    - Use rolling average for smooth metric display
    - _Requirements: 3.2, 3.3, 7.2_

  - [x] 4.2 Write property test for metrics value ranges
    - **Property 3: Metrics Value Ranges**
    - Generate random metric inputs
    - Verify FPS >= 0, inference_ms >= 0, counts >= 0
    - **Validates: Requirements 3.2, 3.3, 3.4, 3.5**

- [x] 5. Checkpoint - Core modules complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Implement data models for dashboard
  - [x] 6.1 Create dashboard data models in dashboard/models.py
    - Implement ConnectionStatus enum (DISCONNECTED, CONNECTING, CONNECTED, ERROR)
    - Implement DetectionLogEntry dataclass (timestamp, wagon_id, damage_type, confidence, frame_index)
    - Implement DashboardMetrics dataclass (fps, inference_ms, object_count, damage_count, damage_detected)
    - Implement SidebarSettings dataclass (video_source, confidence_threshold, frame_skip, enable_damage_detection)
    - Implement SessionState dataclass with all dashboard state
    - _Requirements: 3.2, 3.3, 3.4, 3.5, 5.2_

  - [x] 6.2 Write property test for detection log append-only
    - **Property 4: Detection Log Append-Only**
    - Generate random detection entries
    - Verify log maintains chronological order, entries only appended
    - **Validates: Requirements 5.5**

- [x] 7. Implement sidebar component
  - [x] 7.1 Implement render_sidebar() function in app.py
    - Create video source text input (RTSP URL or file path)
    - Create confidence threshold slider (0.0 to 1.0)
    - Create frame skip number input (1 to 10, default 3)
    - Create Start button that sets is_running=True
    - Create Stop button that sets is_running=False and calls release()
    - Display connection status indicator (Connected/Disconnected)
    - Return SidebarSettings dataclass with current values
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

- [x] 8. Implement metrics panel component
  - [x] 8.1 Implement render_metrics_row() function in app.py
    - Create 4-column layout using st.columns([1, 1, 1, 1])
    - Render FPS metric using st.metric in column 1
    - Render Inference (ms) metric in column 2
    - Render Objects metric in column 3
    - Render Damage metric in column 4 with delta color based on damage_detected
    - Add damage status indicator (green/red) based on damage_detected flag
    - Accept st.empty() placeholder for efficient updates
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 6.1, 6.2, 6.3_

  - [x] 8.2 Write property test for visual feedback consistency
    - **Property 5: Visual Feedback Consistency**
    - Generate random damage_detected states
    - Verify indicator color matches state (True=red, False=green)
    - **Validates: Requirements 6.1, 6.2, 6.4**

- [x] 9. Implement video feed component
  - [x] 9.1 Implement render_video_feed() function in app.py
    - Create centered container for video display
    - Accept st.empty() placeholder for frame updates
    - Convert OpenCV BGR frame to RGB for Streamlit display
    - Use st.image() to display frame within placeholder
    - Handle None frame by showing placeholder image or message
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [x] 10. Implement detection log component
  - [x] 10.1 Implement render_detection_log() function in app.py
    - Create expander or container in bottom row
    - Convert detection_log list to pandas DataFrame
    - Display columns: Timestamp, Wagon ID, Damage Type, Confidence
    - Use st.dataframe() for scrollable table display
    - Accept st.empty() placeholder for efficient updates
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 11. Checkpoint - UI components complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 12. Implement main dashboard orchestrator
  - [x] 12.1 Implement MissionControlDashboard class in app.py
    - Implement __init__ with page config (wide layout, dark theme)
    - Initialize session state with default values
    - Call inject_css() for custom styling
    - Create VideoManager and MetricsCalculator instances
    - _Requirements: 1.1, 1.2, 1.3_

  - [x] 12.2 Implement main processing loop in run() method
    - Create st.empty() placeholders for metrics, video, and log
    - Implement while loop that runs when is_running=True
    - Call video_manager.read_frame() with frame-skipping
    - Update metrics using MetricsCalculator
    - Render metrics row using placeholder.container()
    - Render video feed using placeholder
    - Render detection log using placeholder
    - Check for Stop button press to exit loop
    - Call video_manager.release() on exit
    - _Requirements: 3.6, 4.2, 4.4, 7.4_

  - [x] 12.3 Write property test for UI update isolation
    - **Property 6: UI Update Isolation**
    - Verify only placeholder containers are updated during loop
    - Verify sidebar and header do not re-render
    - **Validates: Requirements 3.6, 4.4, 7.4**

- [x] 13. Integrate with existing pipeline
  - [x] 13.1 Connect dashboard to wagon inspection pipeline
    - Import WagonDetector, DamageDetector from pipelines
    - Process frames through detection pipeline
    - Update object_count from wagon detections
    - Update damage_count and damage_detected from damage detections
    - Append DetectionLogEntry for each damage detection
    - Draw bounding boxes on frame before display
    - _Requirements: 3.4, 3.5, 4.3, 5.5, 6.1, 6.2_

- [x] 14. Create main entry point
  - [x] 14.1 Create run_dashboard.py entry point
    - Import MissionControlDashboard
    - Create and run dashboard instance
    - Add command-line argument parsing for default video source
    - _Requirements: 1.1_

- [x] 15. Final checkpoint - Full dashboard integration
  - Ensure all tests pass, ask the user if questions arise.
  - Run dashboard with test video file
  - Verify metrics update correctly
  - Verify video displays without flickering
  - Verify detection log populates
  - Verify Start/Stop controls work correctly

## Notes

- All tasks are required for comprehensive correctness validation
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties using hypothesis library
- The dashboard integrates with the existing railway wagon inspection pipeline
- Frame-skipping (N=3) maintains UI responsiveness
- st.empty() placeholders prevent sidebar/header flickering
