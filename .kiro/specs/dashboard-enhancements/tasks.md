# Implementation Plan: Dashboard Enhancements

## Overview

This implementation plan breaks down the dashboard enhancements into discrete, incremental coding tasks. Each task builds on previous work, ensuring no orphaned code. The implementation extends the existing Streamlit dashboard with dual video display, wagon tracking visualization, enhanced metrics, frame saving, and OCR logging.

## Tasks

- [x] 1. Extend data models for enhanced dashboard
  - [x] 1.1 Add new data models to dashboard/models.py
    - Add ProcessingType enum (DEBLUR, CLAHE, GAMMA, OCR)
    - Add OCRLogEntry dataclass with timestamp, wagon_id, extracted_text, confidence, frame_index
    - Add FrameSaveConfig dataclass with enabled, save_on_deblur, save_on_illumination, save_on_ocr, output_directory
    - Add FrameMetadata dataclass with timestamp, frame_index, processing_applied, wagon_id
    - Add EnhancedMetrics dataclass extending existing metrics with total_wagon_count
    - _Requirements: 3.1, 5.4, 5.5, 6.2_

  - [x] 1.2 Write property test for OCR log entry completeness
    - **Property 6: OCR Log Entry Completeness**
    - Generate random OCR results and verify all required fields present
    - **Validates: Requirements 6.2, 6.4, 6.6**

- [x] 2. Implement Track ID renderer
  - [x] 2.1 Create TrackIDRenderer class in dashboard/track_renderer.py
    - Implement draw_track_ids() that overlays Track_IDs on frame
    - Use contrasting cyan color (255, 255, 0) with black background
    - Position labels near top of bounding box
    - Handle multiple wagons in single frame
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [x] 2.2 Write property test for Track ID overlay completeness
    - **Property 2: Track ID Overlay Completeness **
    - Generate random sets of tracked wagons
    - Verify all Track_IDs are rendered on frame
    - **Validates: Requirements 2.1, 2.2, 2.4, 2.5**

- [x] 3. Implement dual video display component
  - [x] 3.1 Create DualVideoDisplay class in dashboard/dual_display.py
    - Implement render() with side-by-side st.columns layout
    - Left column shows raw frame with "Raw Input" label
    - Right column shows processed frame with "Processed Output" label
    - Implement render_placeholders() for stopped state
    - Track frame_index for synchronization
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6_

  - [x] 3.2 Write property test for dual frame synchronization
    - **Property 1: Dual Frame Synchronization**
    - Generate random frame pairs with indices
    - Verify both displays show same frame index
    - **Validates: Requirements 1.2, 1.3, 1.4**

- [x] 4. Checkpoint - Display components complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Implement enhanced metrics panel
  - [x] 5.1 Update MetricsCalculator in dashboard/metrics.py
    - Add latency smoothing with configurable window size
    - Add get_smoothed_latency() method
    - Add LATENCY_WARNING_THRESHOLD_MS constant (100ms)
    - Add is_latency_warning() method
    - _Requirements: 4.2, 4.3, 4.4, 4.5_

  - [x] 5.2 Update render_metrics_row() in dashboard/app.py
    - Extend to 5 columns: FPS, Latency, Objects, Total Wagons, Damage
    - Add total_wagon_count parameter
    - Add latency warning visual indicator (red text when exceeded)
    - Add tooltips for each metric
    - _Requirements: 3.1, 3.2, 3.3, 4.1, 7.1, 7.2, 7.5_

  - [x] 5.3 Write property test for latency measurement accuracy
    - **Property 4: Latency Measurement Accuracy**
    - Generate random latency sequences
    - Verify smoothing calculation and threshold detection
    - **Validates: Requirements 4.2, 4.3, 4.4, 4.5**

  - [x] 5.4 Write property test for wagon counting correctness
    - **Property 3: Wagon Counting Correctness**
    - Generate random sequences of wagon crossings
    - Verify count equals unique crossings, no duplicates
    - **Validates: Requirements 3.2, 3.3, 3.4, 3.5**

- [x] 6. Implement frame saver component
  - [x] 6.1 Create FrameSaver class in dashboard/frame_saver.py
    - Implement __init__ with FrameSaveConfig
    - Implement should_save() checking config flags against processing types
    - Implement save_frame() that writes frame to disk with timestamp filename
    - Implement _generate_filename() with format: {timestamp}_{frame_idx}_{processing}.jpg
    - Implement _save_metadata() writing JSON sidecar file
    - Create output directory if not exists
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_

  - [x] 6.2 Write property test for conditional frame saving
    - **Property 5: Conditional Frame Saving**
    - Generate random processing events and configs
    - Verify correct save/no-save decisions
    - **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 8.4**

- [x] 7. Implement OCR log display component
  - [x] 7.1 Create OCRLogDisplay class in dashboard/ocr_log.py
    - Implement __init__ with MAX_LOG_ENTRIES limit (500)
    - Implement append_entry() maintaining chronological order
    - Implement render() using st.expander with scrollable dataframe
    - Implement to_dataframe() converting entries to pandas DataFrame
    - Implement trim_old_entries() to enforce max limit
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

- [x] 8. Checkpoint - All new components complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. Update sidebar with frame saving controls
  - [x] 9.1 Add frame saving section to render_sidebar() in dashboard/app.py
    - Add "Frame Saving" subheader
    - Add enable/disable toggle for frame saving
    - Add checkboxes for: save on deblur, save on illumination, save on OCR
    - Add text input for output directory path
    - Store config in session state
    - _Requirements: 8.1, 8.2, 8.3, 8.4_

  - [x] 9.2 Write property test for configuration persistence
    - **Property 7: Configuration Persistence**
    - Generate random config values
    - Verify persistence across simulated reruns
    - **Validates: Requirements 8.5**

- [x] 10. Integrate all components into main dashboard
  - [x] 10.1 Update MissionControlDashboard class in dashboard/app.py
    - Initialize DualVideoDisplay, TrackIDRenderer, FrameSaver, OCRLogDisplay
    - Update _init_session_state() with new state fields
    - Update run() to use dual video display instead of single
    - Integrate TrackIDRenderer into frame processing
    - Add total_wagon_count tracking from tracker
    - _Requirements: 1.1, 2.1, 3.1, 7.2_

  - [x] 10.2 Update frame processing loop in run()
    - Store raw frame before processing
    - Apply TrackIDRenderer to processed frame
    - Call FrameSaver.should_save() and save_frame() when appropriate
    - Append OCR results to OCRLogDisplay
    - Update latency measurement with smoothing
    - _Requirements: 1.4, 2.5, 4.3, 5.1, 5.2, 5.3, 6.4_

  - [x] 10.3 Update render calls in run()
    - Replace render_video_feed() with DualVideoDisplay.render()
    - Update render_metrics_row() with total_wagon_count and latency
    - Add OCRLogDisplay.render() after detection log
    - _Requirements: 1.1, 3.1, 6.1, 7.2_

- [x] 11. Final checkpoint - Full integration complete
  - Ensure all tests pass, ask the user if questions arise.
  - Run dashboard with sample video
  - Verify dual video display shows raw and processed frames
  - Verify Track IDs appear on wagons
  - Verify total wagon count increments correctly
  - Verify latency display with smoothing
  - Verify frame saving creates files when enabled
  - Verify OCR log populates with extractions

## Notes

- All tasks are required for comprehensive correctness validation
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties using hypothesis library
- Unit tests validate specific examples and edge cases
- The implementation extends existing dashboard code without breaking changes

