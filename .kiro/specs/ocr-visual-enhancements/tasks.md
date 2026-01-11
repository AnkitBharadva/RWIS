# Implementation Plan: OCR Visual Enhancements

## Overview

This implementation plan covers the OCR visual enhancements including bounding box visualization, text overlays, frame saving with metadata, OCR interval control, and processing status indicators in the main dashboard.

## Tasks

- [x] 1. Create OCR data models and visualization utilities
  - [x] 1.1 Add OCRDetection dataclass to utils/data_models.py
    - Add text, confidence, bbox, wagon_id, frame_index fields
    - _Requirements: 1.1, 2.1_
  - [x] 1.2 Add OCRFrameMetadata dataclass to utils/data_models.py
    - Add timestamp, frame_index, wagon_id, detections, deblur_applied, illumination_applied fields
    - _Requirements: 3.3_
  - [x] 1.3 Create dashboard/ocr_visualization.py with OCRVisualization class
    - Implement draw_ocr_boxes() with cyan color for OCR boxes
    - Implement draw_text_overlay() with semi-transparent background
    - Implement adjust_coordinates() for ROI-to-frame transformation
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4_
  - [x] 1.4 Write property test for coordinate transformation
    - **Property 2: OCR Coordinate Transformation Correctness**
    - **Validates: Requirements 1.4**

- [x] 2. Implement confidence-based text coloring
   - [x] 2.1 Add confidence threshold constant (0.5) to OCRVisualization
    - _Requirements: 2.5_
  - [x] 2.2 Implement color selection logic in draw_text_overlay()
    - Orange for confidence < 0.5, white for >= 0.5
    - _Requirements: 2.5_
  - [x] 2.3 Write property test for confidence-based color selection
    - **Property 5: Confidence-Based Color Selection**
    - **Validates: Requirements 2.5**

- [x] 3. Checkpoint - Ensure visualization tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Implement OCR frame interval controller
  - [x] 4.1 Create dashboard/ocr_interval_controller.py with OCRIntervalController class
    - Implement should_run_ocr(frame_index) method
    - Implement get_status_text(frame_index) method
    - Default interval = 5, range 1-30
    - _Requirements: 4.2, 4.3, 4.4_
  - [x] 4.2 Write property test for OCR interval execution
    - **Property 6: OCR Frame Interval Execution**
    - **Validates: Requirements 4.4, 4.6**
  - [x] 4.3 Write property test for OCR status text
    - **Property 7: OCR Status Reflects Execution State**
    - **Validates: Requirements 5.2, 5.3**

- [x] 5. Implement OCR frame saver with metadata
  - [x] 5.1 Create dashboard/ocr_frame_saver.py with OCRFrameSaver class
    - Implement save_ocr_frame() method
    - Implement generate_filename() with timestamp, frame_index, wagon_id
    - Create output directory if not exists
    - _Requirements: 3.1, 3.2, 3.4, 3.5_
  - [x] 5.2 Implement JSON metadata saving
    - Save metadata alongside frame with all required fields
    - _Requirements: 3.3, 6.2_
  - [x] 5.3 Write property test for metadata file contents
    - **Property 9: Metadata File Contains Required Fields**
    - **Validates: Requirements 3.3, 6.2**

- [x] 6. Checkpoint - Ensure interval and saver tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Implement processing status indicators
  - [x] 7.1 Create dashboard/processing_indicators.py with ProcessingStatusIndicator class
    - Define status colors: APPLIED (green), NORMAL (gray), SKIPPED (yellow), OFF (red), ERROR (red)
    - Implement render_indicator() method with color coding
    - _Requirements: 7.2, 7.3, 7.5_
  - [x] 7.2 Implement render_metrics_row() for enhanced metrics layout
    - Order: FPS | Latency | Objects | Wagons | Damage | Illumination | Deblur | OCR
    - _Requirements: 8.1, 8.2_
  - [x] 7.3 Write property test for processing indicators
    - **Property 10: Processing Indicators Reflect Current State**
    - **Validates: Requirements 7.2, 7.3, 7.4**

- [x] 8. Implement text truncation utility
  - [x] 8.1 Add truncate_text() function to OCRVisualization
    - Truncate to 50 characters with ellipsis
    - _Requirements: 5.4_
  - [x] 8.2 Write property test for text truncation
    - **Property 8: Text Truncation at 50 Characters**
    - **Validates: Requirements 5.4**

- [x] 9. Checkpoint - Ensure indicator and truncation tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 10. Integrate OCR visualization into dashboard
  - [x] 10.1 Add OCR interval slider to sidebar in dashboard/app.py
    - Slider range 1-30, default 5
    - Display current interval value
    - _Requirements: 4.1, 4.2, 4.3, 4.5_
  - [x] 10.2 Add OCR status section to sidebar
    - Show ACTIVE/SKIPPED status
    - Show last detected text (truncated)
    - Show last confidence
    - _Requirements: 5.1, 5.2, 5.3, 5.4_
  - [x] 10.3 Update _process_frame_through_pipeline() to use OCR interval
    - Check should_run_ocr() before running OCR
    - Update OCR status in session state
    - _Requirements: 4.4, 4.6_
  - [x] 10.4 Integrate OCRVisualization into frame processing
    - Draw OCR bounding boxes on processed frame
    - Draw text overlays with confidence
    - Adjust coordinates from ROI to frame
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 2.5_

- [x] 11. Integrate processing status indicators into metrics row
  - [x] 11.1 Update render_metrics_row() to include new indicators
    - Add Illumination indicator column
    - Add Deblur indicator column
    - Add OCR indicator column
    - _Requirements: 7.1, 8.1_
  - [x] 11.2 Track processing status in session state
    - Add illumination_applied_this_frame flag
    - Add deblur_applied_this_frame flag
    - Add ocr_applied_this_frame flag
    - _Requirements: 7.4_
  - [x] 11.3 Update processing pipeline to set status flags
    - Set illumination flag when CLAHE/gamma applied
    - Set deblur flag when MPRNet applied
    - Set OCR flag when OCR executed
    - _Requirements: 7.2, 7.3, 7.4_

- [x] 12. Integrate OCR frame saving
  - [x] 12.1 Initialize OCRFrameSaver in dashboard
    - Use configurable output directory from session state
    - _Requirements: 3.4_
  - [x] 12.2 Save frames when OCR detects text
    - Save annotated frame with OCR boxes
    - Save JSON metadata with all detection info
    - Include deblur_applied status
    - _Requirements: 3.1, 3.2, 3.3, 6.2_

- [x] 13. Implement settings persistence for OCR interval
  - [x] 13.1 Add ocr_frame_interval to PipelineSettings in utils/settings_manager.py
    - _Requirements: 4.7_
  - [x] 13.2 Update save_settings() and load_settings() to include OCR interval
    - _Requirements: 4.7_
  - [x] 13.3 Write property test for settings persistence
    - **Property 11: Settings Persistence Round-Trip**
    - **Validates: Requirements 4.7**

- [x] 14. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- All tasks are required for comprehensive implementation
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties
- Unit tests validate specific examples and edge cases
