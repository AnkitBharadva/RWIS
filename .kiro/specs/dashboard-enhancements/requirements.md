# Requirements Document

## Introduction

This document specifies the requirements for enhancing the Mission Control Streamlit dashboard with additional monitoring capabilities. The enhancements include dual video display (raw input vs processed output), wagon tracking visualization, total wagon counting, real-time latency metrics, frame saving for processed images, and OCR logging display.

## Glossary

- **Dashboard**: The Streamlit-based web interface for monitoring the inspection pipeline
- **Raw_Frame**: The original unprocessed video frame from the input source
- **Output_Frame**: The processed frame with detection overlays, bounding boxes, and annotations
- **Track_ID**: Unique identifier assigned to each wagon by the ByteTrack algorithm
- **Total_Wagon_Count**: Cumulative count of unique wagons that have crossed the counting line
- **Processing_Latency**: Time taken to process a single frame through the pipeline in milliseconds
- **Processed_Frame**: A frame that has had deblur, illumination enhancement, or OCR applied
- **OCR_Log**: Historical record of all OCR text extractions with timestamps and confidence scores

## Requirements

### Requirement 1: Dual Video Frame Display

**User Story:** As a railway inspector, I want to see both the raw input video and the processed output side by side, so that I can compare the original footage with the detection results.

#### Acceptance Criteria

1. THE Dashboard SHALL display two video frames in a side-by-side layout
2. THE left frame SHALL show the raw input video without any processing or overlays
3. THE right frame SHALL show the processed output with detection bounding boxes and annotations
4. WHEN frames are updated, THE Dashboard SHALL synchronize both displays to show the same frame index
5. THE Dashboard SHALL label each frame clearly as "Raw Input" and "Processed Output"
6. WHEN video capture is stopped, THE Dashboard SHALL display placeholder messages in both frames

### Requirement 2: Wagon Track ID Display

**User Story:** As a railway inspector, I want to see the track ID of each detected wagon in the video, so that I can identify and follow individual wagons through the frame.

#### Acceptance Criteria

1. WHEN a wagon is detected, THE Dashboard SHALL display its Track_ID on the video overlay
2. THE Track_ID SHALL be displayed near the wagon's bounding box in a readable font
3. THE Track_ID label SHALL use a contrasting color for visibility against the video background
4. WHEN multiple wagons are visible, THE Dashboard SHALL display Track_IDs for all detected wagons
5. THE Track_ID display SHALL update in real-time as wagons move through the frame

### Requirement 3: Total Wagon Count Display

**User Story:** As a railway inspector, I want to see the total count of wagons detected, so that I can track how many wagons have passed through the inspection point.

#### Acceptance Criteria

1. THE Dashboard SHALL display a Total_Wagon_Count metric in the metrics panel
2. THE Total_Wagon_Count SHALL represent unique wagons that have crossed the counting line
3. WHEN a new wagon crosses the counting line, THE Total_Wagon_Count SHALL increment by exactly one
4. THE Total_Wagon_Count SHALL NOT include duplicate counts for the same wagon
5. THE Total_Wagon_Count SHALL persist across frame updates until the session is reset

### Requirement 4: Real-Time Processing Latency Display

**User Story:** As a system operator, I want to see the real-time processing latency, so that I can monitor system performance and identify bottlenecks.

#### Acceptance Criteria

1. THE Dashboard SHALL display processing latency in milliseconds in the metrics panel
2. THE latency metric SHALL measure the time from frame capture to display completion
3. THE latency display SHALL update with each processed frame
4. THE Dashboard SHALL use smoothed averaging to prevent erratic latency display
5. WHEN latency exceeds a warning threshold, THE Dashboard SHALL provide visual feedback

### Requirement 5: Processed Frame Saving

**User Story:** As a railway inspector, I want frames that have been processed with deblur, illumination enhancement, or OCR to be saved automatically, so that I can review them later for quality assurance.

#### Acceptance Criteria

1. WHEN deblur processing is applied to a frame, THE Dashboard SHALL save that frame to disk
2. WHEN illumination enhancement (CLAHE/gamma) is applied to a frame, THE Dashboard SHALL save that frame to disk
3. WHEN OCR is performed on a frame, THE Dashboard SHALL save that frame to disk
4. THE saved frames SHALL be stored in a configurable output directory with timestamps
5. THE saved frames SHALL include metadata indicating which processing was applied
6. THE Dashboard SHALL provide a toggle to enable/disable automatic frame saving

### Requirement 6: OCR Log Display

**User Story:** As a railway inspector, I want to see a log of all OCR detections in the dashboard, so that I can review extracted wagon identification numbers.

#### Acceptance Criteria

1. THE Dashboard SHALL display an OCR_Log section showing all text extractions
2. THE OCR_Log SHALL display timestamp, wagon ID, extracted text, and confidence score for each entry
3. THE OCR_Log SHALL support scrolling for viewing historical entries
4. WHEN new OCR text is extracted, THE Dashboard SHALL append the entry to the OCR_Log
5. THE OCR_Log SHALL be displayed in a collapsible expander for space efficiency
6. THE OCR_Log SHALL maintain chronological order with newest entries visible

### Requirement 7: Enhanced Metrics Panel Layout

**User Story:** As a railway inspector, I want an organized metrics panel that shows all key performance indicators, so that I can monitor the system at a glance.

#### Acceptance Criteria

1. THE Metrics_Panel SHALL display metrics in a clear, organized layout
2. THE Metrics_Panel SHALL include: FPS, Processing Latency, Objects Detected, Total Wagon Count, Damage Count
3. THE Metrics_Panel SHALL use consistent styling with the existing dashboard theme
4. WHEN metrics are updated, THE Dashboard SHALL refresh only the metric containers without flickering
5. THE Metrics_Panel SHALL provide tooltips explaining each metric

### Requirement 8: Frame Saving Configuration

**User Story:** As a system operator, I want to configure frame saving options, so that I can control storage usage and select which processing types trigger saves.

#### Acceptance Criteria

1. THE Sidebar SHALL contain a toggle for enabling/disabling frame saving
2. THE Sidebar SHALL contain checkboxes for selecting which processing types trigger saves (deblur, illumination, OCR)
3. THE Sidebar SHALL contain an input for configuring the output directory path
4. WHEN frame saving is disabled, THE Dashboard SHALL NOT save any processed frames
5. THE configuration SHALL persist across dashboard sessions via session state

