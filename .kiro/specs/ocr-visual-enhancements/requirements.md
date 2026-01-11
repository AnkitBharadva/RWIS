# Requirements Document

## Introduction

This feature enhances the OCR (Optical Character Recognition) visualization and control in the Railway Wagon Inspection System. It adds visual feedback for OCR detections on video frames, saves OCR frames with metadata, and provides frontend controls for OCR processing frequency.

## Glossary

- **OCR_Pipeline**: The text extraction component that uses EasyOCR to detect and read text from wagon ROIs
- **OCR_Bounding_Box**: A rectangular region drawn around detected text on the video frame
- **OCR_Frame_Interval**: The frequency at which OCR is performed (every Nth frame)
- **OCR_Metadata**: Information about OCR results including extracted text, confidence, position, and timestamp
- **Processed_Frame**: A video frame with OCR bounding boxes and text annotations overlaid
- **Frame_Saver**: Component responsible for saving frames with metadata to disk

## Requirements

### Requirement 1: OCR Bounding Box Visualization

**User Story:** As an operator, I want to see bounding boxes around detected text on the video feed, so that I can visually verify what text the system is detecting.

#### Acceptance Criteria

1. WHEN OCR detects text in a wagon ROI, THE System SHALL draw a bounding box around each detected text region
2. THE OCR_Bounding_Box SHALL be displayed in a distinct color (cyan/blue) to differentiate from wagon (green) and damage (red) boxes
3. WHEN multiple text regions are detected, THE System SHALL draw separate bounding boxes for each region
4. THE OCR_Bounding_Box coordinates SHALL be adjusted from ROI-relative to frame-absolute coordinates
5. IF no text is detected, THEN THE System SHALL not draw any OCR bounding boxes

### Requirement 2: OCR Text Overlay on Frame

**User Story:** As an operator, I want to see the detected text displayed on the video frame near the detection, so that I can immediately read what was found without checking the log.

#### Acceptance Criteria

1. WHEN OCR detects text, THE System SHALL display the extracted text as an overlay near the bounding box
2. THE text overlay SHALL include the confidence score in percentage format
3. THE text overlay SHALL have a semi-transparent background for readability
4. THE text overlay SHALL be positioned above or below the bounding box to avoid obscuring the detected region
5. WHEN text confidence is below 50%, THE System SHALL display the text in a warning color (yellow/orange)

### Requirement 3: OCR Frame Saving with Metadata

**User Story:** As a system administrator, I want OCR frames to be saved with complete metadata, so that I can review detections offline and maintain records.

#### Acceptance Criteria

1. WHEN OCR is performed and text is detected, THE Frame_Saver SHALL save the annotated frame to disk
2. THE saved frame filename SHALL include timestamp, frame index, and wagon ID
3. THE System SHALL save a JSON metadata file alongside each saved frame containing:
   - Timestamp (ISO format)
   - Frame index
   - Wagon ID
   - All detected text strings
   - Confidence scores for each detection
   - Bounding box coordinates for each detection
   - Deblur status (whether deblurring was applied)
4. THE saved frames SHALL be stored in a configurable output directory
5. IF the output directory does not exist, THEN THE System SHALL create it automatically

### Requirement 4: OCR Frame Interval Control

**User Story:** As an operator, I want to control how often OCR is performed, so that I can balance between detection coverage and system performance.

#### Acceptance Criteria

1. THE sidebar SHALL display an "OCR Frame Interval" slider control
2. THE slider SHALL allow values from 1 to 30 (run OCR every 1st to 30th frame)
3. THE default OCR frame interval SHALL be 5 (run OCR every 5th frame)
4. WHEN the interval is set to N, THE System SHALL only perform OCR on every Nth frame
5. THE current OCR interval setting SHALL be displayed in the sidebar
6. WHEN OCR is skipped due to interval, THE System SHALL not draw OCR bounding boxes on that frame
7. THE OCR interval setting SHALL persist across session restarts

### Requirement 5: OCR Status Display

**User Story:** As an operator, I want to see the current OCR status, so that I know when OCR is being performed and what was last detected.

#### Acceptance Criteria

1. THE sidebar SHALL display an "OCR Status" section showing:
   - Whether OCR is currently active or skipped (based on frame interval)
   - The last detected text (if any)
   - The confidence of the last detection
2. WHEN OCR is performed, THE status SHALL update to show "ACTIVE"
3. WHEN OCR is skipped due to frame interval, THE status SHALL show "SKIPPED (frame N of M)"
4. THE last detected text SHALL be truncated to 50 characters with ellipsis if longer

### Requirement 6: Deblur Integration with OCR

**User Story:** As an operator, I want deblurring to be applied before OCR when enabled, so that text detection accuracy is improved on blurry frames.

#### Acceptance Criteria

1. WHEN "Enable Deblurring" is ON and a wagon ROI is blurry, THE System SHALL apply deblurring before OCR
2. THE deblur status SHALL be included in the saved frame metadata
3. WHEN deblurring is applied, THE OCR_Bounding_Box color SHALL change to indicate enhanced processing (e.g., bright cyan vs regular cyan)
4. THE System SHALL log whether deblurring improved OCR confidence (before/after comparison)

### Requirement 7: Processing Status Indicators in Main Dashboard

**User Story:** As an operator, I want to see real-time indicators showing whether illumination enhancement and deblurring are being applied, so that I can monitor the processing pipeline status at a glance.

#### Acceptance Criteria

1. THE main dashboard metrics row SHALL display processing status indicators alongside FPS, Latency, Objects, and Damage metrics
2. THE System SHALL display an "Illumination" indicator showing:
   - "APPLIED" (green) when CLAHE/gamma correction is actively enhancing the current frame
   - "NORMAL" (gray) when no illumination enhancement is needed
   - "OFF" (red) when illumination enhancement is disabled
3. THE System SHALL display a "Deblur" indicator showing:
   - "APPLIED" (green) when MPRNet deblurring is actively processing the current frame
   - "SKIPPED" (yellow) when frame is sharp enough (no deblur needed)
   - "OFF" (red) when deblurring is disabled
4. THE indicators SHALL update in real-time as each frame is processed
5. THE indicators SHALL be visually distinct with color-coded backgrounds matching their status
6. WHEN hovering over an indicator, THE System SHALL display a tooltip with additional details (e.g., blur score, luminance level)

### Requirement 8: Enhanced Metrics Row Layout

**User Story:** As an operator, I want all key metrics and processing status visible in a single row, so that I can monitor system performance without scrolling.

#### Acceptance Criteria

1. THE metrics row SHALL display in order: FPS | Latency | Objects | Total Wagons | Damage | Illumination | Deblur | OCR
2. EACH metric/indicator SHALL have a consistent card-style appearance
3. THE layout SHALL be responsive and adjust to screen width
4. THE OCR indicator SHALL show "ACTIVE" when OCR is running on current frame, "SKIPPED" when interval-skipped
5. IF any processing component fails, THE corresponding indicator SHALL show "ERROR" in red
