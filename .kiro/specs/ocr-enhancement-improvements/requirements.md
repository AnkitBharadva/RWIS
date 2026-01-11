# Requirements Document

## Introduction

This document specifies the requirements for enhancing the Railway Wagon Inspection Pipeline with improved OCR capabilities using EasyOCR, self-calibrating image processing thresholds, user-configurable illumination controls, and frontend deblur options. The enhancements aim to improve OCR accuracy, provide adaptive image processing, and give users more control over the processing pipeline.

## Glossary

- **EasyOCR**: Open-source OCR library supporting multiple languages with GPU acceleration
- **OCR_Pipeline**: The text extraction subsystem (replacing PaddleOCR with EasyOCR)
- **Self_Calibration**: Automatic threshold adjustment based on image statistics
- **Illumination_Controller**: Module for adjusting image brightness/contrast
- **Blur_Threshold**: Configurable threshold for blur detection decision
- **Deblur_Threshold**: Configurable threshold for deblur intensity
- **Illumination_Threshold**: Configurable threshold for low-light detection
- **Gamma_Value**: Configurable gamma correction value for illumination adjustment
- **Auto_Calibration_Mode**: Mode where thresholds are automatically computed from image statistics
- **Manual_Mode**: Mode where user-specified thresholds override auto-calibration
- **Deblur_Toggle**: Frontend control to enable/disable deblurring
- **Deblur_Status_Indicator**: Visual indicator showing if deblur is being applied

## Requirements

### Requirement 1: EasyOCR Integration

**User Story:** As a railway inspector, I want the system to use EasyOCR instead of PaddleOCR, so that I can benefit from improved text recognition and easier installation.

#### Acceptance Criteria

1. THE OCR_Pipeline SHALL use EasyOCR library for text extraction
2. WHEN initializing EasyOCR, THE OCR_Pipeline SHALL support GPU acceleration when available
3. WHEN GPU is not available, THE OCR_Pipeline SHALL fall back to CPU processing
4. THE OCR_Pipeline SHALL support configurable language selection (default: English)
5. WHEN extracting text, THE OCR_Pipeline SHALL return text, confidence score, and bounding box
6. THE OCR_Pipeline SHALL handle empty or invalid ROI inputs gracefully
7. WHEN EasyOCR initialization fails, THE OCR_Pipeline SHALL report a clear error message

### Requirement 2: Deblur Implementation Verification

**User Story:** As a system operator, I want to verify that deblurring is being applied correctly, so that I can ensure the pipeline is working as expected.

#### Acceptance Criteria

1. WHEN deblurring is applied, THE Pipeline SHALL log the deblur operation with frame index and wagon ID
2. THE Pipeline SHALL provide a visual indicator in the frontend showing deblur status
3. WHEN deblur is applied, THE Deblur_Status_Indicator SHALL display "DEBLUR ACTIVE" with a distinct color
4. WHEN deblur is skipped, THE Deblur_Status_Indicator SHALL display "DEBLUR SKIPPED" with a different color
5. THE Pipeline SHALL log blur score before and after deblurring for verification
6. THE Pipeline SHALL expose a method to check if deblurring was applied to the current frame

### Requirement 3: Self-Calibrating Blur Thresholds

**User Story:** As a system operator, I want blur thresholds to self-calibrate based on video characteristics, so that the system adapts to different video quality conditions automatically.

#### Acceptance Criteria

1. THE Blur_Detector SHALL support an auto-calibration mode for blur thresholds
2. WHEN auto-calibration is enabled, THE Blur_Detector SHALL compute optimal thresholds from initial frames
3. THE auto-calibration SHALL analyze blur score distribution across a configurable sample of frames
4. THE auto-calibration SHALL set blur_threshold based on statistical analysis (e.g., percentile-based)
5. THE Pipeline SHALL allow switching between auto-calibration and manual threshold modes
6. WHEN manual mode is selected, THE Pipeline SHALL use user-specified blur thresholds
7. THE auto-calibrated thresholds SHALL be logged for user review

### Requirement 4: Self-Calibrating Illumination Thresholds

**User Story:** As a system operator, I want illumination thresholds to self-calibrate based on video lighting conditions, so that the system adapts to different lighting environments automatically.

#### Acceptance Criteria

1. THE Illumination_Controller SHALL support an auto-calibration mode for illumination thresholds
2. WHEN auto-calibration is enabled, THE Illumination_Controller SHALL compute optimal thresholds from initial frames
3. THE auto-calibration SHALL analyze mean luminance distribution across a configurable sample of frames
4. THE auto-calibration SHALL set low_light_threshold based on statistical analysis
5. THE auto-calibration SHALL compute optimal gamma_value based on luminance statistics
6. THE Pipeline SHALL allow switching between auto-calibration and manual illumination modes
7. WHEN manual mode is selected, THE Pipeline SHALL use user-specified illumination thresholds

### Requirement 5: User-Configurable Illumination Controls

**User Story:** As a system operator, I want to manually adjust illumination settings, so that I can fine-tune image brightness for specific conditions.

#### Acceptance Criteria

1. THE Pipeline SHALL expose illumination controls in the frontend sidebar
2. THE frontend SHALL provide a slider for adjusting gamma_value (range: 0.5 to 3.0)
3. THE frontend SHALL provide a slider for adjusting low_light_threshold (range: 0 to 255)
4. WHEN user adjusts illumination settings, THE Pipeline SHALL apply changes in real-time
5. THE frontend SHALL display current illumination status (normal/low-light)
6. THE frontend SHALL provide a toggle to switch between auto and manual illumination modes
7. THE Pipeline SHALL persist user illumination settings across sessions via config

### Requirement 6: User-Configurable Blur/Deblur Controls

**User Story:** As a system operator, I want to manually adjust blur detection thresholds, so that I can fine-tune deblur behavior for specific video conditions.

#### Acceptance Criteria

1. THE Pipeline SHALL expose blur/deblur controls in the frontend sidebar
2. THE frontend SHALL provide a slider for adjusting blur_threshold (range: 0 to 1000)
3. THE frontend SHALL provide a toggle to enable/disable deblurring entirely
4. WHEN user adjusts blur settings, THE Pipeline SHALL apply changes in real-time
5. THE frontend SHALL display current blur score of the processed frame
6. THE frontend SHALL provide a toggle to switch between auto and manual blur threshold modes
7. THE Pipeline SHALL persist user blur settings across sessions via config

### Requirement 7: Frontend Deblur Toggle and Status

**User Story:** As a system operator, I want to enable/disable deblurring from the frontend and see its status, so that I can control processing and verify it's working.

#### Acceptance Criteria

1. THE frontend SHALL provide a toggle button to enable/disable deblurring
2. WHEN deblur is disabled, THE Pipeline SHALL skip all deblurring operations
3. WHEN deblur is enabled, THE Pipeline SHALL apply deblurring based on blur thresholds
4. THE frontend SHALL display a real-time deblur status indicator
5. THE deblur status indicator SHALL show: "ACTIVE" (green), "SKIPPED" (yellow), or "DISABLED" (gray)
6. THE frontend SHALL display the current blur score alongside the status indicator
7. WHEN deblur is applied, THE frontend SHALL optionally show before/after comparison

### Requirement 8: Illumination Increase/Decrease Controls

**User Story:** As a system operator, I want to increase or decrease image illumination manually, so that I can improve visibility in dark or overexposed conditions.

#### Acceptance Criteria

1. THE frontend SHALL provide controls to increase illumination (brighten image)
2. THE frontend SHALL provide controls to decrease illumination (darken image)
3. THE illumination adjustment SHALL use gamma correction for natural-looking results
4. WHEN increasing illumination, THE Pipeline SHALL apply gamma < 1.0 to brighten
5. WHEN decreasing illumination, THE Pipeline SHALL apply gamma > 1.0 to darken
6. THE frontend SHALL display current gamma value being applied
7. THE illumination controls SHALL have a "Reset to Auto" button to restore auto-calibration

### Requirement 9: Configuration Persistence

**User Story:** As a system operator, I want my threshold settings to be saved, so that I don't have to reconfigure them each time I start the system.

#### Acceptance Criteria

1. THE Pipeline SHALL save user-configured thresholds to config.py or a separate settings file
2. WHEN the system starts, THE Pipeline SHALL load previously saved threshold settings
3. THE Pipeline SHALL support both auto-calibration and manual modes in saved configuration
4. THE configuration SHALL include: blur_threshold, low_light_threshold, gamma_value, deblur_enabled
5. THE configuration SHALL include: auto_calibration_blur, auto_calibration_illumination flags
6. THE frontend SHALL provide a "Save Settings" button to persist current configuration
7. THE frontend SHALL provide a "Reset to Defaults" button to restore default settings

### Requirement 10: Calibration Status Display

**User Story:** As a system operator, I want to see the current calibration status and computed thresholds, so that I can understand how the system is adapting to video conditions.

#### Acceptance Criteria

1. THE frontend SHALL display current calibration mode (Auto/Manual) for blur
2. THE frontend SHALL display current calibration mode (Auto/Manual) for illumination
3. WHEN auto-calibration is active, THE frontend SHALL display computed threshold values
4. THE frontend SHALL display calibration progress during initial frame analysis
5. THE frontend SHALL provide a "Recalibrate" button to trigger new auto-calibration
6. THE calibration status SHALL update in real-time as thresholds are adjusted

