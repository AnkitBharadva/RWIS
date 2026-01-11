# Implementation Plan: OCR and Image Enhancement Improvements

## Overview

This implementation plan covers the migration from PaddleOCR to EasyOCR, self-calibrating thresholds for blur and illumination, user-configurable controls in the frontend, and deblur toggle functionality. The implementation follows an incremental approach, building core components first and then integrating them into the dashboard.

## Tasks

- [x] 1. Replace PaddleOCR with EasyOCR
  - [x] 1.1 Update OCR pipeline to use EasyOCR
    - Replace PaddleOCR imports with EasyOCR
    - Update initialization to create EasyOCR Reader with GPU support
    - Implement CPU fallback when GPU is unavailable
    - Update `extract_text` method to use EasyOCR API
    - Parse EasyOCR results into OCRResult dataclass
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7_

  - [x] 1.2 Write property test for OCR result completeness
    - **Property 1: OCR Result Completeness**
    - **Validates: Requirements 1.5**

  - [x] 1.3 Write unit tests for EasyOCR pipeline
    - Test initialization with GPU enabled/disabled
    - Test language configuration
    - Test empty/invalid ROI handling
    - _Requirements: 1.2, 1.3, 1.6_

- [x] 2. Implement Calibration Manager
  - [x] 2.1 Create CalibrationManager class
    - Create `pipelines/calibration_manager.py`
    - Implement sample collection with configurable sample size
    - Implement blur score computation for samples
    - Implement luminance computation for samples
    - Implement percentile-based threshold calculation
    - Implement calibration progress tracking
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 4.1, 4.2, 4.3, 4.4, 4.5_

  - [x] 2.2 Write property test for blur auto-calibration
    - **Property 4: Blur Auto-Calibration**
    - **Validates: Requirements 3.2, 3.3, 3.4**

  - [x] 2.3 Write property test for illumination auto-calibration
    - **Property 5: Illumination Auto-Calibration**
    - **Validates: Requirements 4.2, 4.3, 4.4, 4.5**

- [x] 3. Implement Illumination Controller
  - [x] 3.1 Create IlluminationController class
    - Create `utils/illumination_controller.py`
    - Implement gamma correction with lookup table caching
    - Implement `increase_illumination` method (gamma < 1)
    - Implement `decrease_illumination` method (gamma > 1)
    - Implement low-light detection based on mean luminance
    - Implement auto/manual mode switching
    - Implement settings get/set methods
    - _Requirements: 5.4, 8.3, 8.4, 8.5_

  - [x] 3.2 Write property test for gamma direction correctness
    - **Property 10: Gamma Direction Correctness**
    - **Validates: Requirements 8.3, 8.4, 8.5**

- [x] 4. Enhance Blur Detector with auto-calibration support
  - [x] 4.1 Add auto/manual mode to BlurDetector
    - Add `auto_mode` flag to BlurDetector
    - Add `set_threshold` and `get_threshold` methods
    - Add `update_from_calibration` method
    - Add `get_settings` and `set_settings` methods
    - _Requirements: 3.5, 3.6, 6.4_

  - [x] 4.2 Write property test for manual mode override
    - **Property 6: Manual Mode Override**
    - **Validates: Requirements 3.6, 4.7**

- [x] 5. Enhance Deblur Manager with enable/disable and status tracking
  - [x] 5.1 Add deblur enable/disable functionality
    - Add `deblur_enabled` flag to DeblurManager
    - Add `set_deblur_enabled` and `is_deblur_enabled` methods
    - Modify `process_roi` to skip deblurring when disabled
    - _Requirements: 7.2, 7.3_

  - [x] 5.2 Add deblur status tracking
    - Create `DeblurStatus` dataclass
    - Track blur score before/after deblurring
    - Implement `get_last_status` method
    - Implement `was_deblur_applied` method
    - Add logging for deblur operations
    - _Requirements: 2.1, 2.5, 2.6_

  - [x] 5.3 Write property test for deblur enable/disable
    - **Property 9: Deblur Enable/Disable**
    - **Validates: Requirements 7.2, 7.3**

  - [x] 5.4 Write property test for deblur operation logging
    - **Property 3: Deblur Operation Logging**
    - **Validates: Requirements 2.1, 2.5**

- [x] 6. Implement Settings Manager
  - [x] 6.1 Create SettingsManager class
    - Create `utils/settings_manager.py`
    - Define `PipelineSettings` dataclass with all configurable fields
    - Implement `load_settings` from JSON file
    - Implement `save_settings` to JSON file
    - Implement `reset_to_defaults` method
    - Implement partial update methods for blur and illumination
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

  - [x] 6.2 Write property test for configuration round-trip
    - **Property 8: Configuration Round-Trip**
    - **Validates: Requirements 5.7, 6.7, 9.1, 9.2, 9.3, 9.4, 9.5**

- [x] 7. Checkpoint - Ensure all core component tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Add frontend illumination controls
  - [x] 8.1 Add illumination controls to sidebar
    - Add "Illumination Settings" section to sidebar
    - Add gamma value slider (range: 0.5 to 3.0)
    - Add low-light threshold slider (range: 0 to 255)
    - Add auto/manual mode toggle
    - Add current illumination status display
    - Add "Reset to Auto" button
    - _Requirements: 5.1, 5.2, 5.3, 5.5, 5.6, 8.1, 8.2, 8.6, 8.7_

- [x] 9. Add frontend blur/deblur controls
  - [x] 9.1 Add blur controls to sidebar
    - Add "Blur/Deblur Settings" section to sidebar
    - Add blur threshold slider (range: 0 to 1000)
    - Add deblur enable/disable toggle
    - Add auto/manual mode toggle
    - Add current blur score display
    - _Requirements: 6.1, 6.2, 6.3, 6.5, 6.6_

  - [x] 9.2 Add deblur status indicator
    - Add status indicator showing ACTIVE/SKIPPED/DISABLED
    - Use color coding: green (ACTIVE), yellow (SKIPPED), gray (DISABLED)
    - Display current blur score alongside status
    - _Requirements: 2.2, 2.3, 2.4, 7.4, 7.5, 7.6_

- [x] 10. Add frontend calibration status display
  - [x] 10.1 Add calibration status section
    - Add "Calibration Status" section to sidebar
    - Display current calibration mode (Auto/Manual) for blur
    - Display current calibration mode (Auto/Manual) for illumination
    - Display computed threshold values when in auto mode
    - Add calibration progress indicator during initial analysis
    - Add "Recalibrate" button
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6_

- [x] 11. Add frontend settings persistence controls
  - [x] 11.1 Add save/reset buttons
    - Add "Save Settings" button to persist current configuration
    - Add "Reset to Defaults" button to restore default settings
    - Display confirmation messages on save/reset
    - _Requirements: 9.6, 9.7_

- [x] 12. Integrate components into main pipeline
  - [x] 12.1 Wire up calibration manager
    - Initialize CalibrationManager in dashboard
    - Feed initial frames to calibration during startup
    - Apply calibrated thresholds to blur detector and illumination controller
    - _Requirements: 3.2, 4.2_

  - [x] 12.2 Wire up settings manager
    - Load settings on dashboard startup
    - Apply loaded settings to all components
    - Save settings when user clicks save button
    - _Requirements: 9.1, 9.2_

  - [x] 12.3 Wire up real-time settings updates
    - Connect slider changes to component settings
    - Apply changes immediately without restart
    - _Requirements: 5.4, 6.4_

- [x] 13. Update requirements.txt
  - [x] 13.1 Add EasyOCR dependency
    - Add `easyocr` to requirements.txt
    - Remove `paddleocr` and `paddlepaddle` dependencies
    - _Requirements: 1.1_

- [x] 14. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- All tasks are required for comprehensive implementation
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties
- Unit tests validate specific examples and edge cases
