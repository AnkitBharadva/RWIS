# Implementation Plan: Railway Wagon Inspection Pipeline

## Overview

This implementation plan breaks down the railway wagon inspection pipeline into discrete, incremental coding tasks. Each task builds on previous work, ensuring no orphaned code. The implementation uses Python 3.10 with PyTorch, Ultralytics YOLO, PaddleOCR, MPRNet, and OpenCV.

Key changes from previous implementation:
- Replace NAFNet with MPRNet (GoPro-trained Deblurring variant)
- Add aggressive ROI resizing (max 256px width)
- Add N-th frame execution with ROI caching
- Add FP16 inference with FP32 fallback
- Simplify blur gating to single threshold

## Tasks

- [x] 1. Set up project structure and configuration
  - [x] 1.1 Create project directory structure and requirements.txt
    - Create all directories: models/, pipelines/, tracking/, utils/, outputs/, outputs/debug_frames/
    - Create requirements.txt with: ultralytics, paddleocr, paddlepaddle-gpu, torch, torchvision, opencv-python>=4.9, numpy, hypothesis, pytest
    - _Requirements: 10.2_

  - [x] 1.2 Implement config.py with PipelineConfig dataclass
    - Define PipelineConfig with all configurable parameters
    - Include blur_threshold (single threshold), max_roi_width (256), deblur_frame_interval (3-5)
    - Include use_fp16, fp32_fallback settings
    - Include mprnet_model_path instead of nafnet_model_path
    - Implement load_config() function
    - _Requirements: 1.5, 5.4, 6.4, 7.4, 10.1, 10.6_

  - [x] 1.3 Write property test for configuration loading
    - **Property 15: Configuration Loading**
    - Generate random valid configurations and verify all values are loaded correctly
    - **Validates: Requirements 1.5, 5.4, 6.4, 7.4, 10.1, 10.6**

- [x] 2. Implement data models and enums
  - [x] 2.1 Create data models in utils/data_models.py
    - Implement DamageClass enum (DOOR_DAMAGE, FLOOR_DAMAGE, STRUCTURAL_CRACK, DEFORMATION)
    - Implement BoundingBox dataclass with width, height, center properties
    - Implement WagonDetection, TrackedWagon, DamageDetection, OCRResult dataclasses
    - Implement DeblurResult dataclass with deblur_applied, source_frame, blur_score
    - Implement WagonRecord dataclass with deblur_applied, deblur_source_frame fields
    - _Requirements: 3.3, 3.4, 3.5, 4.6, 8.1, 8.2, 8.3, 8.4, 8.7_

- [x] 3. Implement blur detection module
  - [x] 3.1 Implement BlurDetector class in pipelines/blur_detector.py
    - Implement __init__ with single threshold parameter
    - Implement compute_blur_score() using Laplacian variance on ROI
    - Implement should_deblur() returning bool (True if blur_score < threshold)
    - _Requirements: 1.2, 1.3, 1.4, 1.6_

  - [x] 3.2 Write property test for blur decision logic
    - **Property 1: Blur Decision Logic Consistency**
    - Generate random blur_score values and thresholds
    - Verify decision matches expected logic (score < threshold = deblur)
    - **Validates: Requirements 1.3, 1.4**

  - [x] 3.3 Write property test for blur score determinism
    - **Property 2: Blur Score Computation Determinism**
    - Generate random ROI images and verify computing blur_score twice produces identical results
    - **Validates: Requirements 1.2**

- [x] 4. Implement CLAHE enhancement utility
  - [x] 4.1 Implement CLAHEEnhancer class in utils/clahe.py
    - Implement __init__ with clip_limit and tile_grid_size parameters
    - Implement enhance() that applies CLAHE to L-channel only in LAB color space
    - Convert BGR to LAB, apply CLAHE to L, convert back to BGR
    - _Requirements: 2.4, 11.5_

  - [x] 4.2 Write property test for CLAHE L-channel isolation
    - **Property 4: CLAHE L-Channel Isolation**
    - Generate random BGR images
    - Apply CLAHE and verify A/B channels remain unchanged
    - **Validates: Requirements 2.4, 11.5**

- [x] 5. Implement ROI utilities with resizing
  - [x] 5.1 Implement ROI utility functions in utils/roi_utils.py
    - Implement extract_roi() that clips ROI to frame boundaries
    - Implement resize_roi_for_deblur() that resizes to max_width maintaining aspect ratio
    - Implement validate_roi() that checks ROI validity
    - Ensure no upscaling if ROI already smaller than max_width
    - _Requirements: 2.8, 5.1, 5.2, 5.3, 5.5_

  - [x] 5.2 Write property test for valid ROI generation
    - **Property 7: Valid ROI Generation**
    - Generate random detections within frame bounds
    - Verify all ROIs have valid coordinates (x1 < x2, y1 < y2, within bounds)
    - **Validates: Requirements 2.8**

  - [x] 5.3 Write property test for ROI resizing constraints
    - **Property 8: ROI Resizing Constraints**
    - Generate random ROIs of various sizes
    - Verify resize behavior: width <= max_width, aspect ratio preserved, no upscaling
    - **Validates: Requirements 5.1, 5.2, 5.3, 5.5**

- [x] 6. Checkpoint - Core utilities complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Implement wagon detector
  - [x] 7.1 Implement WagonDetector class in pipelines/wagon_detector.py
    - Implement __init__ that loads YOLOv11n model from model_path
    - Implement detect() that runs inference and returns List[WagonDetection]
    - Implement extract_text_region_roi() that extracts text region ROI from wagon
    - Add input validation to ensure frame is not deblurred
    - _Requirements: 2.1, 2.2, 2.3, 11.3_

- [x] 8. Implement wagon tracker
  - [x] 8.1 Implement WagonTracker class in tracking/tracker.py
    - Implement __init__ with counting_line_y parameter
    - Implement update() using ByteTrack algorithm for tracking
    - Implement line-crossing detection logic
    - Implement get_wagon_count() returning total crossed wagons
    - Implement has_crossed_line() to check if wagon already counted
    - Implement on_wagon_exit() callback for cache cleanup
    - Maintain set of crossed wagon IDs to prevent double-counting
    - _Requirements: 2.5, 2.6, 2.7_

  - [x] 8.2 Write property test for wagon counting accuracy
    - **Property 5: Wagon Counting Accuracy**
    - Generate sequences of wagon positions crossing a line
    - Verify count equals unique wagons that crossed, no double-counts
    - **Validates: Requirements 2.6, 2.7**

  - [x] 8.3 Write property test for unique wagon IDs
    - **Property 6: Unique Wagon ID Assignment**
    - Generate random detection sequences
    - Verify all assigned track IDs are unique
    - **Validates: Requirements 2.5**

- [x] 9. Implement damage detector
  - [x] 9.1 Implement DamageDetector class in pipelines/damage_detector.py
    - Implement __init__ that loads separate YOLOv11n damage model
    - Implement detect() that analyzes wagon ROI for damage
    - Return List[DamageDetection] with damage_class, bbox, confidence, wagon_id
    - Validate damage_class is within expected enum values
    - Ensure input is RAW or CLAHE only (not deblurred)
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

  - [x] 9.2 Write property test for damage detection output completeness
    - **Property 11: Damage Detection Output Completeness**
    - Generate random damage detections
    - Verify all required fields present and valid (damage_class, bbox, confidence, wagon_id)
    - **Validates: Requirements 3.4, 3.5**

- [x] 10. Checkpoint - Detection modules complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 11. Implement MPRNet deblur wrapper
  - [x] 11.1 Implement MPRNetDeblur class in pipelines/mprnet_wrapper.py
    - Implement __init__ that loads MPRNet GoPro-trained deblurring model
    - Load model with FP16 weights if use_fp16=True
    - Implement deblur_roi() that processes only small ROI images
    - Add dimension validation to reject full-frame inputs
    - Implement _check_numerical_stability() for NaN/Inf detection
    - Implement FP32 fallback when instability detected
    - Ensure batch size is always 1, no tiling
    - _Requirements: 4.3, 4.4, 4.7, 7.1, 7.2, 7.3, 9.2, 9.3, 11.1, 11.6_

  - [x] 11.2 Write property test for deblur ROI-only constraint
    - **Property 10: Deblur ROI-Only Constraint**
    - Generate random ROI and full-frame dimensions
    - Verify MPRNet only accepts ROI-sized inputs (width <= max_roi_width), rejects full frames
    - **Validates: Requirements 4.4, 4.7, 11.1, 11.6**

- [x] 12. Implement Deblur Manager with N-th frame execution
  - [x] 12.1 Implement DeblurManager class in pipelines/deblur_manager.py
    - Implement __init__ with mprnet, blur_detector, frame_interval, max_roi_width
    - Implement process_roi() with conditional deblurring logic:
      1. Resize ROI to max_width
      2. Compute blur score
      3. If blur_score >= threshold: return raw ROI
      4. If blur_score < threshold: check N-th frame condition
      5. If frame_count % N == 0: run MPRNet, cache result
      6. Else: return cached result
    - Implement get_cached_roi() for cache retrieval
    - Implement clear_cache() for wagon exit cleanup
    - _Requirements: 5.1, 5.2, 5.3, 6.1, 6.2, 6.3, 6.5_

  - [x] 12.2 Write property test for N-th frame execution logic
    - **Property 9: N-th Frame Execution Logic**
    - Generate random frame sequences with various intervals
    - Verify MPRNet runs at correct intervals, cache is reused correctly
    - **Validates: Requirements 6.1, 6.2, 6.3**

- [x] 13. Implement OCR pipeline
  - [x] 13.1 Implement OCRPipeline class in pipelines/ocr_pipeline.py
    - Implement __init__ that initializes PaddleOCR with GPU support
    - Implement extract_text() that processes ROI (raw or deblurred)
    - Implement _apply_adaptive_gamma() for low-light ROIs
    - Return OCRResult with text and confidence
    - Support partial OCR results for multi-frame voting
    - _Requirements: 4.1, 4.2, 4.5, 4.6, 4.8, 4.9, 4.10_

  - [x] 13.2 Write property test for OCR output completeness
    - **Property 12: OCR Output Completeness**
    - Generate random OCR results
    - Verify text field exists and confidence is in [0.0, 1.0]
    - **Validates: Requirements 4.6**

- [x] 14. Implement logging module
  - [x] 14.1 Implement InspectionLogger class in utils/logger.py
    - Implement __init__ with output_dir, formats list, enable_debug flag
    - Implement log_wagon() that writes WagonRecord to CSV and JSON
    - Include deblur_applied and deblur_source_frame in log output
    - Implement save_debug_frame() that saves annotated frames when enabled
    - Implement flush() to ensure all data written to disk
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7_

  - [x] 14.2 Write property test for log record completeness
    - **Property 13: Log Record Completeness**
    - Generate random wagon records
    - Verify all required fields present including deblur metadata
    - **Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.7**

  - [x] 14.3 Write property test for dual format logging consistency
    - **Property 14: Dual Format Logging Consistency**
    - Generate random wagon records, log to both formats
    - Parse CSV and JSON, verify they produce equivalent records
    - **Validates: Requirements 8.5**

  - [x] 14.4 Write property test for debug frame toggle
    - **Property 16: Debug Frame Toggle**
    - Test with enable_debug_frames True and False
    - Verify frames are created/not created accordingly
    - **Validates: Requirements 10.5**

- [x] 15. Checkpoint - All modules complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 16. Implement main pipeline controller
  - [x] 16.1 Implement main pipeline in main.py
    - Implement video capture loop with frame rate handling
    - Wire together all pipeline stages in correct order:
      1. Frame capture
      2. Optional CLAHE enhancement
      3. Wagon detection (RAW/CLAHE only)
      4. Wagon tracking and counting
      5. ROI extraction and resizing
      6. Blur detection on ROI
      7. Conditional deblurring via DeblurManager
      8. OCR on processed ROI
      9. Damage detection on RAW wagon ROI
      10. Logging results with deblur metadata
    - Register wagon exit callback for cache cleanup
    - Implement graceful shutdown and resource cleanup
    - _Requirements: 1.1, 9.6, 9.7_

  - [x] 16.2 Implement GPU memory management
    - Add memory monitoring during inference
    - Ensure batch size stays at 1 for MPRNet
    - Implement operation queuing for memory safety
    - Log FP32 fallback occurrences
    - _Requirements: 7.5, 9.1, 9.8_

  - [x] 16.3 Implement error handling throughout pipeline
    - Add try/catch blocks for model loading errors
    - Add video source error handling with retry logic
    - Add runtime error handling with graceful degradation
    - Add FP16 instability handling with FP32 fallback
    - Add configuration validation on startup
    - _Requirements: 10.3, 10.4_

- [x] 17. Final checkpoint - Full pipeline integration
  - Ensure all tests pass, ask the user if questions arise.
  - Run end-to-end test with sample video
  - Verify log files are created correctly with deblur metadata
  - Verify GPU memory stays within 6 GB limit
  - Verify N-th frame execution works correctly

## Notes

- All tasks are required for comprehensive correctness validation
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties using hypothesis library
- Unit tests validate specific examples and edge cases
- The pipeline enforces strict rules:
  - YOLO sees only RAW/CLAHE frames
  - MPRNet deblur is ROI-only (max 256px width)
  - MPRNet runs every N frames with caching
  - FP16 inference with FP32 fallback
  - NAFNet is NOT used anywhere
