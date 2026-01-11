# Requirements Document

## Introduction

This document specifies the requirements for an end-to-end AI video processing pipeline for railway wagon counting, damage detection, and OCR logging. The system is designed for industrial deployment on a single Windows 11 machine with an NVIDIA RTX 3050 GPU (6 GB VRAM), requiring real-time or near-real-time inference capabilities for fast-moving vehicles (50–80 km/h), including night-time footage.

## Glossary

- **Pipeline**: The complete video processing system from frame capture to logging
- **Wagon_Detector**: YOLO #1 model responsible for wagon detection, tracking, and counting
- **Damage_Detector**: YOLO #2 model responsible for detecting wagon damage types
- **OCR_Pipeline**: The text extraction subsystem using PaddleOCR
- **MPRNet_Deblur**: Multi-Stage Progressive Image Restoration Network (GoPro-trained Deblurring variant) for ROI-only deblurring
- **Blur_Score**: Laplacian variance metric indicating frame sharpness
- **ROI**: Region of Interest extracted from detected wagons
- **CLAHE**: Contrast Limited Adaptive Histogram Equalization
- **Blur_Threshold**: Single threshold for blur gating decision
- **ByteTrack**: Multi-object tracking algorithm for wagon tracking
- **FP16**: Half-precision floating-point format for GPU inference
- **N_Frame_Interval**: Configurable interval for running MPRNet (every N frames)

## Requirements

### Requirement 1: Video Frame Capture and Blur Detection

**User Story:** As a railway inspector, I want the system to capture video frames and assess their quality, so that appropriate processing decisions can be made for each frame.

#### Acceptance Criteria

1. WHEN a video stream is provided, THE Pipeline SHALL capture frames at the source frame rate
2. WHEN a frame is captured, THE Blur_Detector SHALL compute blur_score using Laplacian variance on the ROI only
3. WHEN blur_score is above the configured threshold (mild blur), THE Pipeline SHALL skip deblurring and pass raw ROI directly to OCR
4. WHEN blur_score is below the configured threshold (severe blur), THE Pipeline SHALL apply MPRNet deblurring to the ROI
5. THE Pipeline SHALL make the blur threshold configurable via config.py
6. THE Blur_Detector SHALL be lightweight and not add significant processing overhead

### Requirement 2: Primary Wagon Detection and Tracking

**User Story:** As a railway inspector, I want wagons to be detected and tracked accurately, so that I can count them without duplicates.

#### Acceptance Criteria

1. THE Wagon_Detector SHALL use Ultralytics YOLOv11n model for wagon detection
2. WHEN detecting wagons, THE Wagon_Detector SHALL receive only RAW or CLAHE-enhanced frames
3. THE Wagon_Detector SHALL NOT receive deblurred frames as input
4. WHEN CLAHE is applied, THE Pipeline SHALL apply it only to the L channel in LAB color space
5. WHEN wagons are detected, THE Tracker SHALL assign unique IDs using ByteTrack algorithm
6. WHEN a wagon crosses the counting line, THE Pipeline SHALL increment the wagon count exactly once
7. THE Pipeline SHALL prevent double-counting of the same wagon
8. WHEN a wagon is detected, THE Wagon_Detector SHALL generate stable ROI coordinates for downstream processing

### Requirement 3: Wagon Damage Detection

**User Story:** As a railway inspector, I want damage on wagons to be automatically detected, so that maintenance issues can be identified and logged.

#### Acceptance Criteria

1. THE Damage_Detector SHALL use a separate Ultralytics YOLOv11n model for damage detection
2. WHEN a wagon ROI is available, THE Damage_Detector SHALL analyze it for damage types
3. THE Damage_Detector SHALL detect door damage, floor damage, structural cracks, and deformation
4. WHEN damage is detected, THE Damage_Detector SHALL output damage class, bounding box, and confidence score
5. WHEN damage is detected, THE Damage_Detector SHALL associate it with the corresponding wagon ID
6. THE Damage_Detector SHALL receive only RAW or CLAHE-enhanced ROI as input

### Requirement 4: OCR Text Extraction Pipeline with MPRNet Deblurring

**User Story:** As a railway inspector, I want wagon identification numbers to be extracted via OCR with optimized deblurring, so that wagons can be identified accurately while maintaining real-time performance.

#### Acceptance Criteria

1. THE OCR_Pipeline SHALL use PaddleOCR with GPU acceleration when available
2. WHEN extracting text, THE OCR_Pipeline SHALL process only the wagon text region ROI
3. WHEN blur_score indicates severe blur, THE OCR_Pipeline SHALL apply MPRNet-Deblur to the OCR ROI only
4. THE OCR_Pipeline SHALL NOT apply deblurring to full frames or full vehicle ROIs
5. WHEN processing low-light ROI, THE OCR_Pipeline SHALL apply adaptive gamma or Zero-DCE enhancement before OCR
6. THE OCR_Pipeline SHALL output extracted text and confidence score
7. THE MPRNet_Deblur model SHALL process only small OCR ROI regions (never full frames)
8. THE OCR_Pipeline SHALL preserve edge contrast and character strokes for OCR accuracy
9. THE OCR_Pipeline SHALL avoid excessive smoothing that degrades OCR performance
10. THE OCR_Pipeline SHALL accept partial OCR results and support multi-frame voting downstream

### Requirement 5: Aggressive ROI Resizing Before Deblurring

**User Story:** As a system operator, I want ROIs to be resized before deblurring, so that GPU memory is conserved and processing speed is maximized.

#### Acceptance Criteria

1. WHEN an ROI is extracted for deblurring, THE Pipeline SHALL resize it to a maximum width of 256 pixels
2. WHEN resizing ROI, THE Pipeline SHALL maintain the original aspect ratio
3. THE resize operation SHALL occur BEFORE passing the ROI to MPRNet
4. THE Pipeline SHALL make the maximum ROI width configurable via config.py
5. IF the ROI width is already below the maximum, THE Pipeline SHALL NOT upscale the ROI

### Requirement 6: N-th Frame Deblurring Execution

**User Story:** As a system operator, I want MPRNet to run only on selected frames, so that processing resources are used efficiently.

#### Acceptance Criteria

1. THE Pipeline SHALL NOT run MPRNet on every frame
2. THE Pipeline SHALL run MPRNet once every N frames, where N is configurable (default 3-5)
3. WHEN MPRNet is not run on a frame, THE Pipeline SHALL reuse the most recent deblurred ROI for OCR
4. THE Pipeline SHALL make the N-frame interval configurable via config.py
5. WHEN a new wagon is first detected, THE Pipeline SHALL run MPRNet on the first available frame for that wagon

### Requirement 7: FP16 Half-Precision Inference

**User Story:** As a system operator, I want MPRNet to use half-precision inference, so that GPU memory usage is reduced and inference speed is improved.

#### Acceptance Criteria

1. THE MPRNet_Deblur model SHALL use FP16 (half precision) for inference by default
2. THE Pipeline SHALL convert both model weights and input tensors to FP16
3. IF numerical instability occurs during FP16 inference, THE Pipeline SHALL fall back to FP32
4. THE Pipeline SHALL make the precision mode configurable via config.py
5. THE Pipeline SHALL log when FP32 fallback occurs

### Requirement 8: Logging and Output

**User Story:** As a railway inspector, I want all detection results logged to files, so that I can review and analyze wagon inspection data.

#### Acceptance Criteria

1. WHEN a wagon is processed, THE Logger SHALL record timestamp, wagon ID, and wagon count index
2. WHEN a wagon is processed, THE Logger SHALL record blur score and frame index
3. WHEN damage is detected, THE Logger SHALL record damage detected flag, damage classes, and bounding boxes
4. WHEN OCR completes, THE Logger SHALL record extracted text and OCR confidence
5. THE Logger SHALL output results to both CSV and JSON formats
6. THE Logger SHALL store debug frames in the outputs/debug_frames/ directory when enabled
7. THE Logger SHALL record whether deblurring was applied and which frame's deblurred ROI was used

### Requirement 9: Performance and Resource Management

**User Story:** As a system operator, I want the pipeline to run efficiently within hardware constraints, so that real-time processing is achieved.

#### Acceptance Criteria

1. THE Pipeline SHALL maintain GPU memory usage at or below 6 GB VRAM
2. THE Pipeline SHALL use batch size of 1 for MPRNet inference
3. THE Pipeline SHALL NOT use tiling for deblurring
4. THE Pipeline SHALL NOT use diffusion models for any processing
5. THE Pipeline SHALL NOT perform full-frame enhancement
6. THE Pipeline SHALL skip deblurring aggressively whenever possible
7. THE Pipeline SHALL achieve real-time or near-real-time inference
8. IF GPU memory exceeds safe limits, THEN THE Pipeline SHALL gracefully degrade or queue operations

### Requirement 10: Configuration and Modularity

**User Story:** As a developer, I want the system to be configurable and modular, so that thresholds and components can be adjusted without code changes.

#### Acceptance Criteria

1. THE Pipeline SHALL load all thresholds from config.py
2. THE Pipeline SHALL follow the mandatory project structure with separate modules for each component
3. WHEN a model file is missing, THE Pipeline SHALL report a clear error message
4. THE Pipeline SHALL support configurable video input sources (file path or stream URL)
5. THE Pipeline SHALL support enabling/disabling debug frame output via configuration
6. THE Pipeline SHALL support configuring: blur threshold, N-frame interval, max ROI width, FP16 mode

### Requirement 11: Image Processing Constraints

**User Story:** As a computer vision engineer, I want strict image processing rules enforced, so that detection accuracy is not compromised and OCR quality is prioritized.

#### Acceptance Criteria

1. THE Pipeline SHALL NOT apply deblurring on full frames
2. THE Pipeline SHALL NOT use Wiener, Richardson-Lucy, or classical deblurring filters
3. THE Pipeline SHALL NOT feed deblurred frames into YOLO detection models
4. THE Pipeline SHALL NOT apply edge hallucination or noise enhancement
5. WHEN enhancement is needed for detection, THE Pipeline SHALL use only CLAHE on L channel
6. THE MPRNet_Deblur model SHALL be the only deblurring method used, and only for OCR ROI
7. THE Pipeline SHALL NOT use NAFNet for any processing
8. THE Pipeline SHALL prioritize OCR accuracy over visual beauty
9. THE Pipeline SHALL avoid over-smoothing and hallucination in deblurred outputs
