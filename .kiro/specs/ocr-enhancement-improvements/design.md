# Design Document: OCR and Image Enhancement Improvements

## Overview

This document describes the technical design for enhancing the Railway Wagon Inspection Pipeline with EasyOCR integration, self-calibrating thresholds, and user-configurable illumination/deblur controls. The enhancements provide adaptive image processing that automatically adjusts to video conditions while giving users manual override capabilities.

Key design principles:
- EasyOCR replaces PaddleOCR for improved text recognition
- Self-calibration analyzes initial frames to compute optimal thresholds
- Users can switch between auto and manual modes at any time
- All settings persist across sessions via configuration
- Frontend provides real-time status indicators and controls

## Architecture

The enhanced pipeline adds the following components:

1. **EasyOCR Pipeline**: Replaces PaddleOCR with EasyOCR for text extraction
2. **Calibration Manager**: Handles auto-calibration for blur and illumination thresholds
3. **Illumination Controller**: Manages gamma correction and low-light detection
4. **Enhanced Blur Detector**: Supports auto-calibration and manual threshold modes
5. **Settings Manager**: Persists and loads user configuration
6. **Frontend Controls**: Sidebar controls for illumination, blur, and deblur settings

```
┌─────────────────────────────────────────────────────────────────┐
│                     Video Frame Input                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Calibration Manager                            │
│  ┌─────────────────────┐  ┌─────────────────────┐               │
│  │ Blur Calibrator     │  │ Illumination        │               │
│  │ - Sample frames     │  │ Calibrator          │               │
│  │ - Compute percentile│  │ - Sample luminance  │               │
│  │ - Set threshold     │  │ - Compute gamma     │               │
│  └─────────────────────┘  └─────────────────────┘               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Processing Pipeline                            │
│  ┌─────────────────────┐  ┌─────────────────────┐               │
│  │ Illumination        │  │ Enhanced Blur       │               │
│  │ Controller          │  │ Detector            │               │
│  │ - Gamma correction  │  │ - Auto/Manual mode  │               │
│  │ - Low-light detect  │  │ - Threshold gating  │               │
│  └─────────────────────┘  └─────────────────────┘               │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────┐  ┌─────────────────────┐               │
│  │ Deblur Manager      │  │ EasyOCR Pipeline    │               │
│  │ - Enable/disable    │  │ - GPU acceleration  │               │
│  │ - Status tracking   │  │ - Multi-language    │               │
│  └─────────────────────┘  └─────────────────────┘               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Settings Manager                               │
│  - Save/Load configuration                                       │
│  - Auto/Manual mode flags                                        │
│  - Threshold persistence                                         │
└─────────────────────────────────────────────────────────────────┘
```

## Components and Interfaces

### 1. EasyOCR Pipeline (pipelines/ocr_pipeline.py)

```python
class OCRPipeline:
    """OCR pipeline using EasyOCR for text extraction."""
    
    DEFAULT_LOW_LIGHT_THRESHOLD = 80
    DEFAULT_GAMMA_VALUE = 1.5
    
    def __init__(
        self,
        gpu_enabled: bool = True,
        language: str = 'en',
        low_light_threshold: int = 80,
        gamma_value: float = 1.5
    ):
        """Initialize EasyOCR with configurable settings.
        
        Args:
            gpu_enabled: Whether to use GPU acceleration
            language: Language code for OCR (default: 'en')
            low_light_threshold: Luminance threshold for low-light detection
            gamma_value: Gamma correction value for low-light enhancement
        """
        
    def _initialize_ocr(self) -> None:
        """Lazy initialization of EasyOCR reader."""
        
    def is_available(self) -> bool:
        """Check if EasyOCR is properly initialized."""
        
    def extract_text(
        self,
        roi: np.ndarray,
        min_confidence: float = 0.5,
        is_low_light: Optional[bool] = None
    ) -> OCRResult:
        """Extract text from ROI using EasyOCR.
        
        Returns:
            OCRResult with text, confidence, and bounding box
        """
        
    def get_low_light_threshold(self) -> int:
        """Get current low-light threshold."""
        
    def set_low_light_threshold(self, threshold: int) -> None:
        """Set low-light threshold (0-255)."""
        
    def get_gamma_value(self) -> float:
        """Get current gamma value."""
        
    def set_gamma_value(self, gamma: float) -> None:
        """Set gamma value (must be positive)."""
```

### 2. Calibration Manager (pipelines/calibration_manager.py)

```python
@dataclass
class CalibrationResult:
    """Result of auto-calibration."""
    blur_threshold: float
    low_light_threshold: int
    gamma_value: float
    sample_count: int
    blur_scores: List[float]
    luminance_values: List[float]

class CalibrationManager:
    """Manages auto-calibration for blur and illumination thresholds."""
    
    DEFAULT_SAMPLE_SIZE = 30
    DEFAULT_BLUR_PERCENTILE = 50  # Median
    DEFAULT_LUMINANCE_PERCENTILE = 25  # Lower quartile
    
    def __init__(
        self,
        sample_size: int = 30,
        blur_percentile: float = 50,
        luminance_percentile: float = 25
    ):
        """Initialize calibration manager.
        
        Args:
            sample_size: Number of frames to sample for calibration
            blur_percentile: Percentile for blur threshold (0-100)
            luminance_percentile: Percentile for low-light threshold (0-100)
        """
        
    def add_sample(self, frame: np.ndarray) -> bool:
        """Add a frame sample for calibration.
        
        Returns:
            True if calibration is complete (enough samples)
        """
        
    def is_calibration_complete(self) -> bool:
        """Check if calibration has enough samples."""
        
    def get_calibration_progress(self) -> float:
        """Get calibration progress (0.0 to 1.0)."""
        
    def compute_calibration(self) -> CalibrationResult:
        """Compute calibrated thresholds from samples.
        
        Returns:
            CalibrationResult with computed thresholds
        """
        
    def reset(self) -> None:
        """Reset calibration state for recalibration."""
```

### 3. Illumination Controller (utils/illumination_controller.py)

```python
@dataclass
class IlluminationSettings:
    """Illumination processing settings."""
    gamma_value: float = 1.0
    low_light_threshold: int = 80
    auto_mode: bool = True
    enabled: bool = True

class IlluminationController:
    """Controls image illumination with gamma correction."""
    
    def __init__(
        self,
        gamma_value: float = 1.0,
        low_light_threshold: int = 80,
        auto_mode: bool = True
    ):
        """Initialize illumination controller.
        
        Args:
            gamma_value: Initial gamma value (< 1 brightens, > 1 darkens)
            low_light_threshold: Luminance threshold for low-light detection
            auto_mode: Whether to use auto-calibrated values
        """
        
    def apply_gamma(self, image: np.ndarray, gamma: float) -> np.ndarray:
        """Apply gamma correction to image.
        
        Args:
            image: Input BGR image
            gamma: Gamma value (< 1 brightens, > 1 darkens)
            
        Returns:
            Gamma-corrected image
        """
        
    def increase_illumination(self, image: np.ndarray, amount: float = 0.1) -> np.ndarray:
        """Increase image brightness by reducing gamma.
        
        Args:
            image: Input BGR image
            amount: Amount to reduce gamma (0.0 to 0.5)
            
        Returns:
            Brightened image
        """
        
    def decrease_illumination(self, image: np.ndarray, amount: float = 0.1) -> np.ndarray:
        """Decrease image brightness by increasing gamma.
        
        Args:
            image: Input BGR image
            amount: Amount to increase gamma (0.0 to 0.5)
            
        Returns:
            Darkened image
        """
        
    def is_low_light(self, image: np.ndarray) -> bool:
        """Check if image is low-light based on mean luminance."""
        
    def get_mean_luminance(self, image: np.ndarray) -> float:
        """Get mean luminance of image."""
        
    def get_settings(self) -> IlluminationSettings:
        """Get current illumination settings."""
        
    def set_settings(self, settings: IlluminationSettings) -> None:
        """Set illumination settings."""
        
    def set_auto_mode(self, enabled: bool) -> None:
        """Enable or disable auto mode."""
        
    def update_from_calibration(self, result: CalibrationResult) -> None:
        """Update settings from calibration result."""
```

### 4. Enhanced Blur Detector (pipelines/blur_detector.py)

```python
@dataclass
class BlurSettings:
    """Blur detection settings."""
    threshold: float = 100.0
    auto_mode: bool = True
    deblur_enabled: bool = True

class BlurDetector:
    """Enhanced blur detector with auto-calibration support."""
    
    def __init__(
        self,
        t1: float,
        t2: float,
        auto_mode: bool = True
    ):
        """Initialize blur detector with thresholds.
        
        Args:
            t1: Lower blur threshold
            t2: Upper blur threshold
            auto_mode: Whether to use auto-calibrated values
        """
        
    def compute_blur_score(self, frame: np.ndarray) -> float:
        """Compute Laplacian variance blur score."""
        
    def get_blur_decision(self, blur_score: float) -> BlurDecision:
        """Get blur decision based on score and thresholds."""
        
    def set_threshold(self, threshold: float) -> None:
        """Set blur threshold (for single-threshold mode)."""
        
    def get_threshold(self) -> float:
        """Get current blur threshold."""
        
    def set_auto_mode(self, enabled: bool) -> None:
        """Enable or disable auto mode."""
        
    def is_auto_mode(self) -> bool:
        """Check if auto mode is enabled."""
        
    def update_from_calibration(self, result: CalibrationResult) -> None:
        """Update threshold from calibration result."""
        
    def get_settings(self) -> BlurSettings:
        """Get current blur settings."""
        
    def set_settings(self, settings: BlurSettings) -> None:
        """Set blur settings."""
```

### 5. Enhanced Deblur Manager (pipelines/deblur_manager.py)

```python
@dataclass
class DeblurStatus:
    """Status of deblur operation."""
    enabled: bool
    applied: bool
    blur_score_before: float
    blur_score_after: Optional[float]
    status_text: str  # "ACTIVE", "SKIPPED", "DISABLED"

class DeblurManager:
    """Enhanced deblur manager with enable/disable and status tracking."""
    
    def __init__(
        self,
        mprnet: "MPRNetDeblur",
        blur_detector: "BlurDetector",
        frame_interval: int = 3,
        max_roi_width: int = 256,
        enable_clahe: bool = True,
        deblur_enabled: bool = True
    ):
        """Initialize deblur manager.
        
        Args:
            mprnet: MPRNet deblurring wrapper
            blur_detector: Blur detector instance
            frame_interval: N-th frame interval for deblurring
            max_roi_width: Maximum ROI width before deblurring
            enable_clahe: Whether to apply CLAHE enhancement
            deblur_enabled: Whether deblurring is enabled
        """
        
    def set_deblur_enabled(self, enabled: bool) -> None:
        """Enable or disable deblurring."""
        
    def is_deblur_enabled(self) -> bool:
        """Check if deblurring is enabled."""
        
    def process_roi(
        self,
        roi: np.ndarray,
        wagon_id: int,
        frame_index: int
    ) -> Tuple[np.ndarray, bool, Optional[int]]:
        """Process ROI with conditional deblurring."""
        
    def get_last_status(self) -> DeblurStatus:
        """Get status of last deblur operation."""
        
    def was_deblur_applied(self) -> bool:
        """Check if deblur was applied to current frame."""
```

### 6. Settings Manager (utils/settings_manager.py)

```python
@dataclass
class PipelineSettings:
    """Complete pipeline settings for persistence."""
    # Blur settings
    blur_threshold: float = 100.0
    blur_auto_mode: bool = True
    deblur_enabled: bool = True
    
    # Illumination settings
    low_light_threshold: int = 80
    gamma_value: float = 1.0
    illumination_auto_mode: bool = True
    
    # OCR settings
    ocr_language: str = 'en'
    ocr_gpu_enabled: bool = True
    
    # Calibration settings
    calibration_sample_size: int = 30
    blur_percentile: float = 50
    luminance_percentile: float = 25

class SettingsManager:
    """Manages persistence of pipeline settings."""
    
    DEFAULT_SETTINGS_PATH = "pipeline_settings.json"
    
    def __init__(self, settings_path: str = None):
        """Initialize settings manager.
        
        Args:
            settings_path: Path to settings file (default: pipeline_settings.json)
        """
        
    def load_settings(self) -> PipelineSettings:
        """Load settings from file.
        
        Returns:
            PipelineSettings with loaded or default values
        """
        
    def save_settings(self, settings: PipelineSettings) -> None:
        """Save settings to file."""
        
    def reset_to_defaults(self) -> PipelineSettings:
        """Reset settings to defaults and save."""
        
    def get_current_settings(self) -> PipelineSettings:
        """Get current in-memory settings."""
        
    def update_blur_settings(
        self,
        threshold: Optional[float] = None,
        auto_mode: Optional[bool] = None,
        deblur_enabled: Optional[bool] = None
    ) -> None:
        """Update blur-related settings."""
        
    def update_illumination_settings(
        self,
        low_light_threshold: Optional[int] = None,
        gamma_value: Optional[float] = None,
        auto_mode: Optional[bool] = None
    ) -> None:
        """Update illumination-related settings."""
```

## Data Models

```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List

class CalibrationMode(Enum):
    """Calibration mode for thresholds."""
    AUTO = "auto"
    MANUAL = "manual"

class DeblurStatusType(Enum):
    """Deblur status types."""
    ACTIVE = "active"      # Deblur was applied
    SKIPPED = "skipped"    # Deblur skipped (image sharp enough)
    DISABLED = "disabled"  # Deblur disabled by user

@dataclass
class OCRResult:
    """Result of OCR text extraction."""
    text: str
    confidence: float
    bbox: Optional[BoundingBox]

@dataclass
class CalibrationResult:
    """Result of auto-calibration."""
    blur_threshold: float
    low_light_threshold: int
    gamma_value: float
    sample_count: int
    blur_scores: List[float]
    luminance_values: List[float]

@dataclass
class DeblurStatus:
    """Status of deblur operation."""
    enabled: bool
    applied: bool
    blur_score_before: float
    blur_score_after: Optional[float]
    status_type: DeblurStatusType

@dataclass
class IlluminationSettings:
    """Illumination processing settings."""
    gamma_value: float
    low_light_threshold: int
    auto_mode: bool
    enabled: bool

@dataclass
class BlurSettings:
    """Blur detection settings."""
    threshold: float
    auto_mode: bool
    deblur_enabled: bool

@dataclass
class PipelineSettings:
    """Complete pipeline settings for persistence."""
    blur_threshold: float
    blur_auto_mode: bool
    deblur_enabled: bool
    low_light_threshold: int
    gamma_value: float
    illumination_auto_mode: bool
    ocr_language: str
    ocr_gpu_enabled: bool
    calibration_sample_size: int
    blur_percentile: float
    luminance_percentile: float
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system - essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: OCR Result Completeness

*For any* ROI passed to the OCR pipeline, the returned OCRResult SHALL contain:
- A text field (string, may be empty)
- A confidence score in range [0.0, 1.0]
- An optional bounding box (may be None if no text detected)

**Validates: Requirements 1.5**

### Property 2: OCR Language Configuration

*For any* valid language code passed to the OCR pipeline initialization, the EasyOCR reader SHALL be configured with that language. Extracting text SHALL use the configured language.

**Validates: Requirements 1.4**

### Property 3: Deblur Operation Logging

*For any* deblur operation that is applied, the log SHALL contain:
- Frame index (non-negative integer)
- Wagon ID (positive integer)
- Blur score before deblurring
- Blur score after deblurring (if computed)

**Validates: Requirements 2.1, 2.5**

### Property 4: Blur Auto-Calibration

*For any* set of sample frames provided to the calibration manager:
- The computed blur_threshold SHALL be at the configured percentile of blur scores
- The threshold SHALL be within the range of observed blur scores
- Calibration SHALL complete when sample_count reaches configured sample_size

**Validates: Requirements 3.2, 3.3, 3.4**

### Property 5: Illumination Auto-Calibration

*For any* set of sample frames provided to the calibration manager:
- The computed low_light_threshold SHALL be at the configured percentile of luminance values
- The computed gamma_value SHALL be inversely proportional to mean luminance
- Calibration SHALL complete when sample_count reaches configured sample_size

**Validates: Requirements 4.2, 4.3, 4.4, 4.5**

### Property 6: Manual Mode Override

*For any* user-specified threshold value when manual mode is selected:
- The pipeline SHALL use the exact user-specified value
- Auto-calibrated values SHALL NOT override manual settings
- Switching to manual mode SHALL preserve the current threshold value

**Validates: Requirements 3.6, 4.7**

### Property 7: Settings Real-Time Application

*For any* settings change made through the frontend:
- The pipeline SHALL apply the new settings immediately
- The next processed frame SHALL use the updated settings
- No restart or reinitialization SHALL be required

**Validates: Requirements 5.4, 6.4**

### Property 8: Configuration Round-Trip

*For any* valid PipelineSettings object:
- Saving to file then loading SHALL produce an equivalent settings object
- All fields SHALL be preserved: blur_threshold, low_light_threshold, gamma_value, deblur_enabled, auto_calibration flags

**Validates: Requirements 5.7, 6.7, 9.1, 9.2, 9.3, 9.4, 9.5**

### Property 9: Deblur Enable/Disable

*For any* frame processed when deblur is disabled:
- No deblurring operation SHALL be performed
- The deblur status SHALL be DISABLED
- The original ROI SHALL be returned unchanged (except for resizing)

*For any* frame processed when deblur is enabled:
- Deblurring SHALL follow the blur threshold logic
- The deblur status SHALL be either ACTIVE or SKIPPED based on blur score

**Validates: Requirements 7.2, 7.3**

### Property 10: Gamma Direction Correctness

*For any* illumination increase operation:
- The applied gamma value SHALL be less than 1.0
- The resulting image mean luminance SHALL be higher than the input

*For any* illumination decrease operation:
- The applied gamma value SHALL be greater than 1.0
- The resulting image mean luminance SHALL be lower than the input

**Validates: Requirements 8.3, 8.4, 8.5**

## Error Handling

### OCR Errors

| Error Condition | Handling Strategy |
|----------------|-------------------|
| EasyOCR not installed | Raise `OCRInitializationError` with installation instructions |
| GPU not available | Fall back to CPU with warning log |
| Invalid ROI (None/empty) | Return empty OCRResult with zero confidence |
| OCR timeout | Return empty OCRResult, log warning |
| Invalid language code | Raise `ValueError` with supported languages list |

### Calibration Errors

| Error Condition | Handling Strategy |
|----------------|-------------------|
| Insufficient samples | Return None, continue collecting samples |
| All frames identical | Use default thresholds, log warning |
| Invalid percentile (< 0 or > 100) | Raise `ValueError` |
| Empty frame provided | Skip frame, log warning |

### Settings Errors

| Error Condition | Handling Strategy |
|----------------|-------------------|
| Settings file not found | Create with defaults, log info |
| Corrupted settings file | Reset to defaults, log warning |
| Invalid threshold values | Clamp to valid range, log warning |
| Permission denied on save | Raise `IOError` with path information |

### Illumination Errors

| Error Condition | Handling Strategy |
|----------------|-------------------|
| Invalid gamma (≤ 0) | Raise `ValueError` |
| Invalid threshold (< 0 or > 255) | Raise `ValueError` |
| Empty image | Raise `ValueError` |

## Testing Strategy

### Unit Tests

Unit tests verify specific examples and edge cases:

1. **EasyOCR Integration Tests**
   - Test initialization with GPU enabled/disabled
   - Test language configuration
   - Test text extraction on sample images
   - Test handling of empty/invalid ROIs

2. **Calibration Manager Tests**
   - Test sample collection
   - Test calibration completion detection
   - Test percentile computation
   - Test reset functionality

3. **Illumination Controller Tests**
   - Test gamma correction application
   - Test increase/decrease illumination
   - Test low-light detection
   - Test settings get/set

4. **Settings Manager Tests**
   - Test save/load round-trip
   - Test default values
   - Test reset to defaults
   - Test partial updates

5. **Deblur Manager Tests**
   - Test enable/disable functionality
   - Test status tracking
   - Test was_deblur_applied method

### Property-Based Tests

Property-based tests verify universal properties across many generated inputs. Each test runs minimum 100 iterations.

**Testing Framework**: pytest with hypothesis library

1. **Property Test: OCR Result Completeness**
   - Generate random valid ROI images
   - Verify OCRResult contains all required fields
   - **Tag: Feature: ocr-enhancement-improvements, Property 1: OCR Result Completeness**

2. **Property Test: OCR Language Configuration**
   - Generate random valid language codes
   - Verify language is correctly configured
   - **Tag: Feature: ocr-enhancement-improvements, Property 2: OCR Language Configuration**

3. **Property Test: Deblur Operation Logging**
   - Generate random deblur operations
   - Verify log contains all required fields
   - **Tag: Feature: ocr-enhancement-improvements, Property 3: Deblur Operation Logging**

4. **Property Test: Blur Auto-Calibration**
   - Generate random sets of blur scores
   - Verify threshold is at correct percentile
   - **Tag: Feature: ocr-enhancement-improvements, Property 4: Blur Auto-Calibration**

5. **Property Test: Illumination Auto-Calibration**
   - Generate random sets of luminance values
   - Verify threshold and gamma are computed correctly
   - **Tag: Feature: ocr-enhancement-improvements, Property 5: Illumination Auto-Calibration**

6. **Property Test: Manual Mode Override**
   - Generate random threshold values
   - Verify manual values are used exactly
   - **Tag: Feature: ocr-enhancement-improvements, Property 6: Manual Mode Override**

7. **Property Test: Configuration Round-Trip**
   - Generate random valid PipelineSettings
   - Save and load, verify equivalence
   - **Tag: Feature: ocr-enhancement-improvements, Property 8: Configuration Round-Trip**

8. **Property Test: Deblur Enable/Disable**
   - Generate random frames and enable/disable states
   - Verify correct behavior based on state
   - **Tag: Feature: ocr-enhancement-improvements, Property 9: Deblur Enable/Disable**

9. **Property Test: Gamma Direction Correctness**
   - Generate random images and gamma adjustments
   - Verify luminance changes in correct direction
   - **Tag: Feature: ocr-enhancement-improvements, Property 10: Gamma Direction Correctness**

### Integration Tests

1. **End-to-End OCR Test**
   - Process test images with known text
   - Verify text extraction accuracy

2. **Calibration Integration Test**
   - Process video and verify calibration completes
   - Verify calibrated thresholds are reasonable

3. **Settings Persistence Test**
   - Change settings, restart application
   - Verify settings are preserved

4. **Frontend Control Test**
   - Adjust sliders and toggles
   - Verify pipeline behavior changes accordingly
