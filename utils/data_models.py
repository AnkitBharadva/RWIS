"""Data models and enums for the Railway Wagon Inspection Pipeline."""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple


class BlurDecision(Enum):
    """Blur decision based on blur score thresholds.
    
    SKIP_DEBLUR: blur_score < T1 (frame is sharp enough)
    ROI_DEBLUR: T1 <= blur_score < T2 (apply deblur to OCR ROI only)
    NO_DEBLUR: blur_score >= T2 (frame too blurry, skip deblur)
    """
    SKIP_DEBLUR = "skip_deblur"
    ROI_DEBLUR = "roi_deblur"
    NO_DEBLUR = "no_deblur"


class CalibrationMode(Enum):
    """Calibration mode for thresholds.
    
    AUTO: Thresholds are automatically computed from image statistics
    MANUAL: User-specified thresholds override auto-calibration
    """
    AUTO = "auto"
    MANUAL = "manual"


class DeblurStatusType(Enum):
    """Deblur status types for frontend display.
    
    ACTIVE: Deblur was applied to the ROI
    SKIPPED: Deblur skipped (image sharp enough or too blurry)
    DISABLED: Deblur disabled by user
    """
    ACTIVE = "active"
    SKIPPED = "skipped"
    DISABLED = "disabled"


class DamageClass(Enum):
    """Types of damage that can be detected on wagons.
    
    These classes match the actual YOLO model classes in wagon_detector.pt:
    {0: 'Bamboo Door', 1: 'Breakage', 2: 'Close Door', 3: 'Damage Door', 
     4: 'Dent', 5: 'Open Door', 6: 'Wagon'}
    Note: Class 6 (Wagon) is excluded - we use a separate model for wagon detection.
    """
    BAMBOO_DOOR = "bamboo_door"
    BREAKAGE = "breakage"
    CLOSE_DOOR = "close_door"
    DAMAGE_DOOR = "damage_door"
    DENT = "dent"
    OPEN_DOOR = "open_door"


@dataclass
class BoundingBox:
    """Bounding box coordinates for detections.
    
    Attributes:
        x1: Left x-coordinate
        y1: Top y-coordinate
        x2: Right x-coordinate
        y2: Bottom y-coordinate
    """
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        """Width of the bounding box."""
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        """Height of the bounding box."""
        return self.y2 - self.y1

    @property
    def center(self) -> Tuple[int, int]:
        """Center point of the bounding box."""
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


@dataclass
class WagonDetection:
    """Detection result from the wagon detector (YOLO #1).
    
    Attributes:
        bbox: Bounding box of the detected wagon
        confidence: Detection confidence score [0.0, 1.0]
        class_id: Class identifier from the model
    """
    bbox: BoundingBox
    confidence: float
    class_id: int


@dataclass
class TrackedWagon:
    """Tracked wagon with assigned ID and counting status.
    
    Attributes:
        track_id: Unique tracking ID assigned by ByteTrack
        bbox: Current bounding box of the wagon
        confidence: Detection confidence score
        crossed_line: Whether the wagon has crossed the counting line
        count_index: Index assigned when wagon crosses line (None if not crossed)
    """
    track_id: int
    bbox: BoundingBox
    confidence: float
    crossed_line: bool
    count_index: Optional[int]


@dataclass
class DamageDetection:
    """Damage detection result from the damage detector (YOLO #2).
    
    Attributes:
        damage_class: Type of damage detected
        bbox: Bounding box of the damage within the wagon ROI
        confidence: Detection confidence score [0.0, 1.0]
        wagon_id: ID of the wagon this damage belongs to
    """
    damage_class: DamageClass
    bbox: BoundingBox
    confidence: float
    wagon_id: int


@dataclass
class OCRResult:
    """OCR extraction result from PaddleOCR.
    
    Attributes:
        text: Extracted text (may be empty if no text detected)
        confidence: OCR confidence score [0.0, 1.0]
        bbox: Bounding box of the text region (optional)
    """
    text: str
    confidence: float
    bbox: Optional[BoundingBox] = None


@dataclass
class OCRDetection:
    """Represents a single OCR detection with position and confidence for visualization.
    
    Attributes:
        text: Extracted text string
        confidence: OCR confidence score [0.0, 1.0]
        bbox: Bounding box of the text region (ROI-relative coordinates)
        wagon_id: ID of the wagon this detection belongs to
        frame_index: Index of the video frame where detection occurred
    """
    text: str
    confidence: float
    bbox: BoundingBox
    wagon_id: int
    frame_index: int


@dataclass
class OCRFrameMetadata:
    """Metadata saved alongside OCR frames for offline review.
    
    Attributes:
        timestamp: ISO format timestamp of the frame capture
        frame_index: Index of the video frame
        wagon_id: ID of the wagon in the frame
        detections: List of detection dictionaries with text, confidence, bbox
        deblur_applied: Whether deblurring was applied before OCR
        illumination_applied: Whether illumination enhancement was applied
        blur_score: Computed blur score for the frame (optional)
        luminance_level: Measured luminance level (optional)
    """
    timestamp: str
    frame_index: int
    wagon_id: int
    detections: List[dict]
    deblur_applied: bool
    illumination_applied: bool
    blur_score: Optional[float] = None
    luminance_level: Optional[float] = None


@dataclass
class DeblurResult:
    """Result from the deblur manager.
    
    Attributes:
        roi: The processed ROI (raw or deblurred)
        deblur_applied: Whether deblurring was applied
        source_frame: Frame index from which deblurred ROI was obtained (None if not deblurred)
        blur_score: Computed blur score for the ROI
    """
    roi: any  # np.ndarray, using any to avoid numpy import
    deblur_applied: bool
    source_frame: Optional[int]
    blur_score: float


@dataclass
class DeblurStatus:
    """Status of deblur operation for frontend display.
    
    Attributes:
        enabled: Whether deblurring is enabled
        applied: Whether deblurring was applied to the current ROI
        blur_score_before: Blur score before deblurring
        blur_score_after: Blur score after deblurring (None if not computed)
        status_type: Type of deblur status (ACTIVE, SKIPPED, DISABLED)
    """
    enabled: bool
    applied: bool
    blur_score_before: float
    blur_score_after: Optional[float]
    status_type: DeblurStatusType
    
    @property
    def status_text(self) -> str:
        """Get human-readable status text for display."""
        return self.status_type.value.upper()


@dataclass
class BlurSettings:
    """Blur detection settings for auto/manual mode support.
    
    Attributes:
        threshold: Current blur threshold value (t2 in dual-threshold mode)
        auto_mode: Whether auto-calibration mode is enabled
        deblur_enabled: Whether deblurring is enabled
    """
    threshold: float = 100.0
    auto_mode: bool = True
    deblur_enabled: bool = True


@dataclass
class WagonRecord:
    """Complete inspection record for a wagon, used for logging.
    
    Attributes:
        timestamp: ISO format timestamp of the inspection
        wagon_id: Unique tracking ID of the wagon
        count_index: Sequential count index when wagon crossed line
        blur_score: Computed blur score for the frame
        frame_index: Index of the video frame
        damage_detected: Whether any damage was detected
        damage_classes: List of damage class names detected
        damage_bboxes: List of bounding boxes for detected damages
        ocr_text: Extracted text from OCR
        ocr_confidence: OCR confidence score [0.0, 1.0]
        deblur_applied: Whether deblurring was applied to the ROI
        deblur_source_frame: Frame index from which deblurred ROI was used (None if not deblurred)
    """
    timestamp: str
    wagon_id: int
    count_index: int
    blur_score: float
    frame_index: int
    damage_detected: bool
    damage_classes: List[str]
    damage_bboxes: List[BoundingBox]
    ocr_text: str
    ocr_confidence: float
    deblur_applied: bool = False
    deblur_source_frame: Optional[int] = None
