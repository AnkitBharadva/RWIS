"""
Data models for the Streamlit Mission Control Dashboard.

This module defines the core data structures used throughout the dashboard:
- ConnectionStatus: Enum for video connection states
- ProcessingType: Enum for frame processing types
- DetectionLogEntry: Individual damage detection record
- OCRLogEntry: Individual OCR extraction record
- DashboardMetrics: Live performance metrics
- EnhancedMetrics: Extended metrics with wagon count
- SidebarSettings: User-configurable settings
- FrameSaveConfig: Configuration for frame saving
- FrameMetadata: Metadata for saved frames
- SessionState: Complete dashboard state

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 5.2, 5.4, 5.5, 6.2
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Any
import numpy as np


class ConnectionStatus(Enum):
    """
    Enum representing the video connection status.
    
    States:
        DISCONNECTED: No active video connection
        CONNECTING: Connection attempt in progress
        CONNECTED: Successfully connected to video source
        ERROR: Connection failed or encountered an error
    """
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


class ProcessingType(Enum):
    """
    Enum representing types of processing applied to frames.
    
    Types:
        DEBLUR: Deblurring processing applied
        CLAHE: CLAHE illumination enhancement applied
        GAMMA: Gamma correction applied
        OCR: OCR text extraction performed
    
    Requirements: 5.4, 5.5
    """
    DEBLUR = "deblur"
    CLAHE = "clahe"
    GAMMA = "gamma"
    OCR = "ocr"


@dataclass
class DetectionLogEntry:
    """
    Represents a single damage detection event.
    
    Attributes:
        timestamp: When the detection occurred
        wagon_id: Identifier of the detected wagon
        damage_type: Type/category of damage detected
        confidence: Detection confidence score (0.0 to 1.0)
        frame_index: Frame number where detection occurred
    
    Requirements: 5.2
    """
    timestamp: datetime
    wagon_id: int
    damage_type: str
    confidence: float
    frame_index: int


@dataclass
class OCRLogEntry:
    """
    Represents a single OCR text extraction event.
    
    Attributes:
        timestamp: When the OCR extraction occurred
        wagon_id: Identifier of the wagon where text was extracted
        extracted_text: The text extracted by OCR
        confidence: OCR confidence score (0.0 to 1.0)
        frame_index: Frame number where extraction occurred
    
    Requirements: 6.2, 6.4, 6.6
    """
    timestamp: datetime
    wagon_id: int
    extracted_text: str
    confidence: float
    frame_index: int


@dataclass
class DashboardMetrics:
    """
    Live performance metrics displayed in the dashboard.
    
    Attributes:
        fps: Current frames per second (>= 0.0)
        inference_ms: Inference latency in milliseconds (>= 0.0)
        object_count: Number of objects/wagons detected (>= 0)
        damage_count: Number of damage detections (>= 0)
        damage_detected: Whether damage was detected in current frame
    
    Requirements: 3.2, 3.3, 3.4, 3.5
    """
    fps: float = 0.0
    inference_ms: float = 0.0
    object_count: int = 0
    damage_count: int = 0
    damage_detected: bool = False


@dataclass
class EnhancedMetrics:
    """
    Extended performance metrics including total wagon count.
    
    Attributes:
        fps: Current frames per second (>= 0.0)
        processing_latency_ms: Processing latency in milliseconds (>= 0.0)
        objects_detected: Number of objects/wagons detected (>= 0)
        total_wagon_count: Cumulative count of unique wagons that crossed counting line
        damage_count: Number of damage detections (>= 0)
        damage_detected: Whether damage was detected in current frame
    
    Requirements: 3.1, 3.2, 3.3, 3.4, 3.5
    """
    fps: float = 0.0
    processing_latency_ms: float = 0.0
    objects_detected: int = 0
    total_wagon_count: int = 0
    damage_count: int = 0
    damage_detected: bool = False


@dataclass
class SidebarSettings:
    """
    User-configurable settings from the sidebar.
    
    Attributes:
        video_source: RTSP URL or file path for video input
        confidence_threshold: Minimum confidence for detections (0.0 to 1.0)
        frame_skip: Number of frames to skip between processing (1-10)
        enable_damage_detection: Whether damage detection is enabled
    """
    video_source: str = ""
    confidence_threshold: float = 0.5
    frame_skip: int = 3
    enable_damage_detection: bool = True


@dataclass
class FrameSaveConfig:
    """
    Configuration for automatic frame saving.
    
    Attributes:
        enabled: Whether frame saving is enabled
        save_on_deblur: Save frames when deblur processing is applied
        save_on_illumination: Save frames when illumination enhancement is applied
        save_on_ocr: Save frames when OCR is performed
        output_directory: Directory path for saved frames
    
    Requirements: 5.4, 5.5, 8.1, 8.2, 8.3, 8.4
    """
    enabled: bool = False
    save_on_deblur: bool = True
    save_on_illumination: bool = True
    save_on_ocr: bool = True
    output_directory: str = "outputs/saved_frames"


@dataclass
class FrameMetadata:
    """
    Metadata for saved frames.
    
    Attributes:
        timestamp: When the frame was saved
        frame_index: Frame number in the video sequence
        processing_applied: List of processing types applied to the frame
        wagon_id: Optional wagon ID if frame contains a specific wagon
    
    Requirements: 5.4, 5.5
    """
    timestamp: datetime
    frame_index: int
    processing_applied: List[str]
    wagon_id: Optional[int] = None


@dataclass
class SessionState:
    """
    Complete dashboard session state.
    
    This dataclass holds all state needed for the dashboard session,
    including running status, connection state, metrics, settings,
    detection history, and the last processed frame.
    
    Attributes:
        is_running: Whether video processing is active
        connection_status: Current video connection status
        metrics: Live performance metrics
        settings: User-configurable settings
        detection_log: Chronological list of detection events (append-only)
        last_frame: Most recently processed video frame
    """
    is_running: bool = False
    connection_status: ConnectionStatus = ConnectionStatus.DISCONNECTED
    metrics: DashboardMetrics = field(default_factory=DashboardMetrics)
    settings: SidebarSettings = field(default_factory=SidebarSettings)
    detection_log: List[DetectionLogEntry] = field(default_factory=list)
    last_frame: Optional[Any] = None  # np.ndarray, using Any to avoid numpy import issues
