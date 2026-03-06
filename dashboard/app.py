"""
Main dashboard application for the Mission Control interface.

Provides a professional industrial monitoring dashboard for the
High-Speed Railway Wagon Inspection System.
"""

import streamlit as st
import numpy as np
import pandas as pd
import cv2
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
import os
import time

from dashboard.styles import inject_css
from dashboard.video_manager import VideoManager
from dashboard.metrics import MetricsCalculator, LATENCY_WARNING_THRESHOLD_MS
from dashboard.models import (
    SidebarSettings, ConnectionStatus, DetectionLogEntry, FrameSaveConfig,
    FrameMetadata, OCRLogEntry
)

# Import new dashboard enhancement components
from dashboard.dual_display import DualVideoDisplay
from dashboard.track_renderer import TrackIDRenderer
from dashboard.frame_saver import FrameSaver
from dashboard.ocr_log import OCRLogDisplay

# CPU Performance Optimization: Enable OpenCV multi-threading
# Uses 8 threads to leverage multi-core CPU for faster image processing
cv2.setNumThreads(8)

# Import pipeline components for wagon and damage detection
from pipelines.wagon_detector import WagonDetector
from pipelines.damage_detector import DamageDetector
from utils.data_models import WagonDetection, DamageDetection, BoundingBox, TrackedWagon

# Import illumination controller for frontend controls (Requirements 5.1, 5.2, 5.3, 8.1, 8.2)
from utils.illumination_controller import IlluminationController, IlluminationSettings

# Import blur detector and deblur manager for frontend controls (Requirements 6.1, 6.2, 6.3, 6.5, 6.6, 7.4, 7.5, 7.6)
from pipelines.blur_detector import BlurDetector
from pipelines.deblur_manager import DeblurManager
from utils.data_models import BlurSettings, DeblurStatus, DeblurStatusType, CalibrationMode

# Import calibration manager for calibration status display (Requirements 10.1, 10.2, 10.3, 10.4, 10.5, 10.6)
from pipelines.calibration_manager import CalibrationManager, CalibrationResult

# Import settings manager for settings persistence (Requirements 9.6, 9.7)
from utils.settings_manager import SettingsManager, PipelineSettings

# Import OCR pipeline for text extraction (Requirements 1.1, 1.2, 1.3, 1.4, 1.5)
from pipelines.ocr_pipeline import OCRPipeline

# Import OCR visualization components (Requirements 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 2.5)
from dashboard.ocr_visualization import OCRVisualization
from dashboard.ocr_interval_controller import OCRIntervalController
from utils.data_models import OCRDetection

# Import processing status indicators for metrics row (Requirements 7.1, 7.2, 7.3, 7.4, 8.1)
from dashboard.processing_indicators import ProcessingStatusIndicator

# Import OCR frame saver for saving OCR frames with metadata (Requirements 3.1, 3.2, 3.3, 3.4, 6.2)
from dashboard.ocr_frame_saver import OCRFrameSaver
from utils.data_models import OCRFrameMetadata

# Import MPRNet wrapper for deblurring (Requirements 4.3, 4.4, 7.1, 7.2, 7.3)
from pipelines.mprnet_wrapper import MPRNetDeblur


def get_frame_save_config_from_session() -> FrameSaveConfig:
    """
    Get the current FrameSaveConfig from session state.
    
    Retrieves frame saving configuration values from Streamlit session state
    and returns a FrameSaveConfig dataclass instance.
    
    Returns:
        FrameSaveConfig with current session state values
        
    Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
    """
    import streamlit as st
    return FrameSaveConfig(
        enabled=st.session_state.get("frame_save_enabled", False),
        save_on_deblur=st.session_state.get("frame_save_on_deblur", True),
        save_on_illumination=st.session_state.get("frame_save_on_illumination", True),
        save_on_ocr=st.session_state.get("frame_save_on_ocr", True),
        output_directory=st.session_state.get("frame_save_output_directory", "outputs/saved_frames")
    )


def get_damage_indicator_state(damage_detected: bool) -> dict:
    """
    Get the visual indicator state based on damage detection status.
    
    This function determines the visual feedback properties for the
    damage status indicator based on whether damage is detected.
    
    Args:
        damage_detected: Whether damage is currently detected
        
    Returns:
        dict with keys:
            - color: "red" if damage detected, "green" otherwise
            - css_class: "status-alert" if damage detected, "status-normal" otherwise
            - text: "DAMAGE DETECTED" if damage detected, "NORMAL" otherwise
            - delta_color: "inverse" if damage detected, "normal" otherwise
            
    Requirements: 6.1, 6.2, 6.4
    """
    if damage_detected:
        return {
            "color": "red",
            "css_class": "status-alert",
            "text": "DAMAGE DETECTED",
            "delta_color": "inverse"
        }
    else:
        return {
            "color": "green",
            "css_class": "status-normal",
            "text": "NORMAL",
            "delta_color": "normal"
        }


def draw_bounding_boxes(
    frame: np.ndarray,
    wagon_detections: List[WagonDetection],
    damage_detections: List[DamageDetection]
) -> np.ndarray:
    """
    Draw bounding boxes on the frame for wagon and damage detections.
    
    Draws green boxes for wagon detections and red boxes for damage detections.
    Also adds labels with confidence scores.
    
    Args:
        frame: Input BGR frame to draw on
        wagon_detections: List of wagon detections
        damage_detections: List of damage detections
        
    Returns:
        Frame with bounding boxes drawn
        
    Requirements: 4.3
    """
    # Make a copy to avoid modifying the original frame
    annotated_frame = frame.copy()
    
    # Draw wagon detections in green
    for detection in wagon_detections:
        bbox = detection.bbox
        # Green color for wagons
        color = (0, 255, 0)
        thickness = 2
        
        # Draw rectangle
        cv2.rectangle(
            annotated_frame,
            (bbox.x1, bbox.y1),
            (bbox.x2, bbox.y2),
            color,
            thickness
        )
        
        # Add label with confidence
        label = f"Wagon {detection.confidence:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # Draw label background
        cv2.rectangle(
            annotated_frame,
            (bbox.x1, bbox.y1 - label_size[1] - 10),
            (bbox.x1 + label_size[0] + 10, bbox.y1),
            color,
            -1  # Filled
        )
        
        # Draw label text
        cv2.putText(
            annotated_frame,
            label,
            (bbox.x1 + 5, bbox.y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),  # Black text
            1
        )
    
    # Draw damage detections in red
    for detection in damage_detections:
        bbox = detection.bbox
        # Red color for damage
        color = (0, 0, 255)
        thickness = 2
        
        # Draw rectangle
        cv2.rectangle(
            annotated_frame,
            (bbox.x1, bbox.y1),
            (bbox.x2, bbox.y2),
            color,
            thickness
        )
        
        # Add label with damage type and confidence
        damage_type = detection.damage_class.value.replace("_", " ").title()
        label = f"{damage_type} {detection.confidence:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # Draw label background
        cv2.rectangle(
            annotated_frame,
            (bbox.x1, bbox.y1 - label_size[1] - 10),
            (bbox.x1 + label_size[0] + 10, bbox.y1),
            color,
            -1  # Filled
        )
        
        # Draw label text
        cv2.putText(
            annotated_frame,
            label,
            (bbox.x1 + 5, bbox.y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),  # White text
            1
        )
    
    return annotated_frame


class MissionControlDashboard:
    """
    Main dashboard class for the Mission Control interface.
    
    Orchestrates the video feed, metrics display, and detection logging
    for the railway wagon inspection system.
    
    Attributes:
        video_manager: VideoManager for handling video capture
        metrics_calculator: MetricsCalculator for performance metrics
        wagon_detector: WagonDetector for wagon detection
        damage_detector: DamageDetector for damage detection
        dual_display: DualVideoDisplay for side-by-side video display
        track_renderer: TrackIDRenderer for drawing track IDs
        frame_saver: FrameSaver for saving processed frames
        ocr_log_display: OCRLogDisplay for OCR log management
    """
    
    # Default model paths
    # NOTE: The model files are named opposite to their function:
    # - damage_detector.pt actually contains wagon_body/wheel classes
    # - wagon_detector.pt actually contains damage classes
    DEFAULT_WAGON_MODEL_PATH = "models/damage_detector.pt"  # Contains: wagon_body, wheel
    DEFAULT_DAMAGE_MODEL_PATH = "models/wagon_detector.pt"  # Contains: damage classes
    
    def __init__(self):
        """
        Initialize dashboard with page config, CSS, and all components.
        
        Initializes:
        - Page configuration and CSS styling
        - Session state with default values
        - VideoManager for video capture
        - MetricsCalculator for performance metrics
        - Detection pipeline components (wagon and damage detectors)
        - DualVideoDisplay for side-by-side video display (Requirement 1.1)
        - TrackIDRenderer for wagon track ID overlays (Requirement 2.1)
        - FrameSaver for automatic frame saving (Requirement 5.1)
        - OCRLogDisplay for OCR log management (Requirement 6.1)
        """
        # Page configuration must be the first Streamlit command
        st.set_page_config(
            page_title="RWIS - Railway Inspection",
            page_icon="🚂",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Inject custom CSS styling
        inject_css()
        
        # Initialize components
        self._init_session_state()
        
        # Use session state to persist video manager across reruns
        if "video_manager" not in st.session_state:
            st.session_state.video_manager = VideoManager()
        self.video_manager = st.session_state.video_manager
        
        # Use session state to persist metrics calculator
        if "metrics_calculator" not in st.session_state:
            st.session_state.metrics_calculator = MetricsCalculator()
        self.metrics_calculator = st.session_state.metrics_calculator
        
        # Initialize detection pipeline components (cached in session state)
        self.wagon_detector = None
        self.damage_detector = None
        self._init_detectors()
        
        # Initialize DualVideoDisplay for side-by-side video display (Requirement 1.1)
        if "dual_display" not in st.session_state:
            st.session_state.dual_display = DualVideoDisplay()
        self.dual_display = st.session_state.dual_display
        
        # Initialize TrackIDRenderer for wagon track ID overlays (Requirement 2.1)
        if "track_renderer" not in st.session_state:
            st.session_state.track_renderer = TrackIDRenderer()
        self.track_renderer = st.session_state.track_renderer
        
        # Initialize FrameSaver with config from session state (Requirement 5.1)
        frame_save_config = get_frame_save_config_from_session()
        if "frame_saver" not in st.session_state:
            st.session_state.frame_saver = FrameSaver(frame_save_config)
        self.frame_saver = st.session_state.frame_saver
        # Update config in case it changed
        self.frame_saver.update_config(frame_save_config)
        
        # Initialize OCRLogDisplay for OCR log management (Requirement 6.1)
        if "ocr_log_display" not in st.session_state:
            st.session_state.ocr_log_display = OCRLogDisplay()
        self.ocr_log_display = st.session_state.ocr_log_display
        
        # Initialize IlluminationController for frontend controls (Requirements 5.1, 5.2, 5.3, 8.1, 8.2)
        if "illumination_controller" not in st.session_state:
            st.session_state.illumination_controller = IlluminationController(
                gamma_value=st.session_state.get("illumination_gamma", 1.0),
                low_light_threshold=st.session_state.get("illumination_low_light_threshold", 80),
                auto_mode=st.session_state.get("illumination_auto_mode", True)
            )
        self.illumination_controller = st.session_state.illumination_controller
        
        # Initialize CalibrationManager for auto-calibration (Requirements 10.1, 10.2, 10.3, 10.4, 10.5, 10.6)
        if "calibration_manager" not in st.session_state:
            st.session_state.calibration_manager = CalibrationManager(
                sample_size=30,
                blur_percentile=50,
                luminance_percentile=25
            )
        self.calibration_manager = st.session_state.calibration_manager
        
        # Initialize BlurDetector for blur detection and threshold management (Requirements 3.5, 3.6, 6.4)
        if "blur_detector" not in st.session_state:
            # Use saved blur threshold or default
            blur_threshold = st.session_state.get("blur_threshold", 100.0)
            blur_auto_mode = st.session_state.get("blur_auto_mode", True)
            # t1 is set to half of t2 as a reasonable default for the lower threshold
            st.session_state.blur_detector = BlurDetector(
                t1=blur_threshold * 0.5,
                t2=blur_threshold,
                auto_mode=blur_auto_mode
            )
        self.blur_detector = st.session_state.blur_detector
        
        # Initialize OCR pipeline for text extraction (Requirements 1.1, 1.2, 1.3, 1.4, 1.5)
        if "ocr_pipeline" not in st.session_state:
            st.session_state.ocr_pipeline = OCRPipeline(
                gpu_enabled=True,
                language='en',
                low_light_threshold=st.session_state.get("illumination_low_light_threshold", 80),
                gamma_value=st.session_state.get("illumination_gamma", 1.5)
            )
        self.ocr_pipeline = st.session_state.ocr_pipeline
        
        # Initialize DeblurManager for ROI deblurring (Requirements 4.3, 4.4, 7.1, 7.2, 7.3)
        if "deblur_manager" not in st.session_state:
            # Check if MPRNet model exists
            mprnet_model_path = "model_best.pth"
            if os.path.exists(mprnet_model_path):
                try:
                    # Initialize MPRNet wrapper (CPU mode for sm_120 compatibility)
                    mprnet = MPRNetDeblur(
                        model_path=mprnet_model_path,
                        device='cpu',
                        use_fp16=False,
                        fp32_fallback=True,
                        max_roi_width=256,
                        max_roi_height=256
                    )
                    mprnet.load_model()
                    
                    # Initialize DeblurManager with MPRNet and BlurDetector
                    st.session_state.deblur_manager = DeblurManager(
                        mprnet=mprnet,
                        blur_detector=self.blur_detector,
                        frame_interval=3,
                        max_roi_width=256,
                        enable_clahe=True,
                        deblur_enabled=st.session_state.get("deblur_enabled", True)
                    )
                except Exception as e:
                    st.warning(f"Failed to initialize DeblurManager: {e}")
                    st.session_state.deblur_manager = None
            else:
                st.session_state.deblur_manager = None
        self.deblur_manager = st.session_state.get("deblur_manager")
        
        # Initialize OCRIntervalController for OCR frame interval control (Requirements 4.1, 4.2, 4.3, 4.4)
        if "ocr_interval_controller" not in st.session_state:
            st.session_state.ocr_interval_controller = OCRIntervalController(
                interval=st.session_state.get("ocr_frame_interval", 5)
            )
        self.ocr_interval_controller = st.session_state.ocr_interval_controller
        
        # Initialize OCRVisualization for drawing OCR boxes and overlays (Requirements 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 2.5)
        if "ocr_visualization" not in st.session_state:
            st.session_state.ocr_visualization = OCRVisualization()
        self.ocr_visualization = st.session_state.ocr_visualization
        
        # Initialize OCRFrameSaver for saving OCR frames with metadata (Requirements 3.1, 3.2, 3.3, 3.4, 6.2)
        ocr_output_directory = st.session_state.get("ocr_frame_output_directory", "outputs/ocr_frames")
        if "ocr_frame_saver" not in st.session_state:
            st.session_state.ocr_frame_saver = OCRFrameSaver(output_directory=ocr_output_directory)
        self.ocr_frame_saver = st.session_state.ocr_frame_saver
        # Update output directory in case it changed in session state
        self.ocr_frame_saver.output_directory = ocr_output_directory
        
        # Initialize WagonTracker for proper wagon counting (Requirements 2.5, 2.6, 2.7)
        if "wagon_tracker" not in st.session_state:
            from tracking.tracker import WagonTracker
            counting_line_pos = st.session_state.get("counting_line_position", 0.5)
            counting_line_orientation = st.session_state.get("counting_line_orientation", "vertical")
            st.session_state.wagon_tracker = WagonTracker(
                counting_line_y=counting_line_pos,
                orientation=counting_line_orientation
            )
        self.wagon_tracker = st.session_state.wagon_tracker
    
    def _init_detectors(self) -> None:
        """
        Initialize wagon and damage detectors if models are available.
        
        Attempts to load the YOLO models for wagon and damage detection.
        If models are not found, detectors remain None and detection is disabled.
        Uses session state to cache detectors across reruns.
        """
        # Check if detectors are already cached in session state
        if "wagon_detector" in st.session_state:
            self.wagon_detector = st.session_state.wagon_detector
        elif os.path.exists(self.DEFAULT_WAGON_MODEL_PATH):
            try:
                confidence = st.session_state.get("confidence_threshold", 0.5)
                self.wagon_detector = WagonDetector(
                    model_path=self.DEFAULT_WAGON_MODEL_PATH,
                    confidence_threshold=confidence
                )
                st.session_state.wagon_detector = self.wagon_detector
            except Exception as e:
                st.warning(f"Failed to load wagon detector: {e}")
                self.wagon_detector = None
        
        if "damage_detector" in st.session_state:
            self.damage_detector = st.session_state.damage_detector
        elif os.path.exists(self.DEFAULT_DAMAGE_MODEL_PATH):
            try:
                confidence = st.session_state.get("confidence_threshold", 0.5)
                self.damage_detector = DamageDetector(
                    model_path=self.DEFAULT_DAMAGE_MODEL_PATH,
                    confidence_threshold=confidence
                )
                st.session_state.damage_detector = self.damage_detector
            except Exception as e:
                st.warning(f"Failed to load damage detector: {e}")
                self.damage_detector = None
    
    def _process_frame_through_pipeline(
        self,
        frame: np.ndarray,
        frame_index: int
    ) -> Tuple[np.ndarray, int, int, bool, List[DamageDetection], List[TrackedWagon]]:
        """
        Process a frame through the wagon and damage detection pipeline.
        
        Args:
            frame: Input BGR frame
            frame_index: Current frame index
            
        Returns:
            Tuple of:
                - annotated_frame: Frame with bounding boxes drawn
                - object_count: Number of wagons detected
                - damage_count: Number of damages detected
                - damage_detected: Whether any damage was detected
                - damage_detections: List of damage detections for logging
                - tracked_wagons: List of tracked wagons for track ID display
                
        Requirements: 2.1, 3.4, 3.5, 4.3, 5.5, 6.1, 6.2
        """
        wagon_detections = []
        damage_detections = []
        tracked_wagons = []
        object_count = 0
        damage_count = 0
        damage_detected = False
        
        # Track if deblur was applied for frame saving
        deblur_applied_this_frame = False
        
        # Track if OCR was applied this frame (Requirements 4.4, 4.6)
        ocr_applied_this_frame = False
        
        # Check if OCR should run on this frame based on interval (Requirements 4.4, 4.6)
        should_run_ocr = False
        if hasattr(self, 'ocr_interval_controller') and self.ocr_interval_controller is not None:
            should_run_ocr = self.ocr_interval_controller.should_run_ocr(frame_index)
            # Update OCR status in session state (Requirements 5.2, 5.3)
            st.session_state.ocr_status = self.ocr_interval_controller.get_status_text(frame_index)
        else:
            # If no interval controller, always run OCR
            should_run_ocr = True
            st.session_state.ocr_status = "ACTIVE"
        
        # Store OCR detections for visualization
        current_ocr_detections = []
        
        # Run wagon detection if detector is available
        if self.wagon_detector is not None:
            try:
                wagon_detections = self.wagon_detector.detect(frame)
                object_count = len(wagon_detections)
                
                # Use WagonTracker for proper counting with ByteTrack
                if hasattr(self, 'wagon_tracker') and self.wagon_tracker is not None:
                    frame_shape = (frame.shape[0], frame.shape[1], frame.shape[2])
                    tracked_wagons = self.wagon_tracker.update(wagon_detections, frame_shape, frame_index)
                else:
                    # Fallback: Create TrackedWagon objects with sequential IDs (no proper tracking)
                    for idx, wagon in enumerate(wagon_detections):
                        tracked_wagon = TrackedWagon(
                            track_id=idx + 1,  # Simple sequential ID
                            bbox=wagon.bbox,
                            confidence=wagon.confidence,
                            crossed_line=False,
                            count_index=None
                        )
                        tracked_wagons.append(tracked_wagon)
            except Exception as e:
                # Log error but continue processing
                pass
        
        # Run damage detection on each wagon ROI if detector is available
        if self.damage_detector is not None and wagon_detections:
            for wagon_idx, wagon in enumerate(wagon_detections):
                try:
                    # Extract wagon ROI
                    roi, _ = self.wagon_detector.extract_roi(frame, wagon)
                    
                    if roi is not None and roi.size > 0:
                        # Apply deblurring to ROI before damage detection and OCR
                        # This improves both damage detection accuracy and OCR quality
                        processed_roi = roi
                        if hasattr(self, 'deblur_manager') and self.deblur_manager is not None:
                            try:
                                # Update deblur_enabled from session state
                                deblur_enabled = st.session_state.get("deblur_enabled", True)
                                self.deblur_manager.set_deblur_enabled(deblur_enabled)
                                
                                # Process ROI through deblur manager
                                processed_roi, was_deblurred, source_frame = self.deblur_manager.process_roi(
                                    roi=roi,
                                    wagon_id=wagon_idx,
                                    frame_index=frame_index
                                )
                                
                                if was_deblurred:
                                    deblur_applied_this_frame = True
                                    
                                # Update deblur status in session state for UI display
                                status = self.deblur_manager.get_last_status()
                                if status is not None:
                                    st.session_state.deblur_status_type = status.status_type
                                    if status.status_type == DeblurStatusType.ACTIVE:
                                        st.session_state.deblur_status = "ACTIVE"
                                    elif status.status_type == DeblurStatusType.SKIPPED:
                                        st.session_state.deblur_status = "SKIPPED"
                                    else:
                                        st.session_state.deblur_status = "DISABLED"
                            except Exception as deblur_error:
                                # If deblurring fails, use original ROI
                                processed_roi = roi
                        
                        # Run damage detection on the (possibly deblurred) ROI
                        damages = self.damage_detector.detect(processed_roi, wagon_id=wagon_idx)
                        
                        # Adjust damage bounding boxes to frame coordinates
                        for damage in damages:
                            # Offset damage bbox by wagon bbox origin
                            adjusted_bbox = BoundingBox(
                                x1=wagon.bbox.x1 + damage.bbox.x1,
                                y1=wagon.bbox.y1 + damage.bbox.y1,
                                x2=wagon.bbox.x1 + damage.bbox.x2,
                                y2=wagon.bbox.y1 + damage.bbox.y2
                            )
                            # Create new detection with adjusted bbox
                            adjusted_damage = DamageDetection(
                                damage_class=damage.damage_class,
                                bbox=adjusted_bbox,
                                confidence=damage.confidence,
                                wagon_id=wagon_idx
                            )
                            damage_detections.append(adjusted_damage)
                        
                        # Run OCR on the (possibly deblurred) ROI to extract text (Requirements 1.1, 1.5, 4.4, 4.6)
                        if should_run_ocr and hasattr(self, 'ocr_pipeline') and self.ocr_pipeline is not None:
                            try:
                                ocr_result = self.ocr_pipeline.extract_text(processed_roi, min_confidence=0.3)
                                if ocr_result.text and ocr_result.confidence > 0.3:
                                    ocr_applied_this_frame = True
                                    
                                    # Update last detected text and confidence in session state (Requirements 5.1, 5.4)
                                    st.session_state.ocr_last_text = ocr_result.text
                                    st.session_state.ocr_last_confidence = ocr_result.confidence
                                    
                                    # Create OCR detection for visualization (Requirements 1.1, 1.2, 1.3, 1.4)
                                    if ocr_result.bbox is not None:
                                        ocr_detection = OCRDetection(
                                            text=ocr_result.text,
                                            confidence=ocr_result.confidence,
                                            bbox=ocr_result.bbox,
                                            wagon_id=wagon_idx,
                                            frame_index=frame_index
                                        )
                                        current_ocr_detections.append((ocr_detection, wagon.bbox))
                                    
                                    # Create OCR log entry
                                    ocr_entry = OCRLogEntry(
                                        timestamp=datetime.now(),
                                        wagon_id=wagon_idx + 1,
                                        extracted_text=ocr_result.text,
                                        confidence=ocr_result.confidence,
                                        frame_index=frame_index
                                    )
                                    # Append to OCR log display
                                    if hasattr(self, 'ocr_log_display'):
                                        self.ocr_log_display.append_entry(ocr_entry)
                            except Exception as ocr_error:
                                # Log OCR error but continue processing
                                pass
                except Exception as e:
                    # Log error but continue processing
                    pass
        
        # Update damage counts
        damage_count = len(damage_detections)
        damage_detected = damage_count > 0
        
        # Run OCR on wagon ROIs even if damage detector is not available (Requirements 1.1, 1.5, 4.4, 4.6)
        # This ensures OCR runs on all detected wagons when interval allows
        if should_run_ocr and hasattr(self, 'ocr_pipeline') and self.ocr_pipeline is not None and wagon_detections:
            if self.damage_detector is None:  # Only run here if not already run in damage detection loop
                for wagon_idx, wagon in enumerate(wagon_detections):
                    try:
                        # Extract wagon ROI
                        roi, _ = self.wagon_detector.extract_roi(frame, wagon)
                        
                        if roi is not None and roi.size > 0:
                            # Apply deblurring to ROI before OCR
                            processed_roi = roi
                            if hasattr(self, 'deblur_manager') and self.deblur_manager is not None:
                                try:
                                    deblur_enabled = st.session_state.get("deblur_enabled", True)
                                    self.deblur_manager.set_deblur_enabled(deblur_enabled)
                                    
                                    processed_roi, was_deblurred, source_frame = self.deblur_manager.process_roi(
                                        roi=roi,
                                        wagon_id=wagon_idx,
                                        frame_index=frame_index
                                    )
                                    
                                    if was_deblurred:
                                        deblur_applied_this_frame = True
                                except Exception:
                                    processed_roi = roi
                            
                            ocr_result = self.ocr_pipeline.extract_text(processed_roi, min_confidence=0.3)
                            if ocr_result.text and ocr_result.confidence > 0.3:
                                ocr_applied_this_frame = True
                                
                                # Update last detected text and confidence in session state (Requirements 5.1, 5.4)
                                st.session_state.ocr_last_text = ocr_result.text
                                st.session_state.ocr_last_confidence = ocr_result.confidence
                                
                                # Create OCR detection for visualization (Requirements 1.1, 1.2, 1.3, 1.4)
                                if ocr_result.bbox is not None:
                                    ocr_detection = OCRDetection(
                                        text=ocr_result.text,
                                        confidence=ocr_result.confidence,
                                        bbox=ocr_result.bbox,
                                        wagon_id=wagon_idx,
                                        frame_index=frame_index
                                    )
                                    current_ocr_detections.append((ocr_detection, wagon.bbox))
                                
                                # Create OCR log entry
                                ocr_entry = OCRLogEntry(
                                    timestamp=datetime.now(),
                                    wagon_id=wagon_idx + 1,
                                    extracted_text=ocr_result.text,
                                    confidence=ocr_result.confidence,
                                    frame_index=frame_index
                                )
                                # Append to OCR log display
                                if hasattr(self, 'ocr_log_display'):
                                    self.ocr_log_display.append_entry(ocr_entry)
                    except Exception as ocr_error:
                        # Log OCR error but continue processing
                        pass
        
        # Store deblur_applied flag in session state for frame saving
        st.session_state.deblur_applied_this_frame = deblur_applied_this_frame
        
        # Store OCR applied flag and detections in session state (Requirements 4.4, 4.6)
        st.session_state.ocr_applied_this_frame = ocr_applied_this_frame
        st.session_state.current_ocr_detections = current_ocr_detections
        
        # Draw bounding boxes on frame (Requirement 4.3)
        annotated_frame = draw_bounding_boxes(frame, wagon_detections, damage_detections)
        
        # Draw OCR bounding boxes and text overlays (Requirements 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 2.5)
        if current_ocr_detections and hasattr(self, 'ocr_visualization') and self.ocr_visualization is not None:
            for ocr_detection, wagon_bbox in current_ocr_detections:
                annotated_frame = self.ocr_visualization.draw_ocr_boxes(
                    frame=annotated_frame,
                    ocr_results=[ocr_detection],
                    wagon_bbox=wagon_bbox,
                    deblur_applied=deblur_applied_this_frame
                )
        
        # Save OCR frames with metadata when OCR detects text (Requirements 3.1, 3.2, 3.3, 6.2)
        if current_ocr_detections and hasattr(self, 'ocr_frame_saver') and self.ocr_frame_saver is not None:
            # Get illumination applied status
            illumination_applied = st.session_state.get("illumination_applied_this_frame", False)
            
            # Save frame for each wagon with OCR detections
            for ocr_detection, wagon_bbox in current_ocr_detections:
                # Build detections list for metadata
                detection_dict = {
                    "text": ocr_detection.text,
                    "confidence": ocr_detection.confidence,
                    "bbox": {
                        "x1": ocr_detection.bbox.x1,
                        "y1": ocr_detection.bbox.y1,
                        "x2": ocr_detection.bbox.x2,
                        "y2": ocr_detection.bbox.y2
                    }
                }
                
                # Create OCR frame metadata (Requirements 3.3, 6.2)
                ocr_metadata = OCRFrameMetadata(
                    timestamp=datetime.now().isoformat(),
                    frame_index=frame_index,
                    wagon_id=ocr_detection.wagon_id,
                    detections=[detection_dict],
                    deblur_applied=deblur_applied_this_frame,
                    illumination_applied=illumination_applied,
                    blur_score=st.session_state.get("current_blur_score"),
                    luminance_level=None  # Could be added if luminance tracking is implemented
                )
                
                # Save annotated frame with OCR boxes and metadata (Requirements 3.1, 3.2)
                try:
                    self.ocr_frame_saver.save_ocr_frame(
                        frame=annotated_frame,
                        ocr_detections=[ocr_detection],
                        metadata=ocr_metadata
                    )
                except Exception as save_error:
                    # Log error but continue processing
                    pass
        
        return annotated_frame, object_count, damage_count, damage_detected, damage_detections, tracked_wagons
    
    def _append_detection_log_entries(
        self,
        damage_detections: List[DamageDetection],
        frame_index: int
    ) -> None:
        """
        Append detection log entries for each NEW damage detection.
        
        Creates DetectionLogEntry objects and appends them to the session
        state detection log. Only logs new detections to prevent counting
        the same damage multiple times across frames.
        
        Args:
            damage_detections: List of damage detections to log
            frame_index: Current frame index
            
        Requirements: 5.5
        """
        detection_log = st.session_state.get("detection_log", [])
        logged_damages = st.session_state.get("logged_damage_keys", set())
        
        for damage in damage_detections:
            # Create a unique key for this damage based on wagon_id and damage_class
            # This prevents logging the same damage multiple times across frames
            damage_key = (damage.wagon_id, damage.damage_class.value)
            
            if damage_key not in logged_damages:
                entry = DetectionLogEntry(
                    timestamp=datetime.now(),
                    wagon_id=damage.wagon_id,
                    damage_type=damage.damage_class.value.replace("_", " ").title(),
                    confidence=damage.confidence,
                    frame_index=frame_index
                )
                detection_log.append(entry)
                logged_damages.add(damage_key)
        
        st.session_state.detection_log = detection_log
        st.session_state.logged_damage_keys = logged_damages
    
    def _init_session_state(self) -> None:
        """
        Initialize session state with default values.
        
        Initializes state for:
        - Video processing control (is_running, video_source)
        - Detection settings (confidence_threshold, frame_skip)
        - Detection logging (detection_log)
        - Connection status
        - Total wagon count tracking (Requirement 3.1)
        - Frame saving configuration (Requirements 8.1-8.5)
        - Tracked wagons list for track ID display (Requirement 2.1)
        - Raw frame storage for dual display (Requirement 1.2)
        - Illumination settings (Requirements 5.1, 5.2, 5.3, 5.5, 5.6, 8.1, 8.2, 8.6, 8.7)
        - Settings persistence (Requirements 9.1, 9.2)
        """
        # Load saved settings if available (Requirements 9.1, 9.2)
        saved_settings = None
        if "settings_loaded" not in st.session_state:
            try:
                settings_manager = SettingsManager()
                saved_settings = settings_manager.load_settings()
                st.session_state.settings_loaded = True
            except Exception:
                # If loading fails, use defaults
                saved_settings = None
                st.session_state.settings_loaded = True
        
        defaults = {
            "is_running": False,
            "video_source": "",
            "confidence_threshold": 0.5,
            "frame_skip": 3,
            "detection_log": [],
            "logged_damage_keys": set(),  # Set of (wagon_id, damage_class) tuples to prevent duplicate logging
            "connection_status": "Disconnected",
            "total_wagon_count": 0,
            # Frame saving configuration defaults (Requirements 8.1, 8.2, 8.3, 8.4, 8.5)
            "frame_save_enabled": False,
            "frame_save_on_deblur": True,
            "frame_save_on_illumination": True,
            "frame_save_on_ocr": True,
            "frame_save_output_directory": "outputs/saved_frames",
            # New state fields for dashboard enhancements
            "tracked_wagons": [],  # List of TrackedWagon for track ID display (Requirement 2.1)
            "raw_frame": None,  # Raw frame for dual display (Requirement 1.2)
            "crossed_wagon_ids": set(),  # Set of wagon IDs that crossed counting line (Requirement 3.3)
            "frame_index": 0,  # Current frame index for synchronization (Requirement 1.4)
            "last_frame": None,  # Last processed frame
            # Illumination settings - use saved values if available (Requirements 5.1, 5.2, 5.3, 5.5, 5.6, 8.1, 8.2, 8.6, 8.7, 9.1, 9.2)
            "illumination_gamma": saved_settings.gamma_value if saved_settings else 1.0,
            "illumination_low_light_threshold": saved_settings.low_light_threshold if saved_settings else 80,
            "illumination_auto_mode": saved_settings.illumination_auto_mode if saved_settings else True,
            "illumination_status": "Normal",  # Current illumination status display
            # Blur/Deblur settings - use saved values if available (Requirements 6.1, 6.2, 6.3, 6.5, 6.6, 7.4, 7.5, 7.6, 9.1, 9.2)
            "blur_threshold": saved_settings.blur_threshold if saved_settings else 100.0,
            "blur_auto_mode": saved_settings.blur_auto_mode if saved_settings else True,
            "deblur_enabled": saved_settings.deblur_enabled if saved_settings else True,
            "current_blur_score": 0.0,  # Current blur score display
            "deblur_status": "DISABLED",  # Deblur status: ACTIVE, SKIPPED, DISABLED
            "deblur_status_type": DeblurStatusType.DISABLED,  # Deblur status type enum
            # Calibration status (Requirements 10.1, 10.2, 10.3, 10.4, 10.5, 10.6)
            "calibration_blur_mode": CalibrationMode.AUTO if (saved_settings.blur_auto_mode if saved_settings else True) else CalibrationMode.MANUAL,
            "calibration_illumination_mode": CalibrationMode.AUTO if (saved_settings.illumination_auto_mode if saved_settings else True) else CalibrationMode.MANUAL,
            "calibration_progress": 0.0,  # Calibration progress (0.0 to 1.0)
            "calibration_complete": False,  # Whether calibration is complete
            "calibrated_blur_threshold": None,  # Computed blur threshold from calibration
            "calibrated_low_light_threshold": None,  # Computed low-light threshold from calibration
            "calibrated_gamma_value": None,  # Computed gamma value from calibration
            "calibration_sample_count": 0,  # Number of samples collected
            # OCR interval and status settings (Requirements 4.1, 4.2, 4.3, 4.5, 5.1, 5.2, 5.3, 5.4)
            "ocr_frame_interval": 5,  # Default OCR frame interval
            "ocr_status": "ACTIVE",  # Current OCR status display
            "ocr_last_text": "",  # Last detected OCR text
            "ocr_last_confidence": 0.0,  # Last OCR confidence score
            "ocr_applied_this_frame": False,  # Whether OCR was applied on current frame
            "current_ocr_detections": [],  # OCR detections for current frame
            # OCR frame saving settings (Requirements 3.1, 3.2, 3.3, 3.4, 6.2)
            "ocr_frame_output_directory": "outputs/ocr_frames",  # Output directory for OCR frames
            # Processing status indicators (Requirements 7.1, 7.2, 7.3, 7.4, 8.1)
            "illumination_applied_this_frame": False,  # Whether illumination enhancement was applied on current frame
            "deblur_applied_this_frame": False,  # Whether deblur was applied on current frame (moved here for clarity)
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def render_sidebar(self) -> SidebarSettings:
        """
        Render sidebar controls and return settings.
        
        Creates the sidebar UI with:
        - Video source input (RTSP URL or file path)
        - Confidence threshold slider (0.0 to 1.0)
        - Frame skip number input (1 to 10)
        - Start/Stop control buttons
        - Connection status indicator
        
        Returns:
            SidebarSettings dataclass with current values
            
        Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6
        """
        with st.sidebar:
            st.header("⚙️ Controls")
            
            # Video source input (Requirement 2.1)
            video_source = st.text_input(
                "Video Source",
                value=st.session_state.get("video_source", ""),
                placeholder="RTSP URL or file path",
                help="Enter an RTSP URL (rtsp://...) or path to a video file"
            )
            st.session_state.video_source = video_source
            
            st.divider()
            
            # Model settings section
            st.subheader("Model Settings")
            
            # Confidence threshold slider (Requirement 2.2)
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.get("confidence_threshold", 0.5),
                step=0.05,
                help="Minimum confidence score for detections"
            )
            st.session_state.confidence_threshold = confidence_threshold
            
            # Frame skip number input (Requirement 2.2)
            frame_skip = st.number_input(
                "Frame Skip",
                min_value=1,
                max_value=10,
                value=st.session_state.get("frame_skip", 3),
                step=1,
                help="Process every Nth frame (higher = faster, lower = smoother)"
            )
            st.session_state.frame_skip = frame_skip
            
            st.divider()
            
            # Control buttons section
            st.subheader("Stream Control")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Start button (Requirement 2.4)
                start_clicked = st.button(
                    "▶️ Start",
                    use_container_width=True,
                    disabled=st.session_state.get("is_running", False)
                )
                if start_clicked:
                    if video_source:
                        # Update video manager frame skip before connecting
                        self.video_manager.frame_skip = frame_skip
                        # Attempt to connect
                        if self.video_manager.connect(video_source):
                            st.session_state.is_running = True
                            st.session_state.connection_status = "Connected"
                            st.rerun()  # Rerun to start processing
                        else:
                            st.session_state.connection_status = "Error"
                            st.session_state.is_running = False
                            st.error("Failed to connect to video source")
                    else:
                        st.warning("Please enter a video source")
            
            with col2:
                # Stop button (Requirement 2.5)
                stop_clicked = st.button(
                    "⏹️ Stop",
                    use_container_width=True,
                    disabled=not st.session_state.get("is_running", False)
                )
                if stop_clicked:
                    st.session_state.is_running = False
                    self.video_manager.release()
                    st.session_state.connection_status = "Disconnected"
            
            st.divider()
            
            # Connection status indicator (Requirement 2.6)
            st.subheader("Status")
            connection_status = st.session_state.get("connection_status", "Disconnected")
            
            if connection_status == "Connected":
                st.success(f"🟢 {connection_status}")
            elif connection_status == "Error":
                st.error(f"🔴 {connection_status}")
            else:
                st.info(f"⚪ {connection_status}")
            
            st.divider()
            
            # Frame Saving section (Requirements 8.1, 8.2, 8.3, 8.4, 8.5)
            st.subheader("💾 Frame Saving")
            
            # Enable/disable toggle for frame saving (Requirement 8.1)
            frame_save_enabled = st.toggle(
                "Enable Frame Saving",
                value=st.session_state.get("frame_save_enabled", False),
                help="Automatically save frames when processing is applied"
            )
            st.session_state.frame_save_enabled = frame_save_enabled
            
            # Checkboxes for selecting which processing types trigger saves (Requirement 8.2)
            # Only show options when frame saving is enabled
            if frame_save_enabled:
                st.markdown("**Save frames on:**")
                
                save_on_deblur = st.checkbox(
                    "Deblur processing",
                    value=st.session_state.get("frame_save_on_deblur", True),
                    help="Save frames when deblur processing is applied"
                )
                st.session_state.frame_save_on_deblur = save_on_deblur
                
                save_on_illumination = st.checkbox(
                    "Illumination enhancement",
                    value=st.session_state.get("frame_save_on_illumination", True),
                    help="Save frames when CLAHE or gamma correction is applied"
                )
                st.session_state.frame_save_on_illumination = save_on_illumination
                
                save_on_ocr = st.checkbox(
                    "OCR extraction",
                    value=st.session_state.get("frame_save_on_ocr", True),
                    help="Save frames when OCR text extraction is performed"
                )
                st.session_state.frame_save_on_ocr = save_on_ocr
                
                # Output directory path input (Requirement 8.3)
                output_directory = st.text_input(
                    "Output Directory",
                    value=st.session_state.get("frame_save_output_directory", "outputs/saved_frames"),
                    help="Directory path where saved frames will be stored"
                )
                st.session_state.frame_save_output_directory = output_directory
            
            st.divider()
            
            # Illumination Settings section (Requirements 5.1, 5.2, 5.3, 5.5, 5.6, 8.1, 8.2, 8.6, 8.7)
            st.subheader("💡 Illumination Settings")
            
            # Auto/manual mode toggle (Requirement 5.6)
            illumination_auto_mode = st.toggle(
                "Auto Mode",
                value=st.session_state.get("illumination_auto_mode", True),
                help="When enabled, illumination settings are auto-calibrated. Disable for manual control."
            )
            st.session_state.illumination_auto_mode = illumination_auto_mode
            
            # Update calibration mode based on auto/manual toggle (Requirement 10.2)
            st.session_state.calibration_illumination_mode = CalibrationMode.AUTO if illumination_auto_mode else CalibrationMode.MANUAL
            
            # Update illumination controller auto mode
            if hasattr(self, 'illumination_controller'):
                self.illumination_controller.set_auto_mode(illumination_auto_mode)
            
            # Gamma value slider (Requirement 5.2, 8.1, 8.2)
            # Only enable manual adjustment when auto mode is off
            gamma_value = st.slider(
                "Gamma Value",
                min_value=0.5,
                max_value=3.0,
                value=st.session_state.get("illumination_gamma", 1.0),
                step=0.1,
                disabled=illumination_auto_mode,
                help="Gamma correction value. < 1.0 brightens, > 1.0 darkens the image."
            )
            st.session_state.illumination_gamma = gamma_value
            
            # Update illumination controller gamma value
            if hasattr(self, 'illumination_controller') and not illumination_auto_mode:
                self.illumination_controller.set_gamma_value(gamma_value)
            
            # Low-light threshold slider (Requirement 5.3)
            low_light_threshold = st.slider(
                "Low-Light Threshold",
                min_value=0,
                max_value=255,
                value=st.session_state.get("illumination_low_light_threshold", 80),
                step=5,
                disabled=illumination_auto_mode,
                help="Luminance threshold for low-light detection. Images below this are considered low-light."
            )
            st.session_state.illumination_low_light_threshold = low_light_threshold
            
            # Update illumination controller threshold
            if hasattr(self, 'illumination_controller') and not illumination_auto_mode:
                self.illumination_controller.set_low_light_threshold(low_light_threshold)
            
            # Current illumination status display (Requirement 5.5)
            st.markdown("**Current Status:**")
            illumination_status = st.session_state.get("illumination_status", "Normal")
            if illumination_status == "Low-Light":
                st.warning(f"🌙 {illumination_status}")
            else:
                st.success(f"☀️ {illumination_status}")
            
            # Display current gamma value being applied (Requirement 8.6)
            current_gamma = st.session_state.get("illumination_gamma", 1.0)
            st.caption(f"Current gamma: {current_gamma:.2f}")
            
            # Reset to Auto button (Requirement 8.7)
            if st.button(
                "🔄 Reset to Auto",
                use_container_width=True,
                help="Reset illumination settings to auto-calibration mode"
            ):
                # Reset to default auto values
                st.session_state.illumination_auto_mode = True
                st.session_state.illumination_gamma = 1.0
                st.session_state.illumination_low_light_threshold = 80
                
                # Update illumination controller
                if hasattr(self, 'illumination_controller'):
                    self.illumination_controller.set_auto_mode(True)
                    self.illumination_controller.set_gamma_value(1.0)
                    self.illumination_controller.set_low_light_threshold(80)
                
                st.rerun()
            
            st.divider()
            
            # Blur/Deblur Settings section (Requirements 6.1, 6.2, 6.3, 6.5, 6.6)
            st.subheader("🔍 Blur/Deblur Settings")
            
            # Auto/manual mode toggle for blur (Requirement 6.6)
            blur_auto_mode = st.toggle(
                "Auto Mode (Blur)",
                value=st.session_state.get("blur_auto_mode", True),
                help="When enabled, blur threshold is auto-calibrated. Disable for manual control."
            )
            st.session_state.blur_auto_mode = blur_auto_mode
            
            # Update calibration mode based on auto/manual toggle (Requirement 10.1)
            st.session_state.calibration_blur_mode = CalibrationMode.AUTO if blur_auto_mode else CalibrationMode.MANUAL
            
            # Blur threshold slider (Requirement 6.2)
            # Only enable manual adjustment when auto mode is off
            blur_threshold = st.slider(
                "Blur Threshold",
                min_value=0,
                max_value=1000,
                value=int(st.session_state.get("blur_threshold", 100)),
                step=10,
                disabled=blur_auto_mode,
                help="Laplacian variance threshold. Higher = stricter blur detection. Images below this are considered blurry."
            )
            st.session_state.blur_threshold = float(blur_threshold)
            
            # Update blur detector threshold in real-time (Requirement 6.4)
            if hasattr(self, 'blur_detector') and not blur_auto_mode:
                try:
                    self.blur_detector.set_threshold(float(blur_threshold))
                except ValueError:
                    # If threshold is too low, adjust t1 first
                    self.blur_detector.t1 = float(blur_threshold) * 0.5
                    self.blur_detector.set_threshold(float(blur_threshold))
            
            # Update blur detector auto mode
            if hasattr(self, 'blur_detector'):
                self.blur_detector.set_auto_mode(blur_auto_mode)
            
            # Deblur enable/disable toggle (Requirement 6.3)
            deblur_enabled = st.toggle(
                "Enable Deblurring",
                value=st.session_state.get("deblur_enabled", True),
                help="When enabled, blurry images will be processed through the deblur pipeline."
            )
            st.session_state.deblur_enabled = deblur_enabled
            
            # Current blur score display (Requirement 6.5)
            st.markdown("**Current Blur Score:**")
            current_blur_score = st.session_state.get("current_blur_score", 0.0)
            st.metric(
                label="Blur Score",
                value=f"{current_blur_score:.1f}",
                help="Current frame's blur score (Laplacian variance). Higher = sharper image."
            )
            
            # Deblur status indicator (Requirements 2.2, 2.3, 2.4, 7.4, 7.5, 7.6)
            st.markdown("**Deblur Status:**")
            deblur_status = st.session_state.get("deblur_status", "DISABLED")
            deblur_status_type = st.session_state.get("deblur_status_type", DeblurStatusType.DISABLED)
            
            # Color coding: green (ACTIVE), yellow (SKIPPED), gray (DISABLED)
            if deblur_status_type == DeblurStatusType.ACTIVE:
                st.success(f"🟢 {deblur_status}")
            elif deblur_status_type == DeblurStatusType.SKIPPED:
                st.warning(f"🟡 {deblur_status}")
            else:  # DISABLED
                st.markdown(f"⚪ {deblur_status}")
            
            st.divider()
            
            # Calibration Status section (Requirements 10.1, 10.2, 10.3, 10.4, 10.5, 10.6)
            st.subheader("📊 Calibration Status")
            
            # Display current calibration mode for blur (Requirement 10.1)
            blur_mode = st.session_state.get("calibration_blur_mode", CalibrationMode.AUTO)
            blur_mode_text = "Auto" if blur_mode == CalibrationMode.AUTO else "Manual"
            st.markdown(f"**Blur Mode:** {blur_mode_text}")
            
            # Display current calibration mode for illumination (Requirement 10.2)
            illumination_mode = st.session_state.get("calibration_illumination_mode", CalibrationMode.AUTO)
            illumination_mode_text = "Auto" if illumination_mode == CalibrationMode.AUTO else "Manual"
            st.markdown(f"**Illumination Mode:** {illumination_mode_text}")
            
            # Display computed threshold values when in auto mode (Requirement 10.3)
            calibration_complete = st.session_state.get("calibration_complete", False)
            
            if calibration_complete:
                st.markdown("**Computed Thresholds:**")
                
                # Display calibrated blur threshold
                calibrated_blur = st.session_state.get("calibrated_blur_threshold")
                if calibrated_blur is not None and blur_mode == CalibrationMode.AUTO:
                    st.caption(f"• Blur threshold: {calibrated_blur:.1f}")
                
                # Display calibrated low-light threshold
                calibrated_low_light = st.session_state.get("calibrated_low_light_threshold")
                if calibrated_low_light is not None and illumination_mode == CalibrationMode.AUTO:
                    st.caption(f"• Low-light threshold: {calibrated_low_light}")
                
                # Display calibrated gamma value
                calibrated_gamma = st.session_state.get("calibrated_gamma_value")
                if calibrated_gamma is not None and illumination_mode == CalibrationMode.AUTO:
                    st.caption(f"• Gamma value: {calibrated_gamma:.2f}")
                
                # Display sample count
                sample_count = st.session_state.get("calibration_sample_count", 0)
                st.caption(f"• Samples used: {sample_count}")
            else:
                # Display calibration progress indicator during initial analysis (Requirement 10.4)
                calibration_progress = st.session_state.get("calibration_progress", 0.0)
                sample_count = st.session_state.get("calibration_sample_count", 0)
                sample_size = 30  # Default sample size
                
                if hasattr(self, 'calibration_manager'):
                    sample_size = self.calibration_manager.sample_size
                
                st.markdown("**Calibration Progress:**")
                st.progress(calibration_progress, text=f"Collecting samples: {sample_count}/{sample_size}")
                
                if calibration_progress < 1.0:
                    st.info("🔄 Calibrating... Start video to collect samples.")
            
            # Recalibrate button (Requirement 10.5, 10.6)
            if st.button(
                "🔄 Recalibrate",
                use_container_width=True,
                help="Reset calibration and collect new samples from the video"
            ):
                # Reset calibration manager
                if hasattr(self, 'calibration_manager'):
                    self.calibration_manager.reset()
                
                # Reset calibration session state
                st.session_state.calibration_complete = False
                st.session_state.calibration_progress = 0.0
                st.session_state.calibration_sample_count = 0
                st.session_state.calibrated_blur_threshold = None
                st.session_state.calibrated_low_light_threshold = None
                st.session_state.calibrated_gamma_value = None
                
                # Reset to auto mode
                st.session_state.calibration_blur_mode = CalibrationMode.AUTO
                st.session_state.calibration_illumination_mode = CalibrationMode.AUTO
                st.session_state.blur_auto_mode = True
                st.session_state.illumination_auto_mode = True
                
                st.success("Calibration reset. Start video to recalibrate.")
                st.rerun()
            
            st.divider()
            
            # OCR Settings section (Requirements 4.1, 4.2, 4.3, 4.5, 5.1, 5.2, 5.3, 5.4)
            st.subheader("📝 OCR Settings")
            
            # OCR Frame Interval slider (Requirements 4.1, 4.2, 4.3, 4.5)
            ocr_frame_interval = st.slider(
                "OCR Frame Interval",
                min_value=1,
                max_value=30,
                value=st.session_state.get("ocr_frame_interval", 5),
                step=1,
                help="Run OCR every Nth frame. Higher values improve performance, lower values increase detection coverage."
            )
            st.session_state.ocr_frame_interval = ocr_frame_interval
            
            # Update OCR interval controller
            if hasattr(self, 'ocr_interval_controller'):
                self.ocr_interval_controller.interval = ocr_frame_interval
            
            # Display current interval value (Requirement 4.5)
            st.caption(f"OCR runs every {ocr_frame_interval} frame(s)")
            
            # OCR Status section (Requirements 5.1, 5.2, 5.3, 5.4)
            st.markdown("**OCR Status:**")
            
            # Show ACTIVE/SKIPPED status (Requirements 5.2, 5.3)
            ocr_status = st.session_state.get("ocr_status", "ACTIVE")
            if "ACTIVE" in ocr_status:
                st.success(f"🟢 {ocr_status}")
            else:
                st.warning(f"🟡 {ocr_status}")
            
            # Show last detected text (truncated) (Requirement 5.4)
            ocr_last_text = st.session_state.get("ocr_last_text", "")
            if ocr_last_text:
                # Truncate to 50 characters with ellipsis if longer
                if hasattr(self, 'ocr_visualization'):
                    truncated_text = self.ocr_visualization.truncate_text(ocr_last_text)
                else:
                    truncated_text = ocr_last_text[:50] + "..." if len(ocr_last_text) > 50 else ocr_last_text
                st.markdown(f"**Last Text:** {truncated_text}")
            else:
                st.markdown("**Last Text:** _No text detected_")
            
            # Show last confidence (Requirement 5.1)
            ocr_last_confidence = st.session_state.get("ocr_last_confidence", 0.0)
            confidence_pct = int(ocr_last_confidence * 100)
            st.caption(f"Last confidence: {confidence_pct}%")
            
            st.divider()
            
            # Settings Persistence section (Requirements 9.6, 9.7)
            st.subheader("💾 Settings Persistence")
            
            # Save Settings button (Requirement 9.6)
            if st.button(
                "💾 Save Settings",
                use_container_width=True,
                help="Save current configuration to persist across sessions"
            ):
                try:
                    # Create PipelineSettings from current session state
                    settings_to_save = PipelineSettings(
                        blur_threshold=st.session_state.get("blur_threshold", 100.0),
                        blur_auto_mode=st.session_state.get("blur_auto_mode", True),
                        deblur_enabled=st.session_state.get("deblur_enabled", True),
                        low_light_threshold=st.session_state.get("illumination_low_light_threshold", 80),
                        gamma_value=st.session_state.get("illumination_gamma", 1.0),
                        illumination_auto_mode=st.session_state.get("illumination_auto_mode", True),
                        ocr_language='en',  # Default OCR language
                        ocr_gpu_enabled=True,  # Default GPU enabled
                        calibration_sample_size=30,  # Default sample size
                        blur_percentile=50.0,  # Default blur percentile
                        luminance_percentile=25.0  # Default luminance percentile
                    )
                    
                    # Initialize settings manager and save
                    settings_manager = SettingsManager()
                    settings_manager.save_settings(settings_to_save)
                    
                    st.success("✅ Settings saved successfully!")
                except Exception as e:
                    st.error(f"❌ Failed to save settings: {str(e)}")
            
            # Reset to Defaults button (Requirement 9.7)
            if st.button(
                "🔄 Reset to Defaults",
                use_container_width=True,
                help="Restore all settings to their default values"
            ):
                try:
                    # Initialize settings manager and reset to defaults
                    settings_manager = SettingsManager()
                    default_settings = settings_manager.reset_to_defaults()
                    
                    # Update session state with default values
                    st.session_state.blur_threshold = default_settings.blur_threshold
                    st.session_state.blur_auto_mode = default_settings.blur_auto_mode
                    st.session_state.deblur_enabled = default_settings.deblur_enabled
                    st.session_state.illumination_low_light_threshold = default_settings.low_light_threshold
                    st.session_state.illumination_gamma = default_settings.gamma_value
                    st.session_state.illumination_auto_mode = default_settings.illumination_auto_mode
                    
                    # Update calibration modes based on auto mode settings
                    st.session_state.calibration_blur_mode = CalibrationMode.AUTO if default_settings.blur_auto_mode else CalibrationMode.MANUAL
                    st.session_state.calibration_illumination_mode = CalibrationMode.AUTO if default_settings.illumination_auto_mode else CalibrationMode.MANUAL
                    
                    # Update illumination controller with default values
                    if hasattr(self, 'illumination_controller'):
                        self.illumination_controller.set_auto_mode(default_settings.illumination_auto_mode)
                        self.illumination_controller.set_gamma_value(default_settings.gamma_value)
                        self.illumination_controller.set_low_light_threshold(default_settings.low_light_threshold)
                    
                    st.success("✅ Settings reset to defaults!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed to reset settings: {str(e)}")
        
        # Return SidebarSettings dataclass with current values
        return SidebarSettings(
            video_source=video_source,
            confidence_threshold=confidence_threshold,
            frame_skip=frame_skip,
            enable_damage_detection=True  # Always enabled for now
        )
    
    def render_metrics_row(
        self,
        metrics_placeholder,
        fps: float,
        inference_ms: float,
        object_count: int,
        damage_count: int,
        damage_detected: bool,
        total_wagon_count: int = 0,
        latency_ms: float = 0.0,
        latency_warning: bool = False,
        illumination_status: str = "NORMAL",
        deblur_status: str = "NORMAL",
        ocr_status: str = "NORMAL"
    ) -> None:
        """
        Render the top metrics row with metrics and processing status indicators.
        
        Creates an 8-column layout displaying:
        - FPS (frames per second)
        - Processing Latency (milliseconds) with warning indicator
        - Object count (wagons detected in current frame)
        - Total Wagon Count (unique wagons that crossed counting line)
        - Damage count with status indicator
        - Illumination indicator (APPLIED/NORMAL/OFF)
        - Deblur indicator (APPLIED/SKIPPED/OFF)
        - OCR indicator (ACTIVE/SKIPPED/OFF)
        
        Uses st.empty() placeholder for efficient updates without
        causing sidebar or header flickering.
        
        Args:
            metrics_placeholder: st.empty() placeholder for updates
            fps: Current frames per second
            inference_ms: Inference latency in milliseconds
            object_count: Number of detected objects in current frame
            damage_count: Number of detected damages
            damage_detected: Whether damage is currently detected
            total_wagon_count: Total unique wagons that crossed counting line (Requirement 3.1)
            latency_ms: Processing latency in milliseconds (Requirement 4.1)
            latency_warning: Whether latency exceeds warning threshold (Requirement 4.5)
            illumination_status: Status for illumination indicator (Requirements 7.1, 7.2)
            deblur_status: Status for deblur indicator (Requirements 7.1, 7.3)
            ocr_status: Status for OCR indicator (Requirements 7.1, 8.1)
            
        Requirements: 3.1, 3.2, 3.3, 4.1, 7.1, 7.2, 7.3, 7.4, 7.5, 8.1
        """
        # Initialize processing status indicator for rendering (Requirements 7.1, 8.1)
        status_indicator = ProcessingStatusIndicator()
        
        # Use the placeholder container for efficient updates (Requirement 7.2)
        with metrics_placeholder.container():
            # Create 8-column layout for metrics and indicators (Requirements 7.1, 8.1)
            # Order: FPS | Latency | Objects | Wagons | Damage | Illumination | Deblur | OCR
            col1, col2, col3, col4, col5, col6, col7, col8 = st.columns([1, 1, 1, 1, 1, 1, 1, 1])
            
            # Column 1: FPS metric (Requirement 3.2)
            with col1:
                st.metric(
                    label="FPS",
                    value=f"{fps:.1f}",
                    help="Frames processed per second"
                )
            
            # Column 2: Processing Latency metric with warning indicator (Requirements 4.1, 4.5)
            with col2:
                # Use red text when latency exceeds warning threshold
                if latency_warning:
                    st.markdown(
                        f"""
                        <div style="padding: 0;">
                            <p style="font-size: 0.875rem; color: #808495; margin-bottom: 0.25rem;">Latency (ms)</p>
                            <p style="font-size: 2rem; font-weight: 600; color: #ff4b4b; margin: 0;">{latency_ms:.1f}</p>
                            <p style="font-size: 0.75rem; color: #ff4b4b; margin-top: 0.25rem;">⚠️ HIGH</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                        help=f"Processing latency in milliseconds. Warning threshold: {LATENCY_WARNING_THRESHOLD_MS}ms"
                    )
                else:
                    st.metric(
                        label="Latency (ms)",
                        value=f"{latency_ms:.1f}",
                        help=f"Processing latency in milliseconds. Warning threshold: {LATENCY_WARNING_THRESHOLD_MS}ms"
                    )
            
            # Column 3: Object count metric (current frame)
            with col3:
                st.metric(
                    label="Objects",
                    value=object_count,
                    help="Number of wagons detected in current frame"
                )
            
            # Column 4: Total Wagon Count metric (Requirements 3.1, 3.2, 3.3)
            with col4:
                st.metric(
                    label="Total Wagons",
                    value=total_wagon_count,
                    help="Total unique wagons that have crossed the counting line"
                )
            
            # Column 5: Damage count with status indicator (Requirements 6.1, 6.2, 6.3)
            with col5:
                # Determine delta color based on damage_detected
                # When damage is detected, show red indicator
                # When no damage, show green indicator
                if damage_detected:
                    delta_color = "inverse"  # Red for damage
                else:
                    delta_color = "normal"  # Green for normal
                
                st.metric(
                    label="Damage",
                    value=damage_count,
                    delta="ALERT" if damage_detected else "OK",
                    delta_color=delta_color,
                    help="Number of damage detections"
                )
            
            # Column 6: Illumination indicator (Requirements 7.1, 7.2)
            with col6:
                illum_tooltip = "Illumination enhancement status"
                if st.session_state.get("illumination_auto_mode", True):
                    illum_tooltip += " (Auto mode)"
                else:
                    illum_tooltip += " (Manual mode)"
                st.markdown(
                    status_indicator.render_indicator_html(
                        label="Illum",
                        status=illumination_status,
                        tooltip=illum_tooltip
                    ),
                    unsafe_allow_html=True
                )
            
            # Column 7: Deblur indicator (Requirements 7.1, 7.3)
            with col7:
                deblur_tooltip = "Deblur processing status"
                blur_score = st.session_state.get("current_blur_score", 0.0)
                if blur_score > 0:
                    deblur_tooltip += f" (Blur score: {blur_score:.1f})"
                st.markdown(
                    status_indicator.render_indicator_html(
                        label="Deblur",
                        status=deblur_status,
                        tooltip=deblur_tooltip
                    ),
                    unsafe_allow_html=True
                )
            
            # Column 8: OCR indicator (Requirements 7.1, 8.1)
            with col8:
                ocr_tooltip = "OCR processing status"
                ocr_interval = st.session_state.get("ocr_frame_interval", 5)
                ocr_tooltip += f" (Interval: every {ocr_interval} frames)"
                st.markdown(
                    status_indicator.render_indicator_html(
                        label="OCR",
                        status=ocr_status,
                        tooltip=ocr_tooltip
                    ),
                    unsafe_allow_html=True
                )
    
    def render_video_feed(
        self,
        video_placeholder,
        frame: Optional[np.ndarray]
    ) -> None:
        """
        Render the live video feed in a centered container.
        
        Displays the video frame using st.image() within the provided
        placeholder for efficient updates without flickering. Handles
        None frames by showing a placeholder message.
        
        Args:
            video_placeholder: st.empty() placeholder for frame updates
            frame: OpenCV BGR frame to display, or None for placeholder
            
        Requirements: 4.1, 4.2, 4.3, 4.4, 4.5
        """
        # Use the placeholder container for efficient updates (Requirement 4.2, 4.4)
        with video_placeholder.container():
            # Create centered container for video display (Requirement 4.1)
            # Use columns to center the video
            col1, col2, col3 = st.columns([0.1, 0.8, 0.1])
            
            with col2:
                if frame is not None:
                    # Convert OpenCV BGR frame to RGB for Streamlit display
                    # OpenCV uses BGR format, Streamlit expects RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Display frame using st.image() (Requirements 4.3, 4.4)
                    # The frame may already have detection bounding boxes overlaid
                    st.image(
                        frame_rgb,
                        caption="Live Video Feed",
                        use_container_width=True,
                        channels="RGB"
                    )
                else:
                    # Handle None frame by showing placeholder message (Requirement 4.5)
                    st.markdown(
                        """
                        <div style="
                            background-color: #1E1E1E;
                            border: 2px solid #3D3D3D;
                            border-radius: 8px;
                            padding: 100px 20px;
                            text-align: center;
                            color: #B0B0B0;
                        ">
                            <h3>📹 No Video Feed</h3>
                            <p>Enter a video source and click Start to begin streaming</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
    
    def render_detection_log(
        self,
        log_placeholder,
        detection_log: List[DetectionLogEntry]
    ) -> None:
        """
        Render the detection log table in the bottom row.
        
        Converts the detection log list to a pandas DataFrame and displays
        it in a scrollable table using st.dataframe(). Uses an expander
        for collapsible view and accepts st.empty() placeholder for
        efficient updates without flickering.
        
        Args:
            log_placeholder: st.empty() placeholder for efficient updates
            detection_log: List of DetectionLogEntry objects to display
            
        Requirements: 5.1, 5.2, 5.3, 5.4, 5.5
        """
        # Use the placeholder container for efficient updates (Requirement 5.5)
        with log_placeholder.container():
            # Create expander for collapsible view (Requirement 5.4)
            with st.expander("📋 Detection Log", expanded=True):
                if detection_log:
                    # Convert detection_log list to pandas DataFrame (Requirement 5.2)
                    log_data = []
                    for entry in detection_log:
                        log_data.append({
                            "Timestamp": entry.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                            "Wagon ID": entry.wagon_id,
                            "Damage Type": entry.damage_type,
                            "Confidence": f"{entry.confidence:.2%}"
                        })
                    
                    df = pd.DataFrame(log_data)
                    
                    # Display columns: Timestamp, Wagon ID, Damage Type, Confidence
                    # Use st.dataframe() for scrollable table display (Requirement 5.3)
                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True,
                        height=200  # Fixed height for scrollable view
                    )
                else:
                    # Show placeholder message when no detections
                    st.markdown(
                        """
                        <div style="
                            text-align: center;
                            color: #B0B0B0;
                            padding: 20px;
                        ">
                            <p>No damage detections recorded yet.</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
    
    def run(self) -> None:
        """
        Main dashboard entry point with processing loop.
        
        Creates st.empty() placeholders for metrics, video, and log,
        then runs a processing loop when is_running=True. The loop:
        - Reads frames with frame-skipping
        - Stores raw frame before processing (Requirement 1.4)
        - Processes frames through detection pipeline
        - Applies TrackIDRenderer to processed frame (Requirement 2.5)
        - Calls FrameSaver for conditional frame saving (Requirements 5.1, 5.2, 5.3)
        - Appends OCR results to OCRLogDisplay (Requirement 6.4)
        - Updates metrics using MetricsCalculator with smoothing (Requirement 4.3)
        - Renders dual video display, metrics, and logs using placeholders
        - Checks for Stop button press to exit
        - Releases video resources on exit
        
        Requirements: 1.4, 2.5, 3.4, 3.5, 3.6, 4.2, 4.3, 4.4, 5.1, 5.2, 5.3, 5.5, 6.1, 6.2, 6.4, 7.4
        """
        st.title("RWIS - Railway Wagon Inspection")
        
        # Render sidebar and get settings
        settings = self.render_sidebar()
        
        # Update frame saver config from session state
        frame_save_config = get_frame_save_config_from_session()
        self.frame_saver.update_config(frame_save_config)
        
        # Create st.empty() placeholders for efficient updates (Requirements 3.6, 4.2, 4.4)
        # These placeholders allow updating only specific containers without
        # re-rendering the entire page (prevents sidebar/header flickering)
        metrics_placeholder = st.empty()
        video_placeholder = st.empty()
        logs_placeholder = st.empty()  # Single placeholder for both logs side-by-side
        
        # Add a stop button in the main area for stopping during the loop
        stop_button_placeholder = st.empty()
        
        # Get current state
        is_running = st.session_state.get("is_running", False)
        detection_log = st.session_state.get("detection_log", [])
        frame_index = st.session_state.get("frame_index", 0)
        
        # Initialize metrics for display
        fps = 0.0
        inference_ms = 0.0
        object_count = 0
        damage_count = len(detection_log)  # Total damage count from log
        damage_detected = False
        
        if is_running and self.video_manager.is_connected():
            # Show stop button during processing
            if stop_button_placeholder.button("⏹️ Stop Processing", key="stop_main"):
                st.session_state.is_running = False
                self.video_manager.release()
                st.session_state.connection_status = "Disconnected"
                st.rerun()
            
            # Main processing loop using while loop with placeholder updates
            # This avoids full page reruns and prevents flickering
            while st.session_state.get("is_running", False) and self.video_manager.is_connected():
                # Start frame timing for latency measurement (Requirement 4.3)
                frame_start_time = time.perf_counter()
                self.metrics_calculator.start_frame()
                
                # Read frame with frame-skipping (handled by VideoManager)
                success, frame = self.video_manager.read_frame()
                
                if success and frame is not None:
                    # Increment frame index
                    frame_index += 1
                    st.session_state.frame_index = frame_index
                    
                    # Store raw frame before processing (Requirement 1.4)
                    raw_frame = frame.copy()
                    st.session_state.raw_frame = raw_frame
                    
                    # Collect calibration samples if calibration is not complete (Requirements 10.4, 10.5)
                    if hasattr(self, 'calibration_manager') and not st.session_state.get("calibration_complete", False):
                        calibration_complete = self.calibration_manager.add_sample(frame)
                        st.session_state.calibration_progress = self.calibration_manager.get_calibration_progress()
                        st.session_state.calibration_sample_count = self.calibration_manager.get_sample_count()
                        
                        if calibration_complete:
                            # Compute calibration results
                            calibration_result = self.calibration_manager.compute_calibration()
                            if calibration_result is not None:
                                st.session_state.calibration_complete = True
                                st.session_state.calibrated_blur_threshold = calibration_result.blur_threshold
                                st.session_state.calibrated_low_light_threshold = calibration_result.low_light_threshold
                                st.session_state.calibrated_gamma_value = calibration_result.gamma_value
                                
                                # Apply calibrated values if in auto mode
                                if st.session_state.get("blur_auto_mode", True):
                                    st.session_state.blur_threshold = calibration_result.blur_threshold
                                    
                                    # Update blur detector with calibrated values (Requirement 3.2)
                                    if hasattr(self, 'blur_detector'):
                                        self.blur_detector.update_from_calibration(calibration_result)
                                
                                if st.session_state.get("illumination_auto_mode", True):
                                    st.session_state.illumination_low_light_threshold = calibration_result.low_light_threshold
                                    st.session_state.illumination_gamma = calibration_result.gamma_value
                                    
                                    # Update illumination controller with calibrated values (Requirement 4.2)
                                    if hasattr(self, 'illumination_controller'):
                                        self.illumination_controller.update_from_calibration(calibration_result)
                    
                    # Update illumination status based on current frame (Requirement 5.5)
                    if hasattr(self, 'illumination_controller'):
                        is_low_light = self.illumination_controller.is_low_light(frame)
                        st.session_state.illumination_status = "Low-Light" if is_low_light else "Normal"
                        # Set illumination_applied_this_frame flag (Requirements 7.2, 7.4)
                        # Illumination is applied when auto mode is on and frame is low-light
                        illumination_auto_mode = st.session_state.get("illumination_auto_mode", True)
                        st.session_state.illumination_applied_this_frame = illumination_auto_mode and is_low_light
                    
                    # Compute and update current blur score (Requirement 6.5)
                    if hasattr(self, 'blur_detector'):
                        current_blur_score = self.blur_detector.compute_blur_score(frame)
                        st.session_state.current_blur_score = current_blur_score
                        
                        # Update deblur status based on blur score and settings (Requirements 2.2, 2.3, 2.4, 7.4, 7.5, 7.6)
                        deblur_enabled = st.session_state.get("deblur_enabled", True)
                        if not deblur_enabled:
                            st.session_state.deblur_status = "DISABLED"
                            st.session_state.deblur_status_type = DeblurStatusType.DISABLED
                        elif self.blur_detector.needs_deblur(current_blur_score):
                            st.session_state.deblur_status = "ACTIVE"
                            st.session_state.deblur_status_type = DeblurStatusType.ACTIVE
                        else:
                            st.session_state.deblur_status = "SKIPPED"
                            st.session_state.deblur_status_type = DeblurStatusType.SKIPPED
                    
                    # Process frame through detection pipeline
                    annotated_frame, object_count, frame_damage_count, damage_detected, damage_detections, tracked_wagons = \
                        self._process_frame_through_pipeline(frame, frame_index)
                    
                    # Store tracked wagons in session state (Requirement 2.1)
                    st.session_state.tracked_wagons = tracked_wagons
                    
                    # Apply TrackIDRenderer to processed frame (Requirement 2.5)
                    if tracked_wagons:
                        annotated_frame = self.track_renderer.draw_track_ids(annotated_frame, tracked_wagons)
                    
                    # Update total wagon count from tracker (Requirement 3.1)
                    # Get cumulative count from WagonTracker
                    if hasattr(self, 'wagon_tracker') and self.wagon_tracker is not None:
                        total_wagon_count = self.wagon_tracker.get_wagon_count()
                        st.session_state.total_wagon_count = total_wagon_count
                    else:
                        # Fallback: track maximum wagons in a single frame
                        total_wagon_count = st.session_state.get("total_wagon_count", 0)
                        if object_count > total_wagon_count:
                            total_wagon_count = object_count
                            st.session_state.total_wagon_count = total_wagon_count
                    
                    # Append detection log entries for new damage detections
                    if damage_detections:
                        self._append_detection_log_entries(damage_detections, frame_index)
                        detection_log = st.session_state.get("detection_log", [])
                    
                    # Check if frame should be saved (Requirements 5.1, 5.2, 5.3)
                    # Get processing flags from session state (set during frame processing)
                    deblur_applied = st.session_state.get("deblur_applied_this_frame", False)
                    illumination_applied = st.session_state.get("illumination_applied_this_frame", False)
                    ocr_performed = st.session_state.get("ocr_applied_this_frame", False)
                    
                    if self.frame_saver.should_save(deblur_applied, illumination_applied, ocr_performed):
                        # Create metadata for saved frame
                        processing_applied = []
                        if deblur_applied:
                            processing_applied.append("deblur")
                        if illumination_applied:
                            processing_applied.append("illumination")
                        if ocr_performed:
                            processing_applied.append("ocr")
                        
                        metadata = FrameMetadata(
                            timestamp=datetime.now(),
                            frame_index=frame_index,
                            processing_applied=processing_applied,
                            wagon_id=tracked_wagons[0].track_id if tracked_wagons else None
                        )
                        self.frame_saver.save_frame(annotated_frame, metadata)
                    
                    # End frame timing and update metrics
                    self.metrics_calculator.end_frame()
                    
                    # Calculate and record processing latency (Requirement 4.3)
                    frame_end_time = time.perf_counter()
                    latency_ms = (frame_end_time - frame_start_time) * 1000
                    self.metrics_calculator.record_latency(latency_ms)
                    
                    # Get smoothed metrics
                    fps = self.metrics_calculator.get_fps()
                    inference_ms = self.metrics_calculator.get_inference_ms()
                    
                    # Update total damage count from log
                    damage_count = len(detection_log)
                    
                    # Store last frame in session state
                    st.session_state.last_frame = annotated_frame
                    
                    # Get smoothed latency and check warning (Requirement 4.4)
                    smoothed_latency_ms = self.metrics_calculator.get_smoothed_latency()
                    latency_warning = self.metrics_calculator.is_latency_warning(smoothed_latency_ms)
                    
                    # Get processing status for indicators (Requirements 7.1, 7.2, 7.3, 7.4, 8.1)
                    illumination_enabled = st.session_state.get("illumination_auto_mode", True)
                    illumination_applied = st.session_state.get("illumination_applied_this_frame", False)
                    deblur_enabled = st.session_state.get("deblur_enabled", True)
                    deblur_applied = st.session_state.get("deblur_applied_this_frame", False)
                    ocr_applied = st.session_state.get("ocr_applied_this_frame", False)
                    
                    # Determine indicator statuses using ProcessingStatusIndicator
                    status_indicator = ProcessingStatusIndicator()
                    illumination_status = status_indicator.get_illumination_status(
                        enabled=illumination_enabled,
                        applied_this_frame=illumination_applied
                    )
                    deblur_status = status_indicator.get_deblur_status(
                        enabled=deblur_enabled,
                        applied_this_frame=deblur_applied,
                        skipped_sharp=not deblur_applied and deblur_enabled
                    )
                    ocr_status = status_indicator.get_ocr_status(
                        enabled=True,  # OCR is always enabled when pipeline is running
                        active_this_frame=ocr_applied,
                        skipped_interval=not ocr_applied
                    )
                    
                    # Update only the placeholders (no full page rerun)
                    self.render_metrics_row(
                        metrics_placeholder,
                        fps=fps,
                        inference_ms=inference_ms,
                        object_count=object_count,
                        damage_count=damage_count,
                        damage_detected=damage_detected,
                        total_wagon_count=total_wagon_count,
                        latency_ms=smoothed_latency_ms,
                        latency_warning=latency_warning,
                        illumination_status=illumination_status,
                        deblur_status=deblur_status,
                        ocr_status=ocr_status
                    )
                    
                    # Render dual video display (Requirement 1.1)
                    self.dual_display.render(
                        raw_frame=raw_frame,
                        processed_frame=annotated_frame,
                        frame_index=frame_index,
                        container=video_placeholder
                    )
                    
                    # Render both logs side-by-side in columns
                    with logs_placeholder.container():
                        col1, col2 = st.columns(2)
                        with col1:
                            self.render_detection_log(st.empty(), detection_log)
                        with col2:
                            self.ocr_log_display.render(st.empty())
                    
                    # Small delay to control frame rate and allow UI responsiveness
                    time.sleep(0.03)
                else:
                    # Frame read failed - video may have ended or disconnected
                    st.session_state.is_running = False
                    st.session_state.connection_status = "Disconnected"
                    self.video_manager.release()
                    break
            
            # After loop ends, show final state
            last_frame = st.session_state.get("last_frame", None)
            raw_frame = st.session_state.get("raw_frame", None)
            total_wagon_count = st.session_state.get("total_wagon_count", 0)
            smoothed_latency_ms = self.metrics_calculator.get_smoothed_latency()
            latency_warning = self.metrics_calculator.is_latency_warning(smoothed_latency_ms)
            
            # Get processing status for indicators (Requirements 7.1, 7.2, 7.3, 7.4, 8.1)
            illumination_enabled = st.session_state.get("illumination_auto_mode", True)
            illumination_applied = st.session_state.get("illumination_applied_this_frame", False)
            deblur_enabled = st.session_state.get("deblur_enabled", True)
            deblur_applied = st.session_state.get("deblur_applied_this_frame", False)
            ocr_applied = st.session_state.get("ocr_applied_this_frame", False)
            
            status_indicator = ProcessingStatusIndicator()
            illumination_status = status_indicator.get_illumination_status(
                enabled=illumination_enabled,
                applied_this_frame=illumination_applied
            )
            deblur_status = status_indicator.get_deblur_status(
                enabled=deblur_enabled,
                applied_this_frame=deblur_applied,
                skipped_sharp=not deblur_applied and deblur_enabled
            )
            ocr_status = status_indicator.get_ocr_status(
                enabled=True,
                active_this_frame=ocr_applied,
                skipped_interval=not ocr_applied
            )
            
            self.render_metrics_row(
                metrics_placeholder,
                fps=fps,
                inference_ms=inference_ms,
                object_count=object_count,
                damage_count=damage_count,
                damage_detected=damage_detected,
                total_wagon_count=total_wagon_count,
                latency_ms=smoothed_latency_ms,
                latency_warning=latency_warning,
                illumination_status=illumination_status,
                deblur_status=deblur_status,
                ocr_status=ocr_status
            )
            # Render dual video display with last frames
            self.dual_display.render(
                raw_frame=raw_frame,
                processed_frame=last_frame,
                frame_index=frame_index,
                container=video_placeholder
            )
            self.render_detection_log(log_placeholder, detection_log)
            self.ocr_log_display.render(ocr_log_placeholder)
        else:
            # Not running - show static UI with placeholders
            last_frame = st.session_state.get("last_frame", None)
            raw_frame = st.session_state.get("raw_frame", None)
            total_wagon_count = st.session_state.get("total_wagon_count", 0)
            smoothed_latency_ms = self.metrics_calculator.get_smoothed_latency()
            latency_warning = self.metrics_calculator.is_latency_warning(smoothed_latency_ms)
            
            # Get processing status for indicators (Requirements 7.1, 7.2, 7.3, 7.4, 8.1)
            # When not running, show OFF status for all indicators
            illumination_enabled = st.session_state.get("illumination_auto_mode", True)
            deblur_enabled = st.session_state.get("deblur_enabled", True)
            
            status_indicator = ProcessingStatusIndicator()
            illumination_status = status_indicator.get_illumination_status(
                enabled=illumination_enabled,
                applied_this_frame=False
            )
            deblur_status = status_indicator.get_deblur_status(
                enabled=deblur_enabled,
                applied_this_frame=False,
                skipped_sharp=False
            )
            ocr_status = status_indicator.get_ocr_status(
                enabled=True,
                active_this_frame=False,
                skipped_interval=False
            )
            
            # Render metrics row with default/last values
            self.render_metrics_row(
                metrics_placeholder,
                fps=fps,
                inference_ms=inference_ms,
                object_count=object_count,
                damage_count=damage_count,
                damage_detected=damage_detected,
                total_wagon_count=total_wagon_count,
                latency_ms=smoothed_latency_ms,
                latency_warning=latency_warning,
                illumination_status=illumination_status,
                deblur_status=deblur_status,
                ocr_status=ocr_status
            )
            
            # Render dual video display (shows placeholders when no frames)
            self.dual_display.render(
                raw_frame=raw_frame,
                processed_frame=last_frame,
                frame_index=frame_index,
                container=video_placeholder
            )
            
            # Render both logs side-by-side in columns
            with logs_placeholder.container():
                col1, col2 = st.columns(2)
                with col1:
                    self.render_detection_log(st.empty(), detection_log)
                with col2:
                    self.ocr_log_display.render(st.empty())
            
            # Release resources if stopped (Requirement 7.4)
            if not is_running and self.video_manager.is_connected():
                self.video_manager.release()
                st.session_state.connection_status = "Disconnected"


def main():
    """
    Entry point for the dashboard application.
    """
    dashboard = MissionControlDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
