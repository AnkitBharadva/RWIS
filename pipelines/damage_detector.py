"""Damage detector module using YOLOv11n for railway wagon damage detection.

This module provides the DamageDetector class that uses a separate Ultralytics
YOLOv11n model to detect damage on railway wagons. It analyzes wagon ROIs and
identifies damage types including door damage, floor damage, structural cracks,
and deformation.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6
"""

from typing import List
import numpy as np

from utils.data_models import BoundingBox, DamageDetection, DamageClass


class DamageDetector:
    """Damage detector using Ultralytics YOLOv11n model.
    
    This class detects damage on railway wagons using a separate YOLO model
    trained specifically for damage detection. It processes wagon ROIs and
    identifies various types of damage.
    
    Attributes:
        model_path: Path to the YOLOv11n damage detection model weights
        confidence_threshold: Minimum confidence score for detections
        model: The loaded YOLO model instance
    
    Requirements:
        - 3.1: Use separate Ultralytics YOLOv11n model for damage detection
        - 3.2: Analyze wagon ROI for damage types
        - 3.3: Detect door damage, floor damage, structural cracks, deformation
        - 3.4: Output damage class, bounding box, and confidence score
        - 3.5: Associate damage with corresponding wagon ID
        - 3.6: Receive only RAW or CLAHE-enhanced ROI as input
    """
    
    # Mapping from YOLO class IDs to DamageClass enum
    # Actual model classes from wagon_detector.pt (which is the damage model):
    # {0: 'Bamboo Door', 1: 'Breakage', 2: 'Close Door', 3: 'Damage Door', 
    #  4: 'Dent', 5: 'Open Door', 6: 'Wagon'}
    # NOTE: Class 6 (Wagon) is excluded - we use a separate model for wagon detection
    DAMAGE_CLASS_MAPPING = {
        0: DamageClass.BAMBOO_DOOR,
        1: DamageClass.BREAKAGE,
        2: DamageClass.CLOSE_DOOR,
        3: DamageClass.DAMAGE_DOOR,
        4: DamageClass.DENT,
        5: DamageClass.OPEN_DOOR,
        # 6: Wagon - SKIPPED (using separate wagon detector model)
    }
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        """Initialize the damage detector with a YOLO model.
        
        Args:
            model_path: Path to the YOLOv11n damage detection model weights file
            confidence_threshold: Minimum confidence threshold for detections [0.0, 1.0]
            
        Raises:
            FileNotFoundError: If the model file does not exist
            ValueError: If confidence_threshold is not in [0.0, 1.0]
        """
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be between 0.0 and 1.0, got {confidence_threshold}"
            )
        
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        
        # Load the YOLO model
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the YOLOv11n damage detection model from the specified path.
        
        Raises:
            FileNotFoundError: If the model file does not exist
            RuntimeError: If the model fails to load
        """
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Damage detection model not found at: {self.model_path}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load damage detection model: {e}")
    
    def _validate_roi(self, wagon_roi: np.ndarray) -> None:
        """Validate that the input ROI is valid for detection.
        
        Args:
            wagon_roi: The input ROI to validate
            
        Raises:
            ValueError: If the ROI is invalid (None, empty, or wrong type)
        """
        if wagon_roi is None:
            raise ValueError("Wagon ROI cannot be None")
        
        if not isinstance(wagon_roi, np.ndarray):
            raise ValueError(f"Wagon ROI must be a numpy array, got {type(wagon_roi)}")
        
        if wagon_roi.size == 0:
            raise ValueError("Wagon ROI cannot be empty")
        
        if len(wagon_roi.shape) < 2:
            raise ValueError(
                f"Wagon ROI must have at least 2 dimensions, got shape {wagon_roi.shape}"
            )
    
    def _map_class_id_to_damage_class(self, class_id: int) -> DamageClass:
        """Map YOLO class ID to DamageClass enum.
        
        Args:
            class_id: The class ID from the YOLO model
            
        Returns:
            Corresponding DamageClass enum value
            
        Raises:
            ValueError: If class_id is not in the expected range
        """
        if class_id not in self.DAMAGE_CLASS_MAPPING:
            raise ValueError(
                f"Invalid damage class ID: {class_id}. "
                f"Expected one of {list(self.DAMAGE_CLASS_MAPPING.keys())}"
            )
        return self.DAMAGE_CLASS_MAPPING[class_id]
    
    def detect(self, wagon_roi: np.ndarray, wagon_id: int) -> List[DamageDetection]:
        """Detect damage in the given wagon ROI.
        
        This method runs YOLO inference on the wagon ROI and returns a list of
        damage detections. The ROI must be RAW or CLAHE-enhanced only.
        
        Args:
            wagon_roi: Input wagon ROI (BGR format, numpy array). 
                      Must be RAW or CLAHE-enhanced.
            wagon_id: The tracking ID of the wagon this ROI belongs to
            
        Returns:
            List of DamageDetection objects containing damage_class, bbox,
            confidence, and wagon_id
            
        Raises:
            ValueError: If the ROI is invalid or wagon_id is invalid
        """
        # Validate inputs
        self._validate_roi(wagon_roi)
        
        if not isinstance(wagon_id, int) or wagon_id < 0:
            raise ValueError(f"wagon_id must be a non-negative integer, got {wagon_id}")
        
        # Run YOLO inference
        results = self.model(wagon_roi, conf=self.confidence_threshold, verbose=False)
        
        detections = []
        for result in results:
            if result.boxes is None:
                continue
                
            boxes = result.boxes
            for i in range(len(boxes)):
                # Get bounding box coordinates (xyxy format)
                xyxy = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                
                # Get confidence and class
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                
                # Map class ID to DamageClass enum and validate
                try:
                    damage_class = self._map_class_id_to_damage_class(cls_id)
                except ValueError as e:
                    # Skip invalid damage classes
                    print(f"Warning: {e}. Skipping detection.")
                    continue
                
                # Create detection object
                bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
                detection = DamageDetection(
                    damage_class=damage_class,
                    bbox=bbox,
                    confidence=conf,
                    wagon_id=wagon_id
                )
                detections.append(detection)
        
        return detections
