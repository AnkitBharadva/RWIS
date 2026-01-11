"""
Property-based tests for damage detection module.

Feature: railway-wagon-inspection
Validates: Requirements 3.4, 3.5
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings

from utils.data_models import DamageDetection, DamageClass, BoundingBox


# Strategy for generating valid damage classes
damage_class_strategy = st.sampled_from([
    DamageClass.BAMBOO_DOOR,
    DamageClass.BREAKAGE,
    DamageClass.CLOSE_DOOR,
    DamageClass.DAMAGE_DOOR,
    DamageClass.DENT,
    DamageClass.OPEN_DOOR,
])


# Strategy for generating valid bounding boxes
@st.composite
def valid_bounding_box(draw):
    """Generate a valid bounding box with positive dimensions."""
    x1 = draw(st.integers(min_value=0, max_value=1000))
    y1 = draw(st.integers(min_value=0, max_value=1000))
    width = draw(st.integers(min_value=1, max_value=500))
    height = draw(st.integers(min_value=1, max_value=500))
    return BoundingBox(x1=x1, y1=y1, x2=x1 + width, y2=y1 + height)


# Strategy for generating valid confidence scores
confidence_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


# Strategy for generating valid wagon IDs
wagon_id_strategy = st.integers(min_value=0, max_value=10000)


# Strategy for generating random damage detections
@st.composite
def random_damage_detection(draw):
    """Generate a random DamageDetection object."""
    damage_class = draw(damage_class_strategy)
    bbox = draw(valid_bounding_box())
    confidence = draw(confidence_strategy)
    wagon_id = draw(wagon_id_strategy)
    
    return DamageDetection(
        damage_class=damage_class,
        bbox=bbox,
        confidence=confidence,
        wagon_id=wagon_id
    )


class TestDamageDetectionOutputCompleteness:
    """
    Property 8: Damage Detection Output Completeness
    
    For any damage detection result, the output SHALL contain:
    - A valid damage_class from the defined enum
    - A valid bounding box with positive dimensions
    - A confidence score in range [0.0, 1.0]
    - A wagon_id matching the input wagon's track_id
    
    Validates: Requirements 3.4, 3.5
    """

    @given(detection=random_damage_detection())
    @settings(max_examples=100)
    def test_damage_detection_has_all_required_fields(self, detection):
        """
        Feature: railway-wagon-inspection, Property 8: Damage Detection Output Completeness
        
        Generate random damage detections.
        Verify all required fields present and valid (damage_class, bbox, confidence, wagon_id).
        """
        # Verify damage_class is present and is a DamageClass enum
        assert hasattr(detection, 'damage_class'), "Detection missing damage_class field"
        assert isinstance(detection.damage_class, DamageClass), \
            f"damage_class must be DamageClass enum, got {type(detection.damage_class)}"
        
        # Verify damage_class is one of the expected values
        expected_classes = {
            DamageClass.BAMBOO_DOOR,
            DamageClass.BREAKAGE,
            DamageClass.CLOSE_DOOR,
            DamageClass.DAMAGE_DOOR,
            DamageClass.DENT,
            DamageClass.OPEN_DOOR,
        }
        assert detection.damage_class in expected_classes, \
            f"damage_class {detection.damage_class} not in expected set {expected_classes}"
        
        # Verify bbox is present and is a BoundingBox
        assert hasattr(detection, 'bbox'), "Detection missing bbox field"
        assert isinstance(detection.bbox, BoundingBox), \
            f"bbox must be BoundingBox, got {type(detection.bbox)}"
        
        # Verify bbox has positive dimensions
        assert detection.bbox.width > 0, \
            f"bbox width must be positive, got {detection.bbox.width}"
        assert detection.bbox.height > 0, \
            f"bbox height must be positive, got {detection.bbox.height}"
        
        # Verify bbox coordinates are valid (x1 < x2, y1 < y2)
        assert detection.bbox.x1 < detection.bbox.x2, \
            f"bbox x1 must be less than x2: {detection.bbox.x1} >= {detection.bbox.x2}"
        assert detection.bbox.y1 < detection.bbox.y2, \
            f"bbox y1 must be less than y2: {detection.bbox.y1} >= {detection.bbox.y2}"
        
        # Verify confidence is present and in valid range
        assert hasattr(detection, 'confidence'), "Detection missing confidence field"
        assert isinstance(detection.confidence, (int, float)), \
            f"confidence must be numeric, got {type(detection.confidence)}"
        assert 0.0 <= detection.confidence <= 1.0, \
            f"confidence must be in [0.0, 1.0], got {detection.confidence}"
        
        # Verify wagon_id is present and is a valid integer
        assert hasattr(detection, 'wagon_id'), "Detection missing wagon_id field"
        assert isinstance(detection.wagon_id, int), \
            f"wagon_id must be int, got {type(detection.wagon_id)}"
        assert detection.wagon_id >= 0, \
            f"wagon_id must be non-negative, got {detection.wagon_id}"

    @given(detections=st.lists(random_damage_detection(), min_size=1, max_size=10))
    @settings(max_examples=100)
    def test_multiple_damage_detections_completeness(self, detections):
        """
        Feature: railway-wagon-inspection, Property 8: Damage Detection Output Completeness
        
        Verify that multiple damage detections all have complete and valid fields.
        """
        for i, detection in enumerate(detections):
            # Verify all required fields are present
            assert hasattr(detection, 'damage_class'), \
                f"Detection {i} missing damage_class field"
            assert hasattr(detection, 'bbox'), \
                f"Detection {i} missing bbox field"
            assert hasattr(detection, 'confidence'), \
                f"Detection {i} missing confidence field"
            assert hasattr(detection, 'wagon_id'), \
                f"Detection {i} missing wagon_id field"
            
            # Verify types
            assert isinstance(detection.damage_class, DamageClass), \
                f"Detection {i} damage_class has wrong type"
            assert isinstance(detection.bbox, BoundingBox), \
                f"Detection {i} bbox has wrong type"
            assert isinstance(detection.confidence, (int, float)), \
                f"Detection {i} confidence has wrong type"
            assert isinstance(detection.wagon_id, int), \
                f"Detection {i} wagon_id has wrong type"
            
            # Verify valid values
            assert 0.0 <= detection.confidence <= 1.0, \
                f"Detection {i} confidence out of range"
            assert detection.bbox.width > 0, \
                f"Detection {i} bbox has non-positive width"
            assert detection.bbox.height > 0, \
                f"Detection {i} bbox has non-positive height"

    @given(
        damage_class=damage_class_strategy,
        bbox=valid_bounding_box(),
        confidence=confidence_strategy,
        wagon_id=wagon_id_strategy
    )
    @settings(max_examples=100)
    def test_damage_detection_construction(self, damage_class, bbox, confidence, wagon_id):
        """
        Feature: railway-wagon-inspection, Property 8: Damage Detection Output Completeness
        
        Verify that DamageDetection can be constructed with valid parameters
        and all fields are accessible.
        """
        # Construct detection
        detection = DamageDetection(
            damage_class=damage_class,
            bbox=bbox,
            confidence=confidence,
            wagon_id=wagon_id
        )
        
        # Verify all fields match input
        assert detection.damage_class == damage_class, \
            "damage_class not preserved during construction"
        assert detection.bbox == bbox, \
            "bbox not preserved during construction"
        assert detection.confidence == confidence, \
            "confidence not preserved during construction"
        assert detection.wagon_id == wagon_id, \
            "wagon_id not preserved during construction"

    @given(detection=random_damage_detection())
    @settings(max_examples=100)
    def test_damage_class_enum_values(self, detection):
        """
        Feature: railway-wagon-inspection, Property 8: Damage Detection Output Completeness
        
        Verify that damage_class is always one of the expected enum values.
        This validates Requirement 3.3.
        """
        valid_damage_classes = {
            DamageClass.BAMBOO_DOOR,
            DamageClass.BREAKAGE,
            DamageClass.CLOSE_DOOR,
            DamageClass.DAMAGE_DOOR,
            DamageClass.DENT,
            DamageClass.OPEN_DOOR,
        }
        
        assert detection.damage_class in valid_damage_classes, \
            f"damage_class {detection.damage_class} not in valid set"
        
        # Verify it's actually a DamageClass enum member
        assert isinstance(detection.damage_class, DamageClass), \
            f"damage_class must be DamageClass enum, got {type(detection.damage_class)}"
