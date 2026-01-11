"""
Property-based tests for Track ID renderer module.

Feature: dashboard-enhancements
Property 2: Track ID Overlay Completeness
Validates: Requirements 2.1, 2.2, 2.4, 2.5
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, HealthCheck

from dashboard.track_renderer import TrackIDRenderer
from utils.data_models import TrackedWagon, BoundingBox


# Strategy for generating valid bounding boxes within frame bounds
@st.composite
def valid_bbox(draw, frame_width=640, frame_height=480):
    """Generate a valid bounding box within frame bounds."""
    x1 = draw(st.integers(min_value=0, max_value=frame_width - 50))
    y1 = draw(st.integers(min_value=0, max_value=frame_height - 50))
    width = draw(st.integers(min_value=20, max_value=min(100, frame_width - x1)))
    height = draw(st.integers(min_value=20, max_value=min(100, frame_height - y1)))
    return BoundingBox(x1=x1, y1=y1, x2=x1 + width, y2=y1 + height)


# Strategy for generating tracked wagons
@st.composite
def tracked_wagon(draw, frame_width=640, frame_height=480):
    """Generate a valid TrackedWagon."""
    bbox = draw(valid_bbox(frame_width, frame_height))
    track_id = draw(st.integers(min_value=1, max_value=1000))
    confidence = draw(st.floats(min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False))
    crossed_line = draw(st.booleans())
    count_index = draw(st.one_of(st.none(), st.integers(min_value=1, max_value=100)))
    return TrackedWagon(
        track_id=track_id,
        bbox=bbox,
        confidence=confidence,
        crossed_line=crossed_line,
        count_index=count_index
    )


# Strategy for generating lists of tracked wagons with unique IDs
@st.composite
def tracked_wagon_list(draw, min_wagons=0, max_wagons=10, frame_width=640, frame_height=480):
    """Generate a list of tracked wagons with unique track IDs."""
    num_wagons = draw(st.integers(min_value=min_wagons, max_value=max_wagons))
    wagons = []
    used_ids = set()
    
    for _ in range(num_wagons):
        bbox = draw(valid_bbox(frame_width, frame_height))
        # Generate unique track ID
        track_id = draw(st.integers(min_value=1, max_value=10000))
        while track_id in used_ids:
            track_id = draw(st.integers(min_value=1, max_value=10000))
        used_ids.add(track_id)
        
        confidence = draw(st.floats(min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False))
        crossed_line = draw(st.booleans())
        count_index = draw(st.one_of(st.none(), st.integers(min_value=1, max_value=100)))
        
        wagons.append(TrackedWagon(
            track_id=track_id,
            bbox=bbox,
            confidence=confidence,
            crossed_line=crossed_line,
            count_index=count_index
        ))
    
    return wagons


class TestTrackIDOverlayCompleteness:
    """
    Property 2: Track ID Overlay Completeness
    
    For any set of tracked wagons in a frame, the processed frame overlay
    SHALL display a Track_ID label for each wagon. The label position SHALL
    be within or adjacent to the wagon's bounding box coordinates.
    
    Validates: Requirements 2.1, 2.2, 2.4, 2.5
    """

    @given(wagons=tracked_wagon_list(min_wagons=1, max_wagons=10))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_all_track_ids_rendered(self, wagons):
        """
        Feature: dashboard-enhancements, Property 2: Track ID Overlay Completeness
        
        Generate random sets of tracked wagons.
        Verify all Track_IDs are rendered on frame by checking that the
        output frame differs from input for each wagon's label region.
        """
        renderer = TrackIDRenderer()
        frame_height, frame_width = 480, 640
        
        # Create a blank frame
        original_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        
        # Render track IDs
        annotated_frame = renderer.draw_track_ids(original_frame, wagons)
        
        # Verify frame was modified (track IDs were drawn)
        assert not np.array_equal(original_frame, annotated_frame), \
            "Frame should be modified when wagons are present"
        
        # For each wagon, verify that pixels near its bounding box were modified
        for wagon in wagons:
            # Check the region near the top of the bounding box where label should be
            label_region_y_start = max(0, wagon.bbox.y1 - 30)
            label_region_y_end = min(frame_height, wagon.bbox.y1 + 30)
            label_region_x_start = max(0, wagon.bbox.x1 - 10)
            label_region_x_end = min(frame_width, wagon.bbox.x1 + 150)
            
            # Extract regions from both frames
            original_region = original_frame[
                label_region_y_start:label_region_y_end,
                label_region_x_start:label_region_x_end
            ]
            annotated_region = annotated_frame[
                label_region_y_start:label_region_y_end,
                label_region_x_start:label_region_x_end
            ]
            
            # Verify the region was modified (label was drawn)
            assert not np.array_equal(original_region, annotated_region), \
                f"Track ID label for wagon {wagon.track_id} should be rendered near bbox"

    @given(wagons=tracked_wagon_list(min_wagons=0, max_wagons=0))
    @settings(max_examples=100)
    def test_empty_wagon_list_unchanged(self, wagons):
        """
        Feature: dashboard-enhancements, Property 2: Track ID Overlay Completeness
        
        Verify that when no wagons are provided, the frame remains unchanged.
        """
        renderer = TrackIDRenderer()
        frame_height, frame_width = 480, 640
        
        # Create a frame with some content
        original_frame = np.random.randint(0, 255, (frame_height, frame_width, 3), dtype=np.uint8)
        original_copy = original_frame.copy()
        
        # Render with empty wagon list
        annotated_frame = renderer.draw_track_ids(original_frame, wagons)
        
        # Frame should be unchanged (but it's a copy, so original is preserved)
        assert np.array_equal(original_copy, original_frame), \
            "Original frame should not be modified"

    @given(wagon=tracked_wagon())
    @settings(max_examples=100)
    def test_label_position_near_bbox(self, wagon):
        """
        Feature: dashboard-enhancements, Property 2: Track ID Overlay Completeness
        
        Verify that the label position is within or adjacent to the wagon's
        bounding box coordinates, accounting for frame boundary adjustments.
        """
        renderer = TrackIDRenderer()
        frame_height, frame_width = 480, 640
        frame_shape = (frame_height, frame_width, 3)
        
        # Calculate label position
        label_pos = renderer._calculate_label_position(wagon.bbox, frame_shape)
        
        # Label x should be within frame bounds (implementation adjusts for edge cases)
        assert label_pos[0] >= 0, \
            f"Label x ({label_pos[0]}) should be >= 0"
        assert label_pos[0] < frame_width, \
            f"Label x ({label_pos[0]}) should be within frame width"
        
        # When bbox is not near frame edge, label should be near bbox x1
        # When bbox is near right edge, label may be adjusted left to fit
        if wagon.bbox.x1 < frame_width - 100:
            assert label_pos[0] >= wagon.bbox.x1 - 20, \
                f"Label x ({label_pos[0]}) should be near bbox x1 ({wagon.bbox.x1})"
        
        assert label_pos[0] <= wagon.bbox.x2 + 50, \
            f"Label x ({label_pos[0]}) should not be too far from bbox"
        
        # Label y should be near bbox top (y1) - either above or just inside
        assert label_pos[1] >= wagon.bbox.y1 - 50, \
            f"Label y ({label_pos[1]}) should be near bbox y1 ({wagon.bbox.y1})"
        assert label_pos[1] <= wagon.bbox.y2 + 30, \
            f"Label y ({label_pos[1]}) should not be too far below bbox"

    @given(wagon=tracked_wagon())
    @settings(max_examples=100)
    def test_label_within_frame_bounds(self, wagon):
        """
        Feature: dashboard-enhancements, Property 2: Track ID Overlay Completeness
        
        Verify that the label position stays within frame bounds.
        """
        renderer = TrackIDRenderer()
        frame_height, frame_width = 480, 640
        frame_shape = (frame_height, frame_width, 3)
        
        # Calculate label position
        label_pos = renderer._calculate_label_position(wagon.bbox, frame_shape)
        
        # Label position should be within frame bounds
        assert 0 <= label_pos[0] < frame_width, \
            f"Label x ({label_pos[0]}) should be within frame width ({frame_width})"
        assert 0 <= label_pos[1] < frame_height, \
            f"Label y ({label_pos[1]}) should be within frame height ({frame_height})"

    @given(wagons=tracked_wagon_list(min_wagons=2, max_wagons=5))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.large_base_example, HealthCheck.too_slow])
    def test_multiple_wagons_all_labeled(self, wagons):
        """
        Feature: dashboard-enhancements, Property 2: Track ID Overlay Completeness
        
        Verify that when multiple wagons are visible, Track_IDs are displayed
        for all detected wagons.
        """
        renderer = TrackIDRenderer()
        frame_height, frame_width = 480, 640
        
        # Create a blank frame
        frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        
        # Render track IDs
        annotated_frame = renderer.draw_track_ids(frame, wagons)
        
        # Count non-zero pixels (where labels were drawn)
        non_zero_pixels = np.count_nonzero(annotated_frame)
        
        # Each label should add some non-zero pixels
        # Minimum expected: at least some pixels per wagon for the label
        min_expected_pixels = len(wagons) * 50  # Conservative estimate
        
        assert non_zero_pixels >= min_expected_pixels, \
            f"Expected at least {min_expected_pixels} non-zero pixels for {len(wagons)} wagons, got {non_zero_pixels}"
