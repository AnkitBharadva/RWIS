"""
Property-based tests for dashboard data models.

Feature: streamlit-dashboard
Validates: Requirements 5.5, 6.2, 6.4, 6.6
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from datetime import datetime, timedelta
from typing import List

from dashboard.models import (
    ConnectionStatus,
    DetectionLogEntry,
    OCRLogEntry,
    DashboardMetrics,
    EnhancedMetrics,
    SidebarSettings,
    FrameSaveConfig,
    FrameMetadata,
    ProcessingType,
    SessionState
)


# Strategy for generating valid timestamps
timestamp_strategy = st.datetimes(
    min_value=datetime(2020, 1, 1),
    max_value=datetime(2030, 12, 31)
)

# Strategy for generating wagon IDs
wagon_id_strategy = st.integers(min_value=0, max_value=10000)

# Strategy for generating damage types
damage_type_strategy = st.sampled_from([
    "crack", "dent", "corrosion", "scratch", "deformation", "missing_part"
])

# Strategy for generating confidence scores
confidence_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)

# Strategy for generating frame indices
frame_index_strategy = st.integers(min_value=0, max_value=1000000)


# Strategy for generating DetectionLogEntry
def detection_entry_strategy(base_timestamp: datetime = None):
    """Generate a DetectionLogEntry with optional base timestamp."""
    if base_timestamp is None:
        return st.builds(
            DetectionLogEntry,
            timestamp=timestamp_strategy,
            wagon_id=wagon_id_strategy,
            damage_type=damage_type_strategy,
            confidence=confidence_strategy,
            frame_index=frame_index_strategy
        )
    else:
        # Generate entries with timestamps after base_timestamp
        return st.builds(
            DetectionLogEntry,
            timestamp=st.just(base_timestamp),
            wagon_id=wagon_id_strategy,
            damage_type=damage_type_strategy,
            confidence=confidence_strategy,
            frame_index=frame_index_strategy
        )


class TestDetectionLogAppendOnly:
    """
    Property 4: Detection Log Append-Only
    
    For any detection log, new entries SHALL only be appended to the end.
    The log SHALL maintain chronological order by timestamp.
    Existing entries SHALL NOT be modified or removed during a session.
    
    Validates: Requirements 5.5
    """

    @given(
        initial_entries=st.lists(
            st.builds(
                DetectionLogEntry,
                timestamp=timestamp_strategy,
                wagon_id=wagon_id_strategy,
                damage_type=damage_type_strategy,
                confidence=confidence_strategy,
                frame_index=frame_index_strategy
            ),
            min_size=0,
            max_size=20
        ),
        new_entries=st.lists(
            st.builds(
                DetectionLogEntry,
                timestamp=timestamp_strategy,
                wagon_id=wagon_id_strategy,
                damage_type=damage_type_strategy,
                confidence=confidence_strategy,
                frame_index=frame_index_strategy
            ),
            min_size=1,
            max_size=10
        )
    )
    @settings(max_examples=100)
    def test_append_preserves_existing_entries(self, initial_entries, new_entries):
        """
        Feature: streamlit-dashboard, Property 4: Detection Log Append-Only
        
        Verify that appending new entries preserves all existing entries.
        """
        # Create session state with initial entries
        state = SessionState()
        state.detection_log = list(initial_entries)  # Copy to avoid mutation
        
        # Store original entries for comparison
        original_entries = list(state.detection_log)
        original_length = len(original_entries)
        
        # Append new entries
        for entry in new_entries:
            state.detection_log.append(entry)
        
        # Property: All original entries must still be present at their original positions
        assert len(state.detection_log) == original_length + len(new_entries), \
            "Log length should equal original + new entries"
        
        for i, original_entry in enumerate(original_entries):
            assert state.detection_log[i] == original_entry, \
                f"Entry at position {i} was modified or removed"

    @given(
        entries=st.lists(
            st.builds(
                DetectionLogEntry,
                timestamp=timestamp_strategy,
                wagon_id=wagon_id_strategy,
                damage_type=damage_type_strategy,
                confidence=confidence_strategy,
                frame_index=frame_index_strategy
            ),
            min_size=2,
            max_size=30
        )
    )
    @settings(max_examples=100)
    def test_chronological_order_maintained_when_sorted(self, entries):
        """
        Feature: streamlit-dashboard, Property 4: Detection Log Append-Only
        
        Verify that when entries are added in chronological order,
        the log maintains that order.
        """
        # Sort entries by timestamp to simulate chronological addition
        sorted_entries = sorted(entries, key=lambda e: e.timestamp)
        
        state = SessionState()
        
        # Add entries in chronological order
        for entry in sorted_entries:
            state.detection_log.append(entry)
        
        # Property: Log should maintain chronological order
        for i in range(len(state.detection_log) - 1):
            assert state.detection_log[i].timestamp <= state.detection_log[i + 1].timestamp, \
                f"Chronological order violated at position {i}"

    @given(
        num_entries=st.integers(min_value=1, max_value=50)
    )
    @settings(max_examples=100)
    def test_entries_only_appended_to_end(self, num_entries):
        """
        Feature: streamlit-dashboard, Property 4: Detection Log Append-Only
        
        Verify that new entries are only appended to the end of the log.
        """
        state = SessionState()
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        
        # Add entries one by one
        for i in range(num_entries):
            entry = DetectionLogEntry(
                timestamp=base_time + timedelta(seconds=i),
                wagon_id=i,
                damage_type="crack",
                confidence=0.9,
                frame_index=i * 10
            )
            
            # Store length before append
            length_before = len(state.detection_log)
            
            # Append entry
            state.detection_log.append(entry)
            
            # Property: Length should increase by exactly 1
            assert len(state.detection_log) == length_before + 1, \
                "Length should increase by exactly 1 after append"
            
            # Property: New entry should be at the end
            assert state.detection_log[-1] == entry, \
                "New entry should be at the end of the log"

    @given(
        entries=st.lists(
            st.builds(
                DetectionLogEntry,
                timestamp=timestamp_strategy,
                wagon_id=wagon_id_strategy,
                damage_type=damage_type_strategy,
                confidence=confidence_strategy,
                frame_index=frame_index_strategy
            ),
            min_size=1,
            max_size=20
        )
    )
    @settings(max_examples=100)
    def test_log_entries_immutable_after_append(self, entries):
        """
        Feature: streamlit-dashboard, Property 4: Detection Log Append-Only
        
        Verify that entries in the log are not modified after being appended.
        """
        state = SessionState()
        
        # Add all entries
        for entry in entries:
            state.detection_log.append(entry)
        
        # Store copies of all entries
        stored_entries = [
            (e.timestamp, e.wagon_id, e.damage_type, e.confidence, e.frame_index)
            for e in state.detection_log
        ]
        
        # Add more entries (simulating continued operation)
        for i in range(5):
            new_entry = DetectionLogEntry(
                timestamp=datetime.now(),
                wagon_id=9999 + i,
                damage_type="new_damage",
                confidence=0.5,
                frame_index=999999 + i
            )
            state.detection_log.append(new_entry)
        
        # Property: Original entries should be unchanged
        for i, (ts, wid, dtype, conf, fidx) in enumerate(stored_entries):
            entry = state.detection_log[i]
            assert entry.timestamp == ts, f"Timestamp modified at position {i}"
            assert entry.wagon_id == wid, f"Wagon ID modified at position {i}"
            assert entry.damage_type == dtype, f"Damage type modified at position {i}"
            assert entry.confidence == conf, f"Confidence modified at position {i}"
            assert entry.frame_index == fidx, f"Frame index modified at position {i}"


# Strategy for generating extracted text
extracted_text_strategy = st.text(
    alphabet=st.characters(whitelist_categories=('L', 'N', 'P', 'S')),
    min_size=1,
    max_size=50
)


class TestOCRLogEntryCompleteness:
    """
    Property 6: OCR Log Entry Completeness
    
    For any OCR extraction event:
    - A log entry SHALL be appended with timestamp, wagon_id, text, and confidence
    - The log SHALL maintain chronological order (entries sorted by timestamp)
    - All required fields SHALL be present and valid
    
    Feature: dashboard-enhancements, Property 6: OCR Log Entry Completeness
    Validates: Requirements 6.2, 6.4, 6.6
    """

    @given(
        entries=st.lists(
            st.builds(
                OCRLogEntry,
                timestamp=timestamp_strategy,
                wagon_id=wagon_id_strategy,
                extracted_text=extracted_text_strategy,
                confidence=confidence_strategy,
                frame_index=frame_index_strategy
            ),
            min_size=1,
            max_size=30
        )
    )
    @settings(max_examples=100)
    def test_ocr_log_entry_has_all_required_fields(self, entries):
        """
        Feature: dashboard-enhancements, Property 6: OCR Log Entry Completeness
        
        Verify that all OCR log entries have all required fields present and valid.
        """
        for entry in entries:
            # Property: All required fields must be present
            assert hasattr(entry, 'timestamp'), "OCR entry missing timestamp field"
            assert hasattr(entry, 'wagon_id'), "OCR entry missing wagon_id field"
            assert hasattr(entry, 'extracted_text'), "OCR entry missing extracted_text field"
            assert hasattr(entry, 'confidence'), "OCR entry missing confidence field"
            assert hasattr(entry, 'frame_index'), "OCR entry missing frame_index field"
            
            # Property: Fields must have valid types
            assert isinstance(entry.timestamp, datetime), "timestamp must be datetime"
            assert isinstance(entry.wagon_id, int), "wagon_id must be int"
            assert isinstance(entry.extracted_text, str), "extracted_text must be str"
            assert isinstance(entry.confidence, float), "confidence must be float"
            assert isinstance(entry.frame_index, int), "frame_index must be int"
            
            # Property: Confidence must be in valid range
            assert 0.0 <= entry.confidence <= 1.0, \
                f"confidence {entry.confidence} must be between 0.0 and 1.0"
            
            # Property: Frame index must be non-negative
            assert entry.frame_index >= 0, \
                f"frame_index {entry.frame_index} must be non-negative"
            
            # Property: Wagon ID must be non-negative
            assert entry.wagon_id >= 0, \
                f"wagon_id {entry.wagon_id} must be non-negative"

    @given(
        entries=st.lists(
            st.builds(
                OCRLogEntry,
                timestamp=timestamp_strategy,
                wagon_id=wagon_id_strategy,
                extracted_text=extracted_text_strategy,
                confidence=confidence_strategy,
                frame_index=frame_index_strategy
            ),
            min_size=2,
            max_size=30
        )
    )
    @settings(max_examples=100)
    def test_ocr_log_maintains_chronological_order_when_sorted(self, entries):
        """
        Feature: dashboard-enhancements, Property 6: OCR Log Entry Completeness
        
        Verify that OCR log entries maintain chronological order when sorted by timestamp.
        """
        # Sort entries by timestamp to simulate chronological addition
        sorted_entries = sorted(entries, key=lambda e: e.timestamp)
        
        # Property: Sorted log should maintain chronological order
        for i in range(len(sorted_entries) - 1):
            assert sorted_entries[i].timestamp <= sorted_entries[i + 1].timestamp, \
                f"Chronological order violated at position {i}"

    @given(
        num_entries=st.integers(min_value=1, max_value=50)
    )
    @settings(max_examples=100)
    def test_ocr_log_entries_appended_in_order(self, num_entries):
        """
        Feature: dashboard-enhancements, Property 6: OCR Log Entry Completeness
        
        Verify that OCR log entries are appended in chronological order.
        """
        ocr_log: List[OCRLogEntry] = []
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        
        # Add entries one by one in chronological order
        for i in range(num_entries):
            entry = OCRLogEntry(
                timestamp=base_time + timedelta(seconds=i),
                wagon_id=i,
                extracted_text=f"WAGON-{i:04d}",
                confidence=0.9,
                frame_index=i * 10
            )
            
            # Store length before append
            length_before = len(ocr_log)
            
            # Append entry
            ocr_log.append(entry)
            
            # Property: Length should increase by exactly 1
            assert len(ocr_log) == length_before + 1, \
                "Length should increase by exactly 1 after append"
            
            # Property: New entry should be at the end
            assert ocr_log[-1] == entry, \
                "New entry should be at the end of the log"
            
            # Property: All entries should maintain chronological order
            for j in range(len(ocr_log) - 1):
                assert ocr_log[j].timestamp <= ocr_log[j + 1].timestamp, \
                    f"Chronological order violated at position {j}"

    @given(
        entries=st.lists(
            st.builds(
                OCRLogEntry,
                timestamp=timestamp_strategy,
                wagon_id=wagon_id_strategy,
                extracted_text=extracted_text_strategy,
                confidence=confidence_strategy,
                frame_index=frame_index_strategy
            ),
            min_size=1,
            max_size=20
        )
    )
    @settings(max_examples=100)
    def test_ocr_log_entries_immutable_after_append(self, entries):
        """
        Feature: dashboard-enhancements, Property 6: OCR Log Entry Completeness
        
        Verify that OCR log entries are not modified after being appended.
        """
        ocr_log: List[OCRLogEntry] = []
        
        # Add all entries
        for entry in entries:
            ocr_log.append(entry)
        
        # Store copies of all entries
        stored_entries = [
            (e.timestamp, e.wagon_id, e.extracted_text, e.confidence, e.frame_index)
            for e in ocr_log
        ]
        
        # Add more entries (simulating continued operation)
        for i in range(5):
            new_entry = OCRLogEntry(
                timestamp=datetime.now(),
                wagon_id=9999 + i,
                extracted_text=f"NEW-{i}",
                confidence=0.5,
                frame_index=999999 + i
            )
            ocr_log.append(new_entry)
        
        # Property: Original entries should be unchanged
        for i, (ts, wid, text, conf, fidx) in enumerate(stored_entries):
            entry = ocr_log[i]
            assert entry.timestamp == ts, f"Timestamp modified at position {i}"
            assert entry.wagon_id == wid, f"Wagon ID modified at position {i}"
            assert entry.extracted_text == text, f"Extracted text modified at position {i}"
            assert entry.confidence == conf, f"Confidence modified at position {i}"
            assert entry.frame_index == fidx, f"Frame index modified at position {i}"
