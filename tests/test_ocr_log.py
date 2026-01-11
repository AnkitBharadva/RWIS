"""
Unit tests for OCRLogDisplay component.

Tests the OCRLogDisplay class functionality including:
- Entry appending and chronological order
- Maximum entry limit enforcement
- DataFrame conversion
- Entry retrieval methods

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6
"""

import pytest
from datetime import datetime, timedelta
from typing import List

from dashboard.ocr_log import OCRLogDisplay
from dashboard.models import OCRLogEntry


class TestOCRLogDisplayInit:
    """Tests for OCRLogDisplay initialization."""
    
    def test_init_creates_empty_entries_list(self):
        """OCRLogDisplay should initialize with empty entries list."""
        display = OCRLogDisplay()
        assert display.entries == []
        assert len(display.entries) == 0
    
    def test_max_log_entries_constant(self):
        """MAX_LOG_ENTRIES should be 500."""
        assert OCRLogDisplay.MAX_LOG_ENTRIES == 500


class TestOCRLogDisplayAppendEntry:
    """Tests for append_entry method."""
    
    def test_append_single_entry(self):
        """Appending a single entry should add it to the list."""
        display = OCRLogDisplay()
        entry = OCRLogEntry(
            timestamp=datetime.now(),
            wagon_id=1,
            extracted_text="WAGON-001",
            confidence=0.95,
            frame_index=100
        )
        
        display.append_entry(entry)
        
        assert len(display.entries) == 1
        assert display.entries[0] == entry
    
    def test_append_multiple_entries_maintains_order(self):
        """Appending multiple entries should maintain chronological order."""
        display = OCRLogDisplay()
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        
        entries = []
        for i in range(5):
            entry = OCRLogEntry(
                timestamp=base_time + timedelta(seconds=i),
                wagon_id=i,
                extracted_text=f"WAGON-{i:03d}",
                confidence=0.9,
                frame_index=i * 10
            )
            entries.append(entry)
            display.append_entry(entry)
        
        assert len(display.entries) == 5
        for i, entry in enumerate(entries):
            assert display.entries[i] == entry


class TestOCRLogDisplayTrimOldEntries:
    """Tests for trim_old_entries method."""
    
    def test_trim_removes_oldest_entries_when_exceeds_limit(self):
        """Trimming should remove oldest entries when limit exceeded."""
        display = OCRLogDisplay()
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        
        # Add more entries than the limit
        num_entries = OCRLogDisplay.MAX_LOG_ENTRIES + 10
        for i in range(num_entries):
            entry = OCRLogEntry(
                timestamp=base_time + timedelta(seconds=i),
                wagon_id=i,
                extracted_text=f"WAGON-{i:04d}",
                confidence=0.9,
                frame_index=i
            )
            display.append_entry(entry)
        
        # Should be trimmed to MAX_LOG_ENTRIES
        assert len(display.entries) == OCRLogDisplay.MAX_LOG_ENTRIES
        
        # Oldest entries should be removed (first 10)
        # First entry should now be the 11th original entry (index 10)
        assert display.entries[0].wagon_id == 10
    
    def test_trim_does_nothing_when_under_limit(self):
        """Trimming should not remove entries when under limit."""
        display = OCRLogDisplay()
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        
        # Add fewer entries than the limit
        num_entries = 10
        for i in range(num_entries):
            entry = OCRLogEntry(
                timestamp=base_time + timedelta(seconds=i),
                wagon_id=i,
                extracted_text=f"WAGON-{i:03d}",
                confidence=0.9,
                frame_index=i
            )
            display.append_entry(entry)
        
        assert len(display.entries) == num_entries


class TestOCRLogDisplayToDataframe:
    """Tests for to_dataframe method."""
    
    def test_to_dataframe_empty_log(self):
        """Empty log should return DataFrame with correct columns."""
        display = OCRLogDisplay()
        df = display.to_dataframe()
        
        assert len(df) == 0
        assert list(df.columns) == [
            "Timestamp", "Wagon ID", "Extracted Text", "Confidence", "Frame"
        ]
    
    def test_to_dataframe_with_entries(self):
        """DataFrame should contain all entry data correctly formatted."""
        display = OCRLogDisplay()
        timestamp = datetime(2024, 1, 15, 10, 30, 45)
        
        entry = OCRLogEntry(
            timestamp=timestamp,
            wagon_id=42,
            extracted_text="ABC-1234",
            confidence=0.875,
            frame_index=500
        )
        display.append_entry(entry)
        
        df = display.to_dataframe()
        
        assert len(df) == 1
        assert df.iloc[0]["Timestamp"] == "2024-01-15 10:30:45"
        assert df.iloc[0]["Wagon ID"] == 42
        assert df.iloc[0]["Extracted Text"] == "ABC-1234"
        assert df.iloc[0]["Confidence"] == "87.50%"
        assert df.iloc[0]["Frame"] == 500
    
    def test_to_dataframe_multiple_entries(self):
        """DataFrame should contain all entries in order."""
        display = OCRLogDisplay()
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        
        for i in range(3):
            entry = OCRLogEntry(
                timestamp=base_time + timedelta(seconds=i),
                wagon_id=i,
                extracted_text=f"TEXT-{i}",
                confidence=0.8 + i * 0.05,
                frame_index=i * 100
            )
            display.append_entry(entry)
        
        df = display.to_dataframe()
        
        assert len(df) == 3
        assert df.iloc[0]["Wagon ID"] == 0
        assert df.iloc[1]["Wagon ID"] == 1
        assert df.iloc[2]["Wagon ID"] == 2


class TestOCRLogDisplayClear:
    """Tests for clear method."""
    
    def test_clear_removes_all_entries(self):
        """Clear should remove all entries from the log."""
        display = OCRLogDisplay()
        
        # Add some entries
        for i in range(5):
            entry = OCRLogEntry(
                timestamp=datetime.now(),
                wagon_id=i,
                extracted_text=f"TEXT-{i}",
                confidence=0.9,
                frame_index=i
            )
            display.append_entry(entry)
        
        assert len(display.entries) == 5
        
        display.clear()
        
        assert len(display.entries) == 0
        assert display.entries == []


class TestOCRLogDisplayGetEntryCount:
    """Tests for get_entry_count method."""
    
    def test_get_entry_count_empty(self):
        """Empty log should return count of 0."""
        display = OCRLogDisplay()
        assert display.get_entry_count() == 0
    
    def test_get_entry_count_with_entries(self):
        """Should return correct count of entries."""
        display = OCRLogDisplay()
        
        for i in range(7):
            entry = OCRLogEntry(
                timestamp=datetime.now(),
                wagon_id=i,
                extracted_text=f"TEXT-{i}",
                confidence=0.9,
                frame_index=i
            )
            display.append_entry(entry)
        
        assert display.get_entry_count() == 7


class TestOCRLogDisplayGetLatestEntries:
    """Tests for get_latest_entries method."""
    
    def test_get_latest_entries_empty_log(self):
        """Empty log should return empty list."""
        display = OCRLogDisplay()
        result = display.get_latest_entries(5)
        assert result == []
    
    def test_get_latest_entries_fewer_than_requested(self):
        """Should return all entries when fewer than requested."""
        display = OCRLogDisplay()
        
        for i in range(3):
            entry = OCRLogEntry(
                timestamp=datetime.now(),
                wagon_id=i,
                extracted_text=f"TEXT-{i}",
                confidence=0.9,
                frame_index=i
            )
            display.append_entry(entry)
        
        result = display.get_latest_entries(10)
        assert len(result) == 3
    
    def test_get_latest_entries_returns_most_recent(self):
        """Should return the most recent entries."""
        display = OCRLogDisplay()
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        
        for i in range(10):
            entry = OCRLogEntry(
                timestamp=base_time + timedelta(seconds=i),
                wagon_id=i,
                extracted_text=f"TEXT-{i}",
                confidence=0.9,
                frame_index=i
            )
            display.append_entry(entry)
        
        result = display.get_latest_entries(3)
        
        assert len(result) == 3
        # Should be the last 3 entries (wagon_id 7, 8, 9)
        assert result[0].wagon_id == 7
        assert result[1].wagon_id == 8
        assert result[2].wagon_id == 9
    
    def test_get_latest_entries_default_count(self):
        """Default count should be 10."""
        display = OCRLogDisplay()
        
        for i in range(15):
            entry = OCRLogEntry(
                timestamp=datetime.now(),
                wagon_id=i,
                extracted_text=f"TEXT-{i}",
                confidence=0.9,
                frame_index=i
            )
            display.append_entry(entry)
        
        result = display.get_latest_entries()
        assert len(result) == 10
