"""
OCR Log Display component for the Mission Control Dashboard.

This module provides the OCRLogDisplay class for managing and displaying
OCR text extraction logs in the Streamlit dashboard.

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6
"""

import streamlit as st
import pandas as pd
from typing import List, Optional
from datetime import datetime

from dashboard.models import OCRLogEntry


class OCRLogDisplay:
    """
    Manages OCR log display and storage.
    
    This class handles the storage, management, and rendering of OCR
    text extraction log entries. It maintains chronological order,
    enforces a maximum entry limit, and provides a scrollable
    dataframe display within a collapsible expander.
    
    Attributes:
        MAX_LOG_ENTRIES: Maximum number of log entries to retain (500)
        entries: List of OCRLogEntry objects in chronological order
    
    Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6
    """
    
    MAX_LOG_ENTRIES: int = 500
    
    def __init__(self):
        """
        Initialize OCR log storage with empty entries list.
        
        The entries list maintains chronological order with newest
        entries at the end.
        """
        self.entries: List[OCRLogEntry] = []
    
    def append_entry(self, entry: OCRLogEntry) -> None:
        """
        Append new OCR entry maintaining chronological order.
        
        Adds the entry to the end of the list (assuming entries are
        added in chronological order) and trims old entries if the
        maximum limit is exceeded.
        
        Args:
            entry: OCRLogEntry to append to the log
            
        Requirements: 6.4, 6.6
        """
        self.entries.append(entry)
        self.trim_old_entries()
    
    def trim_old_entries(self) -> None:
        """
        Enforce maximum log entry limit by removing oldest entries.
        
        If the number of entries exceeds MAX_LOG_ENTRIES, removes
        the oldest entries (from the beginning of the list) to
        maintain the limit.
        
        Requirements: 6.3
        """
        if len(self.entries) > self.MAX_LOG_ENTRIES:
            # Remove oldest entries (from the beginning)
            excess = len(self.entries) - self.MAX_LOG_ENTRIES
            self.entries = self.entries[excess:]
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert log entries to pandas DataFrame.
        
        Creates a DataFrame with columns for Timestamp, Wagon ID,
        Extracted Text, Confidence, and Frame Index. The DataFrame
        is suitable for display in st.dataframe().
        
        Returns:
            pandas DataFrame with OCR log entries, or empty DataFrame
            if no entries exist
            
        Requirements: 6.2
        """
        if not self.entries:
            return pd.DataFrame(columns=[
                "Timestamp", "Wagon ID", "Extracted Text", "Confidence", "Frame"
            ])
        
        log_data = []
        for entry in self.entries:
            log_data.append({
                "Timestamp": entry.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "Wagon ID": entry.wagon_id,
                "Extracted Text": entry.extracted_text,
                "Confidence": f"{entry.confidence:.2%}",
                "Frame": entry.frame_index
            })
        
        return pd.DataFrame(log_data)
    
    def render(self, placeholder: Optional[st.delta_generator.DeltaGenerator] = None) -> None:
        """
        Render OCR log in collapsible expander with scrollable dataframe.
        
        Displays the OCR log entries in a Streamlit expander widget.
        Uses st.dataframe() for a scrollable table view. If a placeholder
        is provided, renders within that container for efficient updates.
        
        Args:
            placeholder: Optional st.empty() placeholder for efficient updates.
                        If None, renders directly to the page.
            
        Requirements: 6.1, 6.3, 6.5
        """
        container = placeholder.container() if placeholder else st
        
        with container:
            # Create expander for collapsible view (Requirement 6.5)
            with st.expander("📝 OCR Log", expanded=True):
                if self.entries:
                    # Convert to DataFrame and display (Requirements 6.2, 6.3)
                    df = self.to_dataframe()
                    
                    # Display scrollable dataframe
                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True,
                        height=200  # Fixed height for scrollable view
                    )
                else:
                    # Show placeholder message when no OCR extractions
                    st.markdown(
                        """
                        <div style="
                            text-align: center;
                            color: #B0B0B0;
                            padding: 20px;
                        ">
                            <p>No OCR extractions recorded yet.</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
    
    def clear(self) -> None:
        """
        Clear all log entries.
        
        Removes all entries from the log. Useful for resetting
        the log when starting a new session.
        """
        self.entries = []
    
    def get_entry_count(self) -> int:
        """
        Get the current number of log entries.
        
        Returns:
            Number of entries in the log
        """
        return len(self.entries)
    
    def get_latest_entries(self, count: int = 10) -> List[OCRLogEntry]:
        """
        Get the most recent log entries.
        
        Args:
            count: Number of recent entries to return (default 10)
            
        Returns:
            List of the most recent OCRLogEntry objects
        """
        if count >= len(self.entries):
            return list(self.entries)
        return self.entries[-count:]
