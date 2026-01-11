"""
Dual video display component for side-by-side raw and processed frame display.

This module provides the DualVideoDisplay class that renders raw input and
processed output video frames side by side in the dashboard.

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6
"""

from typing import Optional, Tuple
import cv2
import numpy as np
import streamlit as st


class DualVideoDisplay:
    """Manages side-by-side raw and processed video display.
    
    This class renders two video frames in a side-by-side layout:
    - Left column: Raw input frame without any processing or overlays
    - Right column: Processed output frame with detection bounding boxes and annotations
    
    Both displays are synchronized to show the same frame index.
    
    Attributes:
        raw_placeholder: Streamlit empty placeholder for raw frame
        processed_placeholder: Streamlit empty placeholder for processed frame
        current_frame_index: Current frame index for synchronization tracking
    
    Requirements:
        - 1.1: Display two video frames in side-by-side layout
        - 1.2: Left frame shows raw input without processing/overlays
        - 1.3: Right frame shows processed output with detection annotations
        - 1.4: Synchronize both displays to show same frame index
        - 1.5: Label each frame clearly as "Raw Input" and "Processed Output"
        - 1.6: Display placeholder messages when video capture is stopped
    """
    
    # Labels for the video displays
    RAW_LABEL = "Raw Input"
    PROCESSED_LABEL = "Processed Output"
    
    def __init__(self):
        """Initialize dual video display with default state."""
        self.raw_placeholder: Optional[st.delta_generator.DeltaGenerator] = None
        self.processed_placeholder: Optional[st.delta_generator.DeltaGenerator] = None
        self.current_frame_index: int = 0
    
    def render(
        self,
        raw_frame: Optional[np.ndarray],
        processed_frame: Optional[np.ndarray],
        frame_index: int,
        container: Optional[st.delta_generator.DeltaGenerator] = None
    ) -> None:
        """Render both video frames side by side.
        
        Displays raw and processed frames in a two-column layout.
        Both frames are synchronized to the same frame index.
        
        Args:
            raw_frame: Original unprocessed BGR frame, or None for placeholder
            processed_frame: Frame with bounding boxes and annotations, or None
            frame_index: Current frame index for synchronization
            container: Optional Streamlit container/placeholder to render into
            
        Requirements:
            - 1.1: Side-by-side layout
            - 1.2: Raw frame without overlays
            - 1.3: Processed frame with annotations
            - 1.4: Same frame index for both
            - 1.5: Clear labels
        """
        # Update current frame index for synchronization tracking
        self.current_frame_index = frame_index
        
        # Use provided container or create columns directly
        if container is not None:
            with container.container():
                self._render_dual_columns(raw_frame, processed_frame, frame_index)
        else:
            self._render_dual_columns(raw_frame, processed_frame, frame_index)
    
    def _render_dual_columns(
        self,
        raw_frame: Optional[np.ndarray],
        processed_frame: Optional[np.ndarray],
        frame_index: int
    ) -> None:
        """Render the dual column layout with frames.
        
        Args:
            raw_frame: Original unprocessed BGR frame
            processed_frame: Frame with annotations
            frame_index: Current frame index
        """
        # Create side-by-side columns (Requirement 1.1)
        col_raw, col_processed = st.columns(2)
        
        # Left column: Raw Input (Requirement 1.2, 1.5)
        with col_raw:
            st.markdown(f"**{self.RAW_LABEL}**")
            self._render_frame(raw_frame, is_raw=True)
        
        # Right column: Processed Output (Requirement 1.3, 1.5)
        with col_processed:
            st.markdown(f"**{self.PROCESSED_LABEL}**")
            self._render_frame(processed_frame, is_raw=False)
    
    def _render_frame(
        self,
        frame: Optional[np.ndarray],
        is_raw: bool
    ) -> None:
        """Render a single frame or placeholder.
        
        Args:
            frame: BGR frame to display, or None for placeholder
            is_raw: Whether this is the raw frame (for placeholder text)
        """
        if frame is not None and self._is_valid_frame(frame):
            # Convert BGR to RGB for Streamlit display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            st.image(
                frame_rgb,
                use_container_width=True,
                channels="RGB"
            )
        else:
            # Show placeholder (Requirement 1.6)
            label = self.RAW_LABEL if is_raw else self.PROCESSED_LABEL
            self._render_placeholder(label)
    
    def _render_placeholder(self, label: str) -> None:
        """Render placeholder message when no frame is available.
        
        Args:
            label: Label for the placeholder (Raw Input or Processed Output)
            
        Requirement 1.6: Display placeholder messages when video capture is stopped
        """
        st.markdown(
            f"""
            <div style="
                background-color: #1E1E1E;
                border: 2px solid #3D3D3D;
                border-radius: 8px;
                padding: 80px 20px;
                text-align: center;
                color: #B0B0B0;
                min-height: 200px;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
            ">
                <h4>📹 No {label}</h4>
                <p>Enter a video source and click Start to begin streaming</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    def render_placeholders(
        self,
        container: Optional[st.delta_generator.DeltaGenerator] = None
    ) -> None:
        """Render placeholder messages in both frames for stopped state.
        
        Displays placeholder messages in both the raw and processed
        frame containers when video capture is stopped.
        
        Args:
            container: Optional Streamlit container/placeholder to render into
            
        Requirement 1.6: Display placeholder messages when video capture is stopped
        """
        if container is not None:
            with container.container():
                self._render_dual_placeholders()
        else:
            self._render_dual_placeholders()
    
    def _render_dual_placeholders(self) -> None:
        """Render placeholders in both columns."""
        col_raw, col_processed = st.columns(2)
        
        with col_raw:
            st.markdown(f"**{self.RAW_LABEL}**")
            self._render_placeholder(self.RAW_LABEL)
        
        with col_processed:
            st.markdown(f"**{self.PROCESSED_LABEL}**")
            self._render_placeholder(self.PROCESSED_LABEL)
    
    def _is_valid_frame(self, frame: np.ndarray) -> bool:
        """Check if frame is valid for display.
        
        Args:
            frame: Frame to validate
            
        Returns:
            True if frame is valid, False otherwise
        """
        if frame is None:
            return False
        if not isinstance(frame, np.ndarray):
            return False
        if len(frame.shape) < 2:
            return False
        if frame.size == 0:
            return False
        return True
    
    def get_frame_index(self) -> int:
        """Get the current frame index for synchronization verification.
        
        Returns:
            Current frame index
            
        Requirement 1.4: Synchronize both displays to show same frame index
        """
        return self.current_frame_index
