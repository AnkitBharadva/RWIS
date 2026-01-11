"""
Processing status indicators for the Mission Control Dashboard.

Provides visual indicators for processing pipeline status including
illumination enhancement, deblurring, and OCR processing states.

Requirements: 7.2, 7.3, 7.5, 8.1, 8.2
"""

from typing import Optional, Dict
from enum import Enum


class ProcessingStatus(Enum):
    """Enumeration of possible processing status values."""
    APPLIED = "APPLIED"
    ACTIVE = "ACTIVE"
    NORMAL = "NORMAL"
    SKIPPED = "SKIPPED"
    OFF = "OFF"
    ERROR = "ERROR"


class ProcessingStatusIndicator:
    """
    Renders processing status indicators with color coding.
    
    Provides visual feedback for processing pipeline components including
    illumination enhancement, deblurring, and OCR processing.
    
    Attributes:
        STATUS_COLORS: Mapping of status values to hex color codes
        
    Requirements: 7.2, 7.3, 7.5
    """
    
    # Status colors mapping (Requirements 7.2, 7.3, 7.5)
    STATUS_COLORS: Dict[str, str] = {
        "APPLIED": "#28a745",   # Green - processing was applied
        "ACTIVE": "#28a745",    # Green - processing is active
        "NORMAL": "#6c757d",    # Gray - no processing needed
        "SKIPPED": "#ffc107",   # Yellow - processing was skipped
        "OFF": "#dc3545",       # Red - processing is disabled
        "ERROR": "#dc3545"      # Red - processing error occurred
    }
    
    # Text colors for contrast on colored backgrounds
    TEXT_COLORS: Dict[str, str] = {
        "APPLIED": "#ffffff",   # White text on green
        "ACTIVE": "#ffffff",    # White text on green
        "NORMAL": "#ffffff",    # White text on gray
        "SKIPPED": "#000000",   # Black text on yellow
        "OFF": "#ffffff",       # White text on red
        "ERROR": "#ffffff"      # White text on red
    }
    
    def get_status_color(self, status: str) -> str:
        """
        Get the background color for a given status.
        
        Args:
            status: Status string (APPLIED, ACTIVE, NORMAL, SKIPPED, OFF, ERROR)
            
        Returns:
            Hex color code for the status background
            
        Requirements: 7.2, 7.3, 7.5
        """
        return self.STATUS_COLORS.get(status.upper(), self.STATUS_COLORS["NORMAL"])
    
    def get_text_color(self, status: str) -> str:
        """
        Get the text color for a given status.
        
        Args:
            status: Status string (APPLIED, ACTIVE, NORMAL, SKIPPED, OFF, ERROR)
            
        Returns:
            Hex color code for the text
        """
        return self.TEXT_COLORS.get(status.upper(), "#ffffff")
    
    def render_indicator_html(
        self,
        label: str,
        status: str,
        tooltip: Optional[str] = None
    ) -> str:
        """
        Generate HTML for a single status indicator with color coding.
        
        Creates a styled HTML element displaying the processing status
        with appropriate background color based on the status value.
        
        Args:
            label: Label text for the indicator (e.g., "Illumination", "Deblur")
            status: Status string (APPLIED, ACTIVE, NORMAL, SKIPPED, OFF, ERROR)
            tooltip: Optional tooltip text for additional details
            
        Returns:
            HTML string for the styled indicator
            
        Requirements: 7.2, 7.3, 7.5
        """
        bg_color = self.get_status_color(status)
        text_color = self.get_text_color(status)
        
        # Build tooltip attribute if provided
        tooltip_attr = f'title="{tooltip}"' if tooltip else ''
        
        html = f'''
        <div style="
            background-color: {bg_color};
            color: {text_color};
            padding: 8px 12px;
            border-radius: 6px;
            text-align: center;
            font-size: 0.85rem;
            font-weight: 600;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            cursor: {'help' if tooltip else 'default'};
        " {tooltip_attr}>
            <div style="font-size: 0.7rem; opacity: 0.9; margin-bottom: 2px; text-transform: uppercase; letter-spacing: 0.5px;">
                {label}
            </div>
            <div style="font-size: 0.9rem;">
                {status.upper()}
            </div>
        </div>
        '''
        return html
    
    def get_illumination_status(
        self,
        enabled: bool,
        applied_this_frame: bool,
        error: bool = False
    ) -> str:
        """
        Determine illumination indicator status based on current state.
        
        Args:
            enabled: Whether illumination enhancement is enabled
            applied_this_frame: Whether CLAHE/gamma was applied to current frame
            error: Whether an error occurred during processing
            
        Returns:
            Status string: APPLIED, NORMAL, OFF, or ERROR
            
        Requirements: 7.2
        """
        if error:
            return "ERROR"
        if not enabled:
            return "OFF"
        if applied_this_frame:
            return "APPLIED"
        return "NORMAL"
    
    def get_deblur_status(
        self,
        enabled: bool,
        applied_this_frame: bool,
        skipped_sharp: bool = False,
        error: bool = False
    ) -> str:
        """
        Determine deblur indicator status based on current state.
        
        Args:
            enabled: Whether deblurring is enabled
            applied_this_frame: Whether MPRNet deblurring was applied to current frame
            skipped_sharp: Whether frame was skipped because it was sharp enough
            error: Whether an error occurred during processing
            
        Returns:
            Status string: APPLIED, SKIPPED, OFF, or ERROR
            
        Requirements: 7.3
        """
        if error:
            return "ERROR"
        if not enabled:
            return "OFF"
        if applied_this_frame:
            return "APPLIED"
        if skipped_sharp:
            return "SKIPPED"
        return "NORMAL"
    
    def get_ocr_status(
        self,
        enabled: bool,
        active_this_frame: bool,
        skipped_interval: bool = False,
        error: bool = False
    ) -> str:
        """
        Determine OCR indicator status based on current state.
        
        Args:
            enabled: Whether OCR is enabled
            active_this_frame: Whether OCR was executed on current frame
            skipped_interval: Whether OCR was skipped due to frame interval
            error: Whether an error occurred during processing
            
        Returns:
            Status string: ACTIVE, SKIPPED, OFF, or ERROR
            
        Requirements: 8.4
        """
        if error:
            return "ERROR"
        if not enabled:
            return "OFF"
        if active_this_frame:
            return "ACTIVE"
        if skipped_interval:
            return "SKIPPED"
        return "NORMAL"


    def render_metrics_row_html(
        self,
        fps: float,
        latency_ms: float,
        object_count: int,
        wagon_count: int,
        damage_count: int,
        illumination_status: str,
        deblur_status: str,
        ocr_status: str,
        tooltips: Optional[Dict[str, str]] = None,
        latency_warning: bool = False,
        damage_detected: bool = False
    ) -> str:
        """
        Generate HTML for the complete enhanced metrics row.
        
        Creates a responsive row layout with all metrics and processing
        status indicators in the specified order.
        
        Order: FPS | Latency | Objects | Wagons | Damage | Illumination | Deblur | OCR
        
        Args:
            fps: Current frames per second
            latency_ms: Processing latency in milliseconds
            object_count: Number of detected objects in current frame
            wagon_count: Total unique wagons counted
            damage_count: Number of damage detections
            illumination_status: Status for illumination indicator
            deblur_status: Status for deblur indicator
            ocr_status: Status for OCR indicator
            tooltips: Optional dict with tooltip text for each indicator
            latency_warning: Whether latency exceeds warning threshold
            damage_detected: Whether damage is currently detected
            
        Returns:
            HTML string for the complete metrics row
            
        Requirements: 8.1, 8.2
        """
        tooltips = tooltips or {}
        
        # Determine colors for latency and damage
        latency_color = "#ff4b4b" if latency_warning else "#ffffff"
        damage_color = "#ff4b4b" if damage_detected else "#28a745"
        damage_status_text = "ALERT" if damage_detected else "OK"
        
        # Build the metrics row HTML
        html = f'''
        <div style="
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            justify-content: space-between;
            align-items: stretch;
            margin-bottom: 15px;
        ">
            <!-- FPS Metric -->
            <div style="
                flex: 1;
                min-width: 80px;
                background-color: #1E1E1E;
                border: 1px solid #3D3D3D;
                border-radius: 8px;
                padding: 10px;
                text-align: center;
            ">
                <div style="font-size: 0.7rem; color: #B0B0B0; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;">FPS</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #ffffff;">{fps:.1f}</div>
            </div>
            
            <!-- Latency Metric -->
            <div style="
                flex: 1;
                min-width: 80px;
                background-color: #1E1E1E;
                border: 1px solid #3D3D3D;
                border-radius: 8px;
                padding: 10px;
                text-align: center;
            ">
                <div style="font-size: 0.7rem; color: #B0B0B0; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;">Latency</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: {latency_color};">{latency_ms:.1f}ms</div>
                {'<div style="font-size: 0.65rem; color: #ff4b4b;">⚠️ HIGH</div>' if latency_warning else ''}
            </div>
            
            <!-- Objects Metric -->
            <div style="
                flex: 1;
                min-width: 80px;
                background-color: #1E1E1E;
                border: 1px solid #3D3D3D;
                border-radius: 8px;
                padding: 10px;
                text-align: center;
            ">
                <div style="font-size: 0.7rem; color: #B0B0B0; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;">Objects</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #ffffff;">{object_count}</div>
            </div>
            
            <!-- Wagons Metric -->
            <div style="
                flex: 1;
                min-width: 80px;
                background-color: #1E1E1E;
                border: 1px solid #3D3D3D;
                border-radius: 8px;
                padding: 10px;
                text-align: center;
            ">
                <div style="font-size: 0.7rem; color: #B0B0B0; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;">Wagons</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #ffffff;">{wagon_count}</div>
            </div>
            
            <!-- Damage Metric -->
            <div style="
                flex: 1;
                min-width: 80px;
                background-color: #1E1E1E;
                border: 1px solid #3D3D3D;
                border-radius: 8px;
                padding: 10px;
                text-align: center;
            ">
                <div style="font-size: 0.7rem; color: #B0B0B0; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;">Damage</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: {damage_color};">{damage_count}</div>
                <div style="font-size: 0.65rem; color: {damage_color};">{damage_status_text}</div>
            </div>
            
            <!-- Illumination Indicator -->
            {self.render_indicator_html("Illum", illumination_status, tooltips.get("illumination"))}
            
            <!-- Deblur Indicator -->
            {self.render_indicator_html("Deblur", deblur_status, tooltips.get("deblur"))}
            
            <!-- OCR Indicator -->
            {self.render_indicator_html("OCR", ocr_status, tooltips.get("ocr"))}
        </div>
        '''
        return html
