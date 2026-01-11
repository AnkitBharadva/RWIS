"""
Dashboard module for the Mission Control Streamlit interface.

This module provides a professional industrial monitoring dashboard
for the High-Speed Railway Wagon Inspection System.
"""

from dashboard.styles import inject_css, CUSTOM_CSS
from dashboard.video_manager import VideoManager
from dashboard.metrics import MetricsCalculator
from dashboard.track_renderer import TrackIDRenderer
from dashboard.frame_saver import FrameSaver
from dashboard.ocr_interval_controller import OCRIntervalController

__all__ = [
    "inject_css",
    "CUSTOM_CSS",
    "VideoManager",
    "MetricsCalculator",
    "TrackIDRenderer",
    "FrameSaver",
    "OCRIntervalController",
]
