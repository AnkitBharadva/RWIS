"""
Metrics calculator for the Mission Control Dashboard.

Provides smoothed FPS and inference latency calculations
using rolling averages.
"""

import time
from typing import List

# Latency warning threshold in milliseconds (Requirement 4.5)
LATENCY_WARNING_THRESHOLD_MS: float = 100.0


class MetricsCalculator:
    """
    Calculates and smooths performance metrics for dashboard display.
    
    Uses rolling averages to provide stable metric values that
    don't fluctuate rapidly.
    """
    
    def __init__(self, history_size: int = 30, latency_smoothing_window: int = 10):
        """
        Initialize metrics tracking.
        
        Args:
            history_size: Number of samples to keep for averaging
            latency_smoothing_window: Window size for latency smoothing (Requirement 4.4)
        """
        self.history_size = history_size
        self.latency_smoothing_window = latency_smoothing_window
        self.last_time: float = 0.0
        self.fps_history: List[float] = []
        self.inference_history: List[float] = []
        self.latency_history: List[float] = []
    
    def start_frame(self) -> None:
        """
        Mark start of frame processing.
        
        Call this at the beginning of each frame processing cycle.
        """
        self.last_time = time.perf_counter()
    
    def end_frame(self) -> None:
        """
        Mark end of frame processing and calculate metrics.
        
        Call this at the end of each frame processing cycle.
        Calculates frame duration and updates FPS history.
        """
        if self.last_time > 0:
            duration = time.perf_counter() - self.last_time
            if duration > 0:
                fps = 1.0 / duration
                self._add_to_history(self.fps_history, fps)
    
    def get_fps(self) -> float:
        """
        Get smoothed FPS value.
        
        Returns:
            Rolling average FPS, or 0.0 if no data
        """
        if not self.fps_history:
            return 0.0
        return max(0.0, sum(self.fps_history) / len(self.fps_history))
    
    def get_inference_ms(self) -> float:
        """
        Get smoothed inference latency in milliseconds.
        
        Returns:
            Rolling average inference time in ms, or 0.0 if no data
        """
        if not self.inference_history:
            return 0.0
        return max(0.0, sum(self.inference_history) / len(self.inference_history))
    
    def record_inference_time(self, duration_ms: float) -> None:
        """
        Record inference duration for averaging.
        
        Args:
            duration_ms: Inference duration in milliseconds
        """
        self._add_to_history(self.inference_history, max(0.0, duration_ms))
    
    def _add_to_history(self, history: List[float], value: float, max_size: int = None) -> None:
        """
        Add value to history buffer, maintaining max size.
        
        Args:
            history: History list to update
            value: Value to add
            max_size: Maximum size for this history (defaults to self.history_size)
        """
        if max_size is None:
            max_size = self.history_size
        history.append(value)
        if len(history) > max_size:
            history.pop(0)
    
    def reset(self) -> None:
        """
        Reset all metrics history.
        """
        self.last_time = 0.0
        self.fps_history.clear()
        self.inference_history.clear()
        self.latency_history.clear()
    
    def record_latency(self, latency_ms: float) -> None:
        """
        Record processing latency for smoothing.
        
        Args:
            latency_ms: Processing latency in milliseconds (Requirement 4.2)
        """
        self._add_to_history(
            self.latency_history, 
            max(0.0, latency_ms), 
            max_size=self.latency_smoothing_window
        )
    
    def get_smoothed_latency(self) -> float:
        """
        Get smoothed latency value using moving average.
        
        Returns:
            Rolling average latency in ms, or 0.0 if no data (Requirement 4.4)
        """
        if not self.latency_history:
            return 0.0
        return max(0.0, sum(self.latency_history) / len(self.latency_history))
    
    def is_latency_warning(self, latency_ms: float = None) -> bool:
        """
        Check if latency exceeds warning threshold.
        
        Args:
            latency_ms: Optional latency value to check. If None, uses smoothed latency.
            
        Returns:
            True if latency exceeds LATENCY_WARNING_THRESHOLD_MS (Requirement 4.5)
        """
        if latency_ms is None:
            latency_ms = self.get_smoothed_latency()
        return latency_ms > LATENCY_WARNING_THRESHOLD_MS
