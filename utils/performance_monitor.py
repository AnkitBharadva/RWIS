"""Performance monitoring utilities for pipeline optimization."""

import time
from collections import deque
from typing import Dict, Optional
import logging


logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitors pipeline performance metrics."""
    
    def __init__(self, window_size: int = 100):
        """Initialize performance monitor.
        
        Args:
            window_size: Number of frames to average over
        """
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.stage_times: Dict[str, deque] = {}
        self.current_stage_start: Optional[float] = None
        self.current_stage_name: Optional[str] = None
        self.frame_start_time: Optional[float] = None
    
    def start_frame(self):
        """Mark the start of frame processing."""
        self.frame_start_time = time.time()
    
    def end_frame(self):
        """Mark the end of frame processing."""
        if self.frame_start_time:
            elapsed = time.time() - self.frame_start_time
            self.frame_times.append(elapsed)
            self.frame_start_time = None
    
    def start_stage(self, stage_name: str):
        """Mark the start of a pipeline stage.
        
        Args:
            stage_name: Name of the stage
        """
        self.current_stage_name = stage_name
        self.current_stage_start = time.time()
    
    def end_stage(self):
        """Mark the end of a pipeline stage."""
        if self.current_stage_start and self.current_stage_name:
            elapsed = time.time() - self.current_stage_start
            
            if self.current_stage_name not in self.stage_times:
                self.stage_times[self.current_stage_name] = deque(maxlen=self.window_size)
            
            self.stage_times[self.current_stage_name].append(elapsed)
            self.current_stage_start = None
            self.current_stage_name = None
    
    def get_fps(self) -> float:
        """Get current FPS.
        
        Returns:
            Frames per second
        """
        if not self.frame_times:
            return 0.0
        avg_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0
    
    def get_avg_frame_time(self) -> float:
        """Get average frame processing time in milliseconds.
        
        Returns:
            Average time in ms
        """
        if not self.frame_times:
            return 0.0
        return (sum(self.frame_times) / len(self.frame_times)) * 1000
    
    def get_stage_time(self, stage_name: str) -> float:
        """Get average time for a specific stage in milliseconds.
        
        Args:
            stage_name: Name of the stage
            
        Returns:
            Average time in ms
        """
        if stage_name not in self.stage_times or not self.stage_times[stage_name]:
            return 0.0
        times = self.stage_times[stage_name]
        return (sum(times) / len(times)) * 1000
    
    def get_summary(self) -> Dict[str, float]:
        """Get performance summary.
        
        Returns:
            Dictionary with performance metrics
        """
        summary = {
            'fps': self.get_fps(),
            'avg_frame_time_ms': self.get_avg_frame_time(),
            'total_frames': len(self.frame_times)
        }
        
        # Add stage times
        for stage_name in self.stage_times:
            summary[f'{stage_name}_ms'] = self.get_stage_time(stage_name)
        
        return summary
    
    def print_summary(self):
        """Print performance summary to console."""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        print(f"FPS: {summary['fps']:.1f}")
        print(f"Avg Frame Time: {summary['avg_frame_time_ms']:.1f} ms")
        print(f"Total Frames: {summary['total_frames']}")
        print("\nStage Breakdown:")
        
        for key, value in summary.items():
            if key.endswith('_ms') and key != 'avg_frame_time_ms':
                stage_name = key[:-3]
                print(f"  {stage_name}: {value:.1f} ms")
        
        print("="*60 + "\n")
