"""GPU memory management utilities for the Railway Wagon Inspection Pipeline.

This module provides GPU memory monitoring, batch size adjustment,
and operation queuing for memory safety.

Requirements: 6.1, 6.5
"""

import threading
import time
from queue import Queue, Empty
from typing import Callable, Dict, Any, Optional, List, Tuple


class GPUMemoryMonitor:
    """Monitors GPU memory usage and provides alerts.
    
    Continuously monitors GPU memory in a background thread and
    triggers callbacks when memory thresholds are exceeded.
    """
    
    def __init__(
        self,
        memory_limit_bytes: int = 6 * 1024 * 1024 * 1024,
        check_interval: float = 0.5,
        warning_threshold: float = 0.7,
        critical_threshold: float = 0.9
    ):
        """Initialize GPU memory monitor.
        
        Args:
            memory_limit_bytes: Maximum GPU memory in bytes (default: 6 GB)
            check_interval: Interval between memory checks in seconds
            warning_threshold: Fraction of limit that triggers warning
            critical_threshold: Fraction of limit that triggers critical alert
        """
        self.memory_limit = memory_limit_bytes
        self.check_interval = check_interval
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        
        self._warning_bytes = int(memory_limit_bytes * warning_threshold)
        self._critical_bytes = int(memory_limit_bytes * critical_threshold)
        
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable[[str, Dict], None]] = []
        self._last_status = "normal"
    
    def add_callback(self, callback: Callable[[str, Dict], None]) -> None:
        """Add a callback for memory status changes.
        
        Args:
            callback: Function called with (status, memory_info) when status changes
                     status is one of: 'normal', 'warning', 'critical'
        """
        self._callbacks.append(callback)
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get current GPU memory information.
        
        Returns:
            Dictionary with memory statistics
        """
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                max_allocated = torch.cuda.max_memory_allocated()
                
                return {
                    'allocated': allocated,
                    'reserved': reserved,
                    'max_allocated': max_allocated,
                    'limit': self.memory_limit,
                    'usage_percent': (allocated / self.memory_limit) * 100,
                    'available': True
                }
        except ImportError:
            pass
        
        return {
            'allocated': 0,
            'reserved': 0,
            'max_allocated': 0,
            'limit': self.memory_limit,
            'usage_percent': 0,
            'available': False
        }

    def _get_status(self, allocated: int) -> str:
        """Determine memory status based on allocation.
        
        Args:
            allocated: Current allocated memory in bytes
            
        Returns:
            Status string: 'normal', 'warning', or 'critical'
        """
        if allocated >= self._critical_bytes:
            return "critical"
        elif allocated >= self._warning_bytes:
            return "warning"
        return "normal"
    
    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            info = self.get_memory_info()
            
            if info['available']:
                status = self._get_status(info['allocated'])
                
                # Notify callbacks on status change
                if status != self._last_status:
                    self._last_status = status
                    for callback in self._callbacks:
                        try:
                            callback(status, info)
                        except Exception:
                            pass
            
            time.sleep(self.check_interval)
    
    def start(self) -> None:
        """Start the background monitoring thread."""
        if self._running:
            return
        
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self._monitor_thread.start()
    
    def stop(self) -> None:
        """Stop the background monitoring thread."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
            self._monitor_thread = None


class AdaptiveBatchSizer:
    """Dynamically adjusts batch size based on GPU memory pressure.
    
    Monitors memory usage and adjusts batch size to prevent OOM errors
    while maximizing throughput.
    """
    
    def __init__(
        self,
        initial_batch_size: int = 4,
        min_batch_size: int = 1,
        max_batch_size: int = 16,
        memory_limit_bytes: int = 6 * 1024 * 1024 * 1024
    ):
        """Initialize adaptive batch sizer.
        
        Args:
            initial_batch_size: Starting batch size
            min_batch_size: Minimum allowed batch size
            max_batch_size: Maximum allowed batch size
            memory_limit_bytes: GPU memory limit in bytes
        """
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.memory_limit = memory_limit_bytes
        
        self._current_batch_size = initial_batch_size
        self._pressure_threshold = 0.8
        self._growth_threshold = 0.5
        self._lock = threading.Lock()
        
        # Track memory usage history for smarter adjustment
        self._memory_history: List[float] = []
        self._history_size = 10

    def get_batch_size(self) -> int:
        """Get current batch size.
        
        Returns:
            Current batch size
        """
        with self._lock:
            return self._current_batch_size
    
    def _get_memory_usage_fraction(self) -> float:
        """Get current memory usage as fraction of limit."""
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated()
                return allocated / self.memory_limit
        except ImportError:
            pass
        return 0.0
    
    def update(self) -> int:
        """Update batch size based on current memory pressure.
        
        Returns:
            Updated batch size
        """
        with self._lock:
            usage = self._get_memory_usage_fraction()
            
            # Update history
            self._memory_history.append(usage)
            if len(self._memory_history) > self._history_size:
                self._memory_history.pop(0)
            
            # Calculate average usage
            avg_usage = sum(self._memory_history) / len(self._memory_history)
            
            # Adjust batch size
            if avg_usage > self._pressure_threshold:
                # Reduce batch size
                self._current_batch_size = max(
                    self.min_batch_size,
                    self._current_batch_size // 2
                )
            elif avg_usage < self._growth_threshold and len(self._memory_history) >= 5:
                # Gradually increase batch size
                self._current_batch_size = min(
                    self.max_batch_size,
                    self._current_batch_size + 1
                )
            
            return self._current_batch_size
    
    def reset(self) -> None:
        """Reset batch size to initial value."""
        with self._lock:
            self._current_batch_size = 4
            self._memory_history.clear()


class OperationQueue:
    """Queue for GPU operations with memory-aware scheduling.
    
    Queues operations and executes them when GPU memory is available,
    preventing OOM errors during high-load periods.
    """
    
    def __init__(
        self,
        memory_limit_bytes: int = 6 * 1024 * 1024 * 1024,
        safe_threshold: float = 0.7
    ):
        """Initialize operation queue.
        
        Args:
            memory_limit_bytes: GPU memory limit in bytes
            safe_threshold: Memory usage fraction below which operations execute
        """
        self.memory_limit = memory_limit_bytes
        self.safe_threshold = safe_threshold
        self._safe_bytes = int(memory_limit_bytes * safe_threshold)
        
        self._queue: Queue = Queue()
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def _is_memory_safe(self) -> bool:
        """Check if memory usage is below safe threshold."""
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated()
                return allocated < self._safe_bytes
        except ImportError:
            pass
        return True
    
    def enqueue(
        self,
        operation: Callable,
        args: Tuple = (),
        kwargs: Optional[Dict] = None,
        priority: int = 0
    ) -> None:
        """Add an operation to the queue.
        
        Args:
            operation: Callable to execute
            args: Positional arguments
            kwargs: Keyword arguments
            priority: Priority (lower = higher priority)
        """
        if kwargs is None:
            kwargs = {}
        self._queue.put((priority, operation, args, kwargs))
    
    def _worker_loop(self) -> None:
        """Background worker that processes queued operations."""
        while self._running:
            try:
                # Wait for memory to be available
                while self._running and not self._is_memory_safe():
                    time.sleep(0.1)
                
                if not self._running:
                    break
                
                # Get and execute operation
                try:
                    priority, operation, args, kwargs = self._queue.get(timeout=0.5)
                    operation(*args, **kwargs)
                    self._queue.task_done()
                except Empty:
                    continue
                except Exception as e:
                    print(f"Operation queue error: {e}")
                    
            except Exception:
                pass
    
    def start(self) -> None:
        """Start the background worker thread."""
        if self._running:
            return
        
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True
        )
        self._worker_thread.start()
    
    def stop(self) -> None:
        """Stop the background worker thread."""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=2.0)
            self._worker_thread = None
    
    def pending_count(self) -> int:
        """Get number of pending operations."""
        return self._queue.qsize()
    
    def clear(self) -> None:
        """Clear all pending operations."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Empty:
                break


def clear_gpu_cache() -> None:
    """Clear GPU memory cache to free unused memory."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


def get_gpu_memory_summary() -> str:
    """Get a human-readable GPU memory summary.
    
    Returns:
        Formatted string with memory statistics
    """
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            max_alloc = torch.cuda.max_memory_allocated() / (1024**3)
            
            return (
                f"GPU Memory: {allocated:.2f} GB allocated, "
                f"{reserved:.2f} GB reserved, "
                f"{max_alloc:.2f} GB peak"
            )
    except ImportError:
        pass
    
    return "GPU not available"
