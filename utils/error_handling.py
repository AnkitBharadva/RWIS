"""Error handling utilities for the Railway Wagon Inspection Pipeline.

This module provides custom exceptions, retry logic, and graceful
degradation mechanisms for robust pipeline operation.

Requirements: 7.3, 7.4
"""

import functools
import time
import traceback
from typing import Callable, Optional, Type, Tuple, Any, List


# Custom Exceptions

class PipelineError(Exception):
    """Base exception for all pipeline errors."""
    pass


class ModelNotFoundError(PipelineError):
    """Raised when a required model file is not found."""
    
    def __init__(self, model_path: str, model_type: str = "model"):
        self.model_path = model_path
        self.model_type = model_type
        super().__init__(
            f"{model_type.capitalize()} not found at: {model_path}. "
            f"Please ensure the model file exists and the path is correct."
        )


class ModelLoadError(PipelineError):
    """Raised when a model fails to load."""
    
    def __init__(self, model_path: str, reason: str = ""):
        self.model_path = model_path
        self.reason = reason
        message = f"Failed to load model from: {model_path}"
        if reason:
            message += f". Reason: {reason}"
        super().__init__(message)


class VideoSourceError(PipelineError):
    """Raised when video source cannot be opened or read."""
    
    def __init__(self, source: str, reason: str = ""):
        self.source = source
        self.reason = reason
        message = f"Failed to open video source: {source}"
        if reason:
            message += f". Reason: {reason}"
        super().__init__(message)


class StreamConnectionError(PipelineError):
    """Raised when stream connection fails after retries."""
    
    def __init__(self, url: str, attempts: int):
        self.url = url
        self.attempts = attempts
        super().__init__(
            f"Failed to connect to stream after {attempts} attempts: {url}"
        )


class ConfigurationError(PipelineError):
    """Raised when configuration is invalid."""
    
    def __init__(self, message: str, field: Optional[str] = None):
        self.field = field
        if field:
            message = f"Configuration error in '{field}': {message}"
        super().__init__(message)


class FrameProcessingError(PipelineError):
    """Raised when frame processing fails."""
    
    def __init__(self, frame_index: int, stage: str, reason: str = ""):
        self.frame_index = frame_index
        self.stage = stage
        self.reason = reason
        message = f"Frame {frame_index} processing failed at stage '{stage}'"
        if reason:
            message += f": {reason}"
        super().__init__(message)


class GPUMemoryError(PipelineError):
    """Raised when GPU memory is exhausted."""
    
    def __init__(self, allocated: int, limit: int):
        self.allocated = allocated
        self.limit = limit
        super().__init__(
            f"GPU memory exhausted: {allocated / (1024**3):.2f} GB allocated, "
            f"limit is {limit / (1024**3):.2f} GB"
        )


class FP16InstabilityError(PipelineError):
    """Raised when FP16 numerical instability is detected."""
    
    def __init__(self, operation: str = "inference", fallback_used: bool = False):
        self.operation = operation
        self.fallback_used = fallback_used
        if fallback_used:
            message = f"FP16 instability detected during {operation}, fell back to FP32"
        else:
            message = f"FP16 instability detected during {operation}, FP32 fallback disabled"
        super().__init__(message)


class DeblurError(PipelineError):
    """Raised when deblurring operation fails."""
    
    def __init__(self, reason: str = "", wagon_id: Optional[int] = None):
        self.reason = reason
        self.wagon_id = wagon_id
        message = "Deblurring operation failed"
        if wagon_id is not None:
            message += f" for wagon {wagon_id}"
        if reason:
            message += f": {reason}"
        super().__init__(message)


# Retry Decorator

def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None
) -> Callable:
    """Decorator that retries a function with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        backoff_factor: Multiplier for delay after each retry
        exceptions: Tuple of exception types to catch and retry
        on_retry: Optional callback called on each retry with (exception, attempt)
    
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        if on_retry:
                            on_retry(e, attempt + 1)
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        raise
            
            raise last_exception
        
        return wrapper
    return decorator


# Graceful Degradation

class GracefulDegradation:
    """Manages graceful degradation when components fail.
    
    Tracks component failures and provides fallback behavior
    to keep the pipeline running with reduced functionality.
    """
    
    def __init__(self):
        """Initialize graceful degradation manager."""
        self._disabled_components: set = set()
        self._error_counts: dict = {}
        self._max_errors_before_disable = 5
    
    def record_error(self, component: str, error: Exception) -> bool:
        """Record an error for a component.
        
        Args:
            component: Name of the component that failed
            error: The exception that occurred
            
        Returns:
            True if component should be disabled, False otherwise
        """
        if component not in self._error_counts:
            self._error_counts[component] = 0
        
        self._error_counts[component] += 1
        
        if self._error_counts[component] >= self._max_errors_before_disable:
            self.disable_component(component)
            return True
        
        return False
    
    def disable_component(self, component: str) -> None:
        """Disable a component.
        
        Args:
            component: Name of the component to disable
        """
        self._disabled_components.add(component)
        print(f"WARNING: Component '{component}' has been disabled due to repeated errors")
    
    def is_disabled(self, component: str) -> bool:
        """Check if a component is disabled.
        
        Args:
            component: Name of the component to check
            
        Returns:
            True if component is disabled, False otherwise
        """
        return component in self._disabled_components
    
    def reset_component(self, component: str) -> None:
        """Reset a component's error count and re-enable it.
        
        Args:
            component: Name of the component to reset
        """
        self._disabled_components.discard(component)
        self._error_counts.pop(component, None)
    
    def get_status(self) -> dict:
        """Get status of all components.
        
        Returns:
            Dictionary with component status information
        """
        return {
            'disabled_components': list(self._disabled_components),
            'error_counts': dict(self._error_counts)
        }


# Error Context Manager

class ErrorContext:
    """Context manager for handling errors in pipeline stages.
    
    Provides consistent error handling and logging for pipeline operations.
    """
    
    def __init__(
        self,
        stage_name: str,
        frame_index: Optional[int] = None,
        graceful: GracefulDegradation = None,
        suppress: bool = False
    ):
        """Initialize error context.
        
        Args:
            stage_name: Name of the pipeline stage
            frame_index: Optional frame index for error reporting
            graceful: Optional graceful degradation manager
            suppress: If True, suppress exceptions and return None
        """
        self.stage_name = stage_name
        self.frame_index = frame_index
        self.graceful = graceful
        self.suppress = suppress
        self.error: Optional[Exception] = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            self.error = exc_val
            
            # Log the error
            frame_info = f" (frame {self.frame_index})" if self.frame_index else ""
            print(f"Error in {self.stage_name}{frame_info}: {exc_val}")
            
            # Record error for graceful degradation
            if self.graceful:
                self.graceful.record_error(self.stage_name, exc_val)
            
            # Suppress exception if requested
            return self.suppress
        
        return False
    
    def succeeded(self) -> bool:
        """Check if the operation succeeded without errors."""
        return self.error is None


# Configuration Validation

def validate_config(config) -> List[str]:
    """Validate pipeline configuration and return list of errors.
    
    Args:
        config: PipelineConfig object to validate
        
    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    
    # Validate blur thresholds (legacy)
    if config.blur_threshold_t1 >= config.blur_threshold_t2:
        errors.append(
            f"blur_threshold_t1 ({config.blur_threshold_t1}) must be less than "
            f"blur_threshold_t2 ({config.blur_threshold_t2})"
        )
    
    if config.blur_threshold_t1 < 0:
        errors.append(f"blur_threshold_t1 must be non-negative, got {config.blur_threshold_t1}")
    
    if config.blur_threshold_t2 < 0:
        errors.append(f"blur_threshold_t2 must be non-negative, got {config.blur_threshold_t2}")
    
    # Validate single blur threshold
    if hasattr(config, 'blur_threshold') and config.blur_threshold < 0:
        errors.append(f"blur_threshold must be non-negative, got {config.blur_threshold}")
    
    # Validate counting line position
    if not 0.0 <= config.counting_line_position <= 1.0:
        errors.append(
            f"counting_line_position must be between 0.0 and 1.0, "
            f"got {config.counting_line_position}"
        )
    
    # Validate confidence thresholds
    if not 0.0 <= config.wagon_confidence_threshold <= 1.0:
        errors.append(
            f"wagon_confidence_threshold must be between 0.0 and 1.0, "
            f"got {config.wagon_confidence_threshold}"
        )
    
    if not 0.0 <= config.damage_confidence_threshold <= 1.0:
        errors.append(
            f"damage_confidence_threshold must be between 0.0 and 1.0, "
            f"got {config.damage_confidence_threshold}"
        )
    
    # Validate batch size
    if config.max_batch_size < 1:
        errors.append(f"max_batch_size must be at least 1, got {config.max_batch_size}")
    
    # Validate N-th frame interval
    if hasattr(config, 'deblur_frame_interval') and config.deblur_frame_interval < 1:
        errors.append(f"deblur_frame_interval must be at least 1, got {config.deblur_frame_interval}")
    
    # Validate max ROI width
    if hasattr(config, 'max_roi_width') and config.max_roi_width < 1:
        errors.append(f"max_roi_width must be at least 1, got {config.max_roi_width}")
    
    # Validate log formats
    valid_formats = {'csv', 'json'}
    invalid_formats = set(config.log_format) - valid_formats
    if invalid_formats:
        errors.append(f"Invalid log formats: {invalid_formats}. Valid formats: {valid_formats}")
    
    return errors


def validate_model_paths(config) -> List[str]:
    """Validate that model files exist.
    
    Args:
        config: PipelineConfig object with model paths
        
    Returns:
        List of error messages for missing models
    """
    import os
    errors = []
    
    model_paths = [
        ('wagon_model_path', config.wagon_model_path, 'Wagon detection'),
        ('damage_model_path', config.damage_model_path, 'Damage detection'),
    ]
    
    # Check for MPRNet model path (preferred)
    if hasattr(config, 'mprnet_model_path') and config.mprnet_model_path:
        model_paths.append(('mprnet_model_path', config.mprnet_model_path, 'MPRNet deblur'))
    # Fall back to NAFNet path for backward compatibility
    elif hasattr(config, 'nafnet_model_path') and config.nafnet_model_path:
        model_paths.append(('nafnet_model_path', config.nafnet_model_path, 'NAFNet deblur'))
    
    for field, path, name in model_paths:
        if path and not os.path.exists(path):
            errors.append(f"{name} model not found at: {path}")
    
    return errors


def format_error_report(errors: List[str]) -> str:
    """Format a list of errors into a readable report.
    
    Args:
        errors: List of error messages
        
    Returns:
        Formatted error report string
    """
    if not errors:
        return "No errors found."
    
    report = f"Found {len(errors)} configuration error(s):\n"
    for i, error in enumerate(errors, 1):
        report += f"  {i}. {error}\n"
    
    return report
