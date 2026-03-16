"""Main pipeline controller for Railway Wagon Inspection.

This module implements the main video processing pipeline that wires together
all pipeline stages: frame capture, CLAHE enhancement, wagon detection, tracking,
ROI extraction, blur detection, conditional deblurring via DeblurManager,
OCR, damage detection, and logging.

Key features:
- Uses MPRNet for ROI-only deblurring
- N-th frame execution with ROI caching
- FP16 inference with FP32 fallback
- Wagon exit callback for cache cleanup
- GPU memory management within 6 GB VRAM limit

Requirements: 1.1, 9.6, 9.7, 7.5, 9.1, 9.8, 10.3, 10.4
"""

import os
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any, Set

import cv2
import numpy as np

from config import PipelineConfig, load_config
from pipelines.blur_detector import BlurDetector
from utils.fast_roi_utils import extract_roi_fast, resize_roi_fast, ROIBufferPool
from pipelines.fast_blur_detector import FastBlurDetector
from pipelines.mprnet_onnx_wrapper import MPRNetONNXWrapper
from pipelines.wagon_detector import WagonDetector, FrameProcessingState
from pipelines.damage_detector import DamageDetector
from pipelines.ocr_pipeline import OCRPipeline
from pipelines.mprnet_wrapper import MPRNetDeblur
from pipelines.deblur_manager import DeblurManager
from pipelines.parallel_processor import ParallelROIProcessor
from pipelines.cached_ocr_pipeline import CachedOCRPipeline
from tracking.tracker import WagonTracker
from utils.clahe import CLAHEEnhancer
from utils.logger import InspectionLogger
from utils.performance_monitor import PerformanceMonitor
from utils.roi_utils import extract_roi, resize_roi_for_deblur
from utils.data_models import (
    BoundingBox, WagonRecord, TrackedWagon, DamageDetection, OCRResult
)
from utils.gpu_utils import (
    GPUMemoryMonitor, AdaptiveBatchSizer, OperationQueue,
    clear_gpu_cache, get_gpu_memory_summary
)
from utils.error_handling import (
    PipelineError, ModelNotFoundError, ModelLoadError,
    VideoSourceError, StreamConnectionError, ConfigurationError,
    FrameProcessingError, GPUMemoryError,
    retry_with_backoff, GracefulDegradation, ErrorContext,
    validate_config, validate_model_paths, format_error_report
)


class GPUMemoryManager:
    """Manages GPU memory to stay within safe limits (6 GB VRAM).
    
    Monitors GPU memory usage and provides mechanisms for batch size
    adjustment and operation queuing when memory pressure is detected.
    
    Requirements: 7.5, 9.1, 9.8
    """
    
    # Default memory limit (6 GB in bytes)
    DEFAULT_MEMORY_LIMIT = 6 * 1024 * 1024 * 1024
    
    # Memory pressure threshold (80% of limit)
    PRESSURE_THRESHOLD = 0.8
    
    def __init__(self, memory_limit: int = DEFAULT_MEMORY_LIMIT):
        """Initialize GPU memory manager.
        
        Args:
            memory_limit: Maximum GPU memory in bytes (default: 6 GB)
        """
        self.memory_limit = memory_limit
        self.pressure_threshold = int(memory_limit * self.PRESSURE_THRESHOLD)
        
        # Use enhanced GPU utilities
        self._memory_monitor = GPUMemoryMonitor(memory_limit_bytes=memory_limit)
        self._batch_sizer = AdaptiveBatchSizer(
            initial_batch_size=1,  # NAFNet always uses batch size 1
            min_batch_size=1,
            max_batch_size=1,  # Enforce batch size 1 for NAFNet
            memory_limit_bytes=memory_limit
        )
        self._operation_queue = OperationQueue(memory_limit_bytes=memory_limit)
        
        # Register memory status callback
        self._memory_monitor.add_callback(self._on_memory_status_change)
        
        # Track FP32 fallback occurrences
        self._fp32_fallback_count = 0
    
    def _on_memory_status_change(self, status: str, info: Dict) -> None:
        """Handle memory status changes.
        
        Args:
            status: Memory status ('normal', 'warning', 'critical')
            info: Memory information dictionary
        """
        if status == "critical":
            print(f"CRITICAL: GPU memory at {info['usage_percent']:.1f}%")
            self.clear_cache()
        elif status == "warning":
            print(f"WARNING: GPU memory at {info['usage_percent']:.1f}%")
    
    def start_monitoring(self) -> None:
        """Start background memory monitoring."""
        self._memory_monitor.start()
        self._operation_queue.start()
    
    def stop_monitoring(self) -> None:
        """Stop background memory monitoring."""
        self._memory_monitor.stop()
        self._operation_queue.stop()
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Get current GPU memory usage.
        
        Returns:
            Dictionary with 'allocated', 'reserved', 'max_allocated' in bytes
        """
        return self._memory_monitor.get_memory_info()
    
    def is_memory_pressure(self) -> bool:
        """Check if GPU memory is under pressure.
        
        Returns:
            True if memory usage exceeds pressure threshold
        """
        info = self.get_memory_usage()
        return info['allocated'] > self.pressure_threshold

    def get_batch_size(self) -> int:
        """Get current batch size (always 1 for NAFNet)."""
        return 1  # NAFNet always uses batch size 1
    
    def clear_cache(self) -> None:
        """Clear GPU memory cache."""
        clear_gpu_cache()
    
    def queue_operation(self, operation: callable, *args, **kwargs) -> None:
        """Queue an operation for later execution when memory is available.
        
        Args:
            operation: Callable to execute
            *args: Positional arguments for the operation
            **kwargs: Keyword arguments for the operation
        """
        self._operation_queue.enqueue(operation, args, kwargs)
    
    def process_queued_operations(self) -> int:
        """Process queued operations if memory is available.
        
        Returns:
            Number of pending operations
        """
        return self._operation_queue.pending_count()
    
    def log_fp32_fallback(self) -> None:
        """Log an FP32 fallback occurrence."""
        self._fp32_fallback_count += 1
        print(f"WARNING: FP32 fallback triggered (total: {self._fp32_fallback_count})")
    
    def get_fp32_fallback_count(self) -> int:
        """Get the number of FP32 fallback occurrences."""
        return self._fp32_fallback_count
    
    def get_memory_summary(self) -> str:
        """Get human-readable memory summary."""
        summary = get_gpu_memory_summary()
        if self._fp32_fallback_count > 0:
            summary += f" | FP32 fallbacks: {self._fp32_fallback_count}"
        return summary


class RailwayWagonPipeline:
    """Main pipeline controller for railway wagon inspection.
    
    Orchestrates all pipeline stages in correct order:
    1. Frame capture from video source
    2. Optional CLAHE enhancement
    3. Wagon detection (RAW/CLAHE only - never deblurred)
    4. Wagon tracking and counting
    5. ROI extraction and resizing (max 256px width)
    6. Blur detection on ROI
    7. Conditional deblurring via DeblurManager (N-th frame execution)
    8. OCR on processed ROI (raw or deblurred)
    9. Damage detection on RAW wagon ROI
    10. Logging results with deblur metadata
    
    Key constraints:
    - YOLO models receive only RAW or CLAHE-enhanced frames (never deblurred)
    - NAFNet deblur is ROI-only (max 256px width)
    - NAFNet runs every N frames with caching
    - FP16 inference with FP32 fallback
    - NAFNet is NOT used anywhere
    
    Requirements: 1.1, 9.6, 9.7
    """
    
    # Retry settings for stream connection
    MAX_STREAM_RETRIES = 3
    RETRY_DELAY_SECONDS = 2.0
    
    def __init__(self, config: PipelineConfig, display: bool = False):
        """Initialize the pipeline with configuration.
        
        Args:
            config: Pipeline configuration object
            display: Whether to show video output window
            
        Raises:
            ConfigurationError: If configuration is invalid
            ModelNotFoundError: If required model files are missing
        """
        self.config = config
        self._display = display
        self._validate_config()
        
        # Pipeline state
        self._running = False
        self._shutdown_event = threading.Event()
        self._frame_index = 0
        
        # GPU memory manager
        self.memory_manager = GPUMemoryManager()
        
        # Graceful degradation manager
        self._graceful = GracefulDegradation()
        
        # Frame processing state (shared between components)
        self._frame_state = FrameProcessingState()
        
        # Track active wagon IDs for cache cleanup
        self._active_wagon_ids: Set[int] = set()
        self._previous_wagon_ids: Set[int] = set()
        
        # Initialize components (lazy loading)
        self._blur_detector: Optional[BlurDetector] = None
        self._wagon_detector: Optional[WagonDetector] = None
        self._damage_detector: Optional[DamageDetector] = None
        self._ocr_pipeline: Optional[OCRPipeline] = None
        self._mprnet: Optional[MPRNetDeblur] = None
        self._deblur_manager: Optional[DeblurManager] = None
        self._tracker: Optional[WagonTracker] = None
        self._clahe: Optional[CLAHEEnhancer] = None
        self._logger: Optional[InspectionLogger] = None
        
        # Performance optimization components
        self.perf_monitor = PerformanceMonitor(window_size=100)
        self.parallel_processor: Optional[ParallelROIProcessor] = None
        
        # Video capture
        self._video_capture: Optional[cv2.VideoCapture] = None
        self._frame_width = 0
        self._frame_height = 0
        self._fps = 30.0
        
        # Thread pool for async processing
        self._executor: Optional[ThreadPoolExecutor] = None

    def _validate_config(self) -> None:
        """Validate pipeline configuration.
        
        Raises:
            ConfigurationError: If configuration is invalid
        """
        # Validate configuration values
        errors = validate_config(self.config)
        
        if not self.config.video_source:
            errors.append("video_source must be specified")
        
        if errors:
            raise ConfigurationError(format_error_report(errors))
        
        # Validate model paths (warning only, not fatal)
        model_errors = validate_model_paths(self.config)
        for error in model_errors:
            print(f"Warning: {error}")

    def _init_components(self) -> None:
        """Initialize all pipeline components.
        
        Raises:
            ModelNotFoundError: If required model files are missing
            ModelLoadError: If model loading fails
        """
        print("Initializing pipeline components...")
        
        # Initialize blur detector with single threshold
        try:
            # Use optimized blur detector if enabled
            if getattr(self.config, 'use_fast_blur_detector', False):
                threshold_t1 = getattr(self.config, 'blur_threshold_t1', 100.0)
                threshold_t2 = getattr(self.config, 'blur_threshold_t2', 300.0)
                self._blur_detector = FastBlurDetector(
                    threshold_t1=threshold_t1,
                    threshold_t2=threshold_t2
                )
                print(f"  Fast blur detector initialized (t1: {threshold_t1}, t2: {threshold_t2})")
            else:
                # Use single threshold for simplified blur gating
                threshold = getattr(self.config, 'blur_threshold', self.config.blur_threshold_t1)
                self._blur_detector = BlurDetector(
                    t1=threshold,
                    t2=threshold * 3  # Upper bound for severe blur
                )
                print(f"  Blur detector initialized (threshold: {threshold})")
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize blur detector: {e}")
        
        # Initialize wagon detector
        try:
            self._wagon_detector = WagonDetector(
                model_path=self.config.wagon_model_path,
                confidence_threshold=self.config.wagon_confidence_threshold,
                frame_state=self._frame_state
            )
            print(f"  Wagon detector initialized")
        except FileNotFoundError:
            raise ModelNotFoundError(
                self.config.wagon_model_path,
                "Wagon detection"
            )
        except Exception as e:
            raise ModelLoadError(self.config.wagon_model_path, str(e))
        
        # Initialize damage detector
        try:
            self._damage_detector = DamageDetector(
                model_path=self.config.damage_model_path,
                confidence_threshold=self.config.damage_confidence_threshold
            )
            print(f"  Damage detector initialized")
        except FileNotFoundError:
            raise ModelNotFoundError(
                self.config.damage_model_path,
                "Damage detection"
            )
        except Exception as e:
            raise ModelLoadError(self.config.damage_model_path, str(e))
        
        # Initialize MPRNet deblur (non-fatal if fails)
        try:
            max_roi_width = getattr(self.config, 'max_roi_width', 256)
            device = 'cuda' if self.config.ocr_gpu_enabled else 'cpu'
            use_fp16 = getattr(self.config, 'use_fp16', True)
            
            # Try ONNX wrapper first if enabled
            use_onnx = getattr(self.config, 'use_onnx_mprnet', False)
            onnx_path = getattr(self.config, 'mprnet_onnx_path', 'models/mprnet_optimized.onnx')
            
            if use_onnx and os.path.exists(onnx_path):
                print(f"  Using ONNX MPRNet (2-3x faster)")
                self._mprnet = MPRNetONNXWrapper(
                    onnx_model_path=onnx_path,
                    device=device,
                    max_roi_width=max_roi_width,
                    max_roi_height=max_roi_width
                )
                self._mprnet.load_model()
                print(f"  ONNX MPRNet initialized")
            else:
                if use_onnx:
                    print(f"  ONNX model not found at {onnx_path}, falling back to PyTorch")
                self._mprnet = MPRNetDeblur(
                    model_path=self.config.mprnet_model_path,
                    device=device,
                    use_fp16=use_fp16,
                    fp32_fallback=getattr(self.config, 'fp32_fallback', True),
                    max_roi_width=max_roi_width,
                    max_roi_height=max_roi_width  # Use same for height
                )
                self._mprnet.load_model()
                print(f"  PyTorch MPRNet initialized (FP16: {self._mprnet.use_fp16})")
        except Exception as e:
            print(f"Warning: MPRNet model loading failed: {e}. Deblurring disabled.")
            self._graceful.disable_component('mprnet')
            self._mprnet = None
        
        # Initialize DeblurManager (only if MPRNet is available)
        if self._mprnet is not None:
            try:
                frame_interval = getattr(self.config, 'deblur_frame_interval', 3)
                max_roi_width = getattr(self.config, 'max_roi_width', 256)
                self._deblur_manager = DeblurManager(
                    mprnet=self._mprnet,
                    blur_detector=self._blur_detector,
                    frame_interval=frame_interval,
                    max_roi_width=max_roi_width
                )
                print(f"  DeblurManager initialized (interval: {frame_interval}, max_width: {max_roi_width})")
            except Exception as e:
                print(f"Warning: DeblurManager initialization failed: {e}. Deblurring disabled.")
                self._graceful.disable_component('deblur_manager')
                self._deblur_manager = None
        
        # Initialize OCR pipeline
        try:
            self._ocr_pipeline = OCRPipeline(
                gpu_enabled=self.config.ocr_gpu_enabled,
                language=self.config.ocr_language
            )
            print(f"  OCR pipeline initialized")
            
            # Wrap with caching if enabled
            if getattr(self.config, 'enable_ocr_cache', False):
                cache_size = getattr(self.config, 'ocr_cache_size', 100)
                self._ocr_pipeline = CachedOCRPipeline(
                    self._ocr_pipeline,
                    cache_size=cache_size
                )
                print(f"  OCR caching enabled (cache size: {cache_size})")
        except Exception as e:
            print(f"Warning: OCR pipeline initialization failed: {e}. OCR disabled.")
            self._graceful.disable_component('ocr')
        
        # Initialize parallel processor if enabled
        if getattr(self.config, 'parallel_roi_processing', False):
            try:
                max_workers = getattr(self.config, 'max_parallel_workers', 4)
                self.parallel_processor = ParallelROIProcessor(
                    ocr_pipeline=self._ocr_pipeline,
                    damage_detector=self._damage_detector,
                    deblur_manager=self._deblur_manager,
                    blur_detector=self._blur_detector,
                    logger_instance=self._logger,
                    max_workers=max_workers
                )
                print(f"  Parallel ROI processing enabled (workers: {max_workers})")
            except Exception as e:
                print(f"Warning: Parallel processor initialization failed: {e}. Using sequential processing.")
                self.parallel_processor = None
        
        # Initialize tracker
        try:
            orientation = getattr(self.config, 'counting_line_orientation', 'vertical')
            self._tracker = WagonTracker(
                counting_line_y=self.config.counting_line_position,
                orientation=orientation
            )
            print(f"  Wagon tracker initialized (line: {self.config.counting_line_position}, orientation: {orientation})")
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize tracker: {e}")
        
        # Initialize CLAHE enhancer
        self._clahe = CLAHEEnhancer()
        print(f"  CLAHE enhancer initialized")
        
        # Initialize logger
        try:
            self._logger = InspectionLogger(
                output_dir=self.config.output_dir,
                formats=self.config.log_format,
                enable_debug=self.config.enable_debug_frames
            )
            print(f"  Logger initialized (output: {self.config.output_dir})")
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize logger: {e}")
        
        print("All components initialized successfully.")

    def _open_video_source(self) -> None:
        """Open video source with retry logic.
        
        Raises:
            VideoSourceError: If video file cannot be opened
            StreamConnectionError: If stream connection fails after retries
        """
        source = self.config.video_source
        is_stream = source.startswith(('rtsp://', 'http://', 'https://'))
        
        def on_retry(error, attempt):
            print(f"Stream connection attempt {attempt} failed: {error}")
            print(f"Retrying in {self.RETRY_DELAY_SECONDS}s...")
        
        @retry_with_backoff(
            max_retries=self.MAX_STREAM_RETRIES if is_stream else 0,
            initial_delay=self.RETRY_DELAY_SECONDS,
            exceptions=(Exception,),
            on_retry=on_retry
        )
        def try_open():
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                raise VideoSourceError(source, "Failed to open video capture")
            return cap
        
        try:
            self._video_capture = try_open()
            
            # Get video properties
            self._frame_width = int(
                self._video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            )
            self._frame_height = int(
                self._video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            )
            self._fps = self._video_capture.get(cv2.CAP_PROP_FPS) or 30.0
            
            # Set full frame dimensions for NAFNet rejection
            if self._mprnet:
                self._mprnet.set_full_frame_dimensions(
                    self._frame_width, self._frame_height
                )
                
        except VideoSourceError:
            if is_stream:
                raise StreamConnectionError(source, self.MAX_STREAM_RETRIES)
            raise
    
    def _read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the video source.
        
        Returns:
            Tuple of (success, frame). Frame is None if read failed.
        """
        if self._video_capture is None:
            return False, None
        
        ret, frame = self._video_capture.read()
        if not ret:
            return False, None
        
        return True, frame

    def _on_wagon_exit(self, wagon_id: int) -> None:
        """Callback for wagon exit - clears deblur cache.
        
        This is called when a wagon exits the tracking area to free
        memory and prevent stale cache entries.
        
        Args:
            wagon_id: ID of the wagon that exited
        """
        if self._deblur_manager is not None:
            self._deblur_manager.clear_cache(wagon_id)
    
    def _check_wagon_exits(self, current_wagon_ids: Set[int]) -> None:
        """Check for wagons that have exited and trigger cleanup.
        
        Args:
            current_wagon_ids: Set of wagon IDs currently being tracked
        """
        # Find wagons that were in previous frame but not in current
        exited_wagons = self._previous_wagon_ids - current_wagon_ids
        
        for wagon_id in exited_wagons:
            self._on_wagon_exit(wagon_id)
        
        # Update previous wagon IDs for next frame
        self._previous_wagon_ids = current_wagon_ids.copy()

    def _process_frame(
        self,
        frame: np.ndarray,
        frame_index: int
    ) -> Tuple[List[WagonRecord], List[TrackedWagon], List]:
        """Process a single frame through all pipeline stages.
        
        Pipeline stages (in order):
        1. Frame capture (already done)
        2. Optional CLAHE enhancement
        3. Wagon detection (RAW/CLAHE only)
        4. Wagon tracking and counting
        5. ROI extraction and resizing
        6. Blur detection on ROI
        7. Conditional deblurring via DeblurManager
        8. OCR on processed ROI
        9. Damage detection on RAW wagon ROI
        10. Logging results with deblur metadata
        
        Args:
            frame: Input frame (BGR format)
            frame_index: Index of the frame
            
        Returns:
            Tuple of (List of WagonRecord objects, List of TrackedWagon objects, List of raw detections)
        """
        records = []
        tracked_wagons = []
        raw_detections = []
        
        # Stage 2: Optional CLAHE enhancement for detection
        # YOLO receives RAW or CLAHE-enhanced frames only (never deblurred)
        detection_frame = frame
        with ErrorContext('clahe_enhancement', frame_index, self._graceful, suppress=True) as ctx:
            detection_frame = self._clahe.enhance(frame)
        if not ctx.succeeded():
            detection_frame = frame
        
        # Stage 3: Wagon detection (RAW/CLAHE only)
        detections = []
        with ErrorContext('wagon_detection', frame_index, self._graceful, suppress=True) as ctx:
            detections = self._wagon_detector.detect(
                detection_frame,
                frame_id=frame_index
            )
        
        # Store raw detections for display
        raw_detections = detections if detections else []
        
        if not ctx.succeeded() or not detections:
            return records, tracked_wagons, raw_detections
        
        # Stage 4: Wagon tracking and counting
        frame_shape = (self._frame_height, self._frame_width, 3)
        with ErrorContext('wagon_tracking', frame_index, self._graceful, suppress=True) as ctx:
            tracked_wagons = self._tracker.update(detections, frame_shape, frame_index)
        
        if not ctx.succeeded():
            return records, [], raw_detections
        
        # Track current wagon IDs for exit detection
        current_wagon_ids = {w.track_id for w in tracked_wagons}
        self._check_wagon_exits(current_wagon_ids)
        
        # Use parallel processing if available, otherwise sequential
        if self.parallel_processor is not None:
            # Parallel processing of all wagons
            records = self.parallel_processor.process_wagon_batch(
                frame, tracked_wagons, frame_index, self.config.max_roi_width
            )
        else:
            # Sequential processing (original method)
            records = self._process_wagons_sequential(frame, tracked_wagons, frame_index)
        
        return records, tracked_wagons, raw_detections
    
    def _process_wagons_sequential(
        self,
        frame: np.ndarray,
        tracked_wagons: List[TrackedWagon],
        frame_index: int
    ) -> List[WagonRecord]:
        """Process wagons sequentially (fallback method).
        
        Args:
            frame: Input frame
            tracked_wagons: List of tracked wagons
            frame_index: Current frame index
            
        Returns:
            List of wagon records
        """
        records = []
        
        # Process each tracked wagon
        for wagon in tracked_wagons:
            # Only process wagons that have crossed the counting line
            if not wagon.crossed_line or wagon.count_index is None:
                continue
            
            # Stage 5: ROI extraction (from RAW frame for damage detection)
            raw_roi = None
            with ErrorContext('roi_extraction', frame_index, self._graceful, suppress=True):
                # Use fast ROI extraction if enabled
                if getattr(self.config, 'use_fast_roi_utils', False):
                    raw_roi, actual_bbox = extract_roi_fast(frame, wagon.bbox, clip_to_bounds=True)
                else:
                    raw_roi, actual_bbox = extract_roi(frame, wagon.bbox, clip_to_bounds=True)
            
            if raw_roi is None or raw_roi.size == 0:
                continue
            
            # Stage 6 & 7: Blur detection and conditional deblurring via DeblurManager
            processed_roi = raw_roi
            deblur_applied = False
            deblur_source_frame = None
            blur_score = 0.0
            
            if self._deblur_manager is not None and not self._graceful.is_disabled('deblur_manager'):
                with ErrorContext('deblur_processing', frame_index, self._graceful, suppress=True) as ctx:
                    # DeblurManager handles: resize, blur detection, N-th frame logic, caching
                    processed_roi, deblur_applied, deblur_source_frame = self._deblur_manager.process_roi(
                        raw_roi,
                        wagon.track_id,
                        frame_index
                    )
                    # Get blur score for logging
                    resized_roi, _ = resize_roi_for_deblur(raw_roi, self.config.max_roi_width)
                    blur_score = self._blur_detector.compute_blur_score(resized_roi)
                
                if not ctx.succeeded():
                    processed_roi = raw_roi
                    deblur_applied = False
                    deblur_source_frame = None
                
                # Log FP32 fallback if it occurred
                if self._mprnet and self._mprnet.is_using_fp32_fallback():
                    self.memory_manager.log_fp32_fallback()
            else:
                # No deblur manager - compute blur score anyway for logging
                with ErrorContext('blur_detection', frame_index, self._graceful, suppress=True):
                    blur_score = self._blur_detector.compute_blur_score(raw_roi)
            
            # Stage 8: OCR on processed ROI (raw or deblurred)
            ocr_result = self._extract_ocr(processed_roi)
            
            # Stage 9: Damage detection on RAW wagon ROI (never deblurred)
            damage_detections = self._detect_damage(raw_roi, wagon.track_id)
            
            # Stage 10: Create and log wagon record with deblur metadata
            record = self._create_wagon_record(
                wagon=wagon,
                blur_score=blur_score,
                frame_index=frame_index,
                damage_detections=damage_detections,
                ocr_result=ocr_result,
                deblur_applied=deblur_applied,
                deblur_source_frame=deblur_source_frame
            )
            records.append(record)
            
            # Log the record
            with ErrorContext('logging', frame_index, self._graceful, suppress=True):
                self._logger.log_wagon(record)
        
        return records

    def _detect_damage(
        self,
        roi: np.ndarray,
        wagon_id: int
    ) -> List[DamageDetection]:
        """Detect damage on wagon ROI (RAW only, never deblurred).
        
        Args:
            roi: Wagon ROI image (RAW or CLAHE-enhanced)
            wagon_id: Track ID of the wagon
            
        Returns:
            List of damage detections
        """
        # Skip if damage detection is disabled
        if self._graceful.is_disabled('damage_detection'):
            return []
        
        with ErrorContext('damage_detection', graceful=self._graceful, suppress=True) as ctx:
            return self._damage_detector.detect(roi, wagon_id)
        
        return []
    
    def _extract_ocr(self, roi: np.ndarray) -> OCRResult:
        """Extract text from wagon ROI using OCR.
        
        Args:
            roi: Wagon ROI image (raw or deblurred)
            
        Returns:
            OCR result with extracted text and confidence
        """
        # Skip if OCR is disabled
        if self._graceful.is_disabled('ocr'):
            return OCRResult(text="", confidence=0.0, bbox=None)
        
        with ErrorContext('ocr', graceful=self._graceful, suppress=True) as ctx:
            return self._ocr_pipeline.extract_text(roi)
        
        return OCRResult(text="", confidence=0.0, bbox=None)
    
    def _create_wagon_record(
        self,
        wagon: TrackedWagon,
        blur_score: float,
        frame_index: int,
        damage_detections: List[DamageDetection],
        ocr_result: OCRResult,
        deblur_applied: bool = False,
        deblur_source_frame: Optional[int] = None
    ) -> WagonRecord:
        """Create a wagon record for logging with deblur metadata.
        
        Args:
            wagon: Tracked wagon object
            blur_score: Computed blur score
            frame_index: Frame index
            damage_detections: List of damage detections
            ocr_result: OCR extraction result
            deblur_applied: Whether deblurring was applied
            deblur_source_frame: Frame index where deblurring was performed
            
        Returns:
            WagonRecord for logging
        """
        damage_classes = [d.damage_class.value for d in damage_detections]
        damage_bboxes = [d.bbox for d in damage_detections]
        
        return WagonRecord(
            timestamp=datetime.now().isoformat(),
            wagon_id=wagon.track_id,
            count_index=wagon.count_index,
            blur_score=blur_score,
            frame_index=frame_index,
            damage_detected=len(damage_detections) > 0,
            damage_classes=damage_classes,
            damage_bboxes=damage_bboxes,
            ocr_text=ocr_result.text,
            ocr_confidence=ocr_result.confidence,
            deblur_applied=deblur_applied,
            deblur_source_frame=deblur_source_frame
        )

    def _save_debug_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        tracked_wagons: List[TrackedWagon]
    ) -> None:
        """Save annotated debug frame.
        
        Args:
            frame: Original frame
            frame_index: Frame index
            tracked_wagons: List of tracked wagons to annotate
        """
        annotations = []
        
        # Add wagon annotations
        for wagon in tracked_wagons:
            color = (0, 255, 0) if wagon.crossed_line else (0, 255, 255)
            label = f"ID:{wagon.track_id}"
            if wagon.count_index:
                label += f" #{wagon.count_index}"
            
            annotations.append({
                'bbox': wagon.bbox,
                'label': label,
                'color': color
            })
        
        # Draw counting line
        line_y = int(self.config.counting_line_position * self._frame_height)
        annotated = frame.copy()
        cv2.line(
            annotated,
            (0, line_y),
            (self._frame_width, line_y),
            (255, 0, 0),
            2
        )
        
        self._logger.save_debug_frame(annotated, frame_index, annotations)

    def _draw_display_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        tracked_wagons: List[TrackedWagon],
        raw_detections: List = None
    ) -> np.ndarray:
        """Draw annotations on frame for display.
        
        Args:
            frame: Original frame
            frame_index: Frame index
            tracked_wagons: List of tracked wagons to annotate
            raw_detections: List of raw detections from detector
            
        Returns:
            Annotated frame for display
        """
        display_frame = frame.copy()
        
        # Draw counting line based on orientation
        orientation = getattr(self.config, 'counting_line_orientation', 'vertical')
        
        if orientation == "vertical":
            # Vertical line (for horizontal wagon movement)
            line_x = int(self.config.counting_line_position * self._frame_width)
            cv2.line(
                display_frame,
                (line_x, 0),
                (line_x, self._frame_height),
                (0, 0, 255),  # Red line
                2
            )
            line_label = f"Counting Line (x={self.config.counting_line_position:.2f})"
            cv2.putText(display_frame, line_label, (line_x + 10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            # Horizontal line (for vertical wagon movement)
            line_y = int(self.config.counting_line_position * self._frame_height)
            cv2.line(
                display_frame,
                (0, line_y),
                (self._frame_width, line_y),
                (0, 0, 255),  # Red line
                2
            )
            line_label = f"Counting Line (y={self.config.counting_line_position:.2f})"
            cv2.putText(display_frame, line_label, (10, line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw raw detections first (cyan boxes) - these are direct from YOLO
        raw_det_count = 0
        if raw_detections:
            for det in raw_detections:
                try:
                    if hasattr(det, 'bbox'):
                        x1, y1 = int(det.bbox.x1), int(det.bbox.y1)
                        x2, y2 = int(det.bbox.x2), int(det.bbox.y2)
                        conf = det.confidence if hasattr(det, 'confidence') else 0.0
                    else:
                        continue
                    
                    # Draw cyan box for raw detection
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 255, 0), 1)  # Cyan
                    cv2.putText(display_frame, f"{conf:.2f}", (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                    raw_det_count += 1
                except:
                    pass
        
        # Draw tracked wagon bounding boxes and labels (thicker boxes)
        for wagon in tracked_wagons:
            # Green if crossed line, yellow if not
            color = (0, 255, 0) if wagon.crossed_line else (0, 255, 255)
            
            # Draw bounding box
            x1, y1, x2, y2 = int(wagon.bbox.x1), int(wagon.bbox.y1), int(wagon.bbox.x2), int(wagon.bbox.y2)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 3)
            
            # Create label
            label = f"ID:{wagon.track_id}"
            if wagon.count_index:
                label += f" #{wagon.count_index}"
            
            # Draw label background
            (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(display_frame, (x1, y1 - label_h - 10), (x1 + label_w + 5, y1), color, -1)
            
            # Draw label text
            cv2.putText(display_frame, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Draw info overlay
        wagon_count = self._tracker.get_wagon_count() if self._tracker else 0
        tracked_count = len(tracked_wagons)
        
        # Show track IDs for debugging
        track_ids = [w.track_id for w in tracked_wagons] if tracked_wagons else []
        
        info_text = f"Frame: {frame_index} | Counted: {wagon_count} | Tracked: {tracked_count} | Raw: {raw_det_count}"
        cv2.putText(display_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        # Show track IDs
        if track_ids:
            ids_text = f"IDs: {track_ids[:5]}"  # Show first 5 IDs
            cv2.putText(display_frame, ids_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Draw FPS info
        fps_text = f"FPS: {self._fps:.1f}"
        cv2.putText(display_frame, fps_text, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return display_frame

    def run(self) -> None:
        """Run the main pipeline loop.
        
        Captures frames from video source and processes them through
        all pipeline stages. Handles graceful shutdown on interrupt.
        
        Requirements: 1.1, 9.6, 9.7
        """
        print(f"Starting Railway Wagon Inspection Pipeline...")
        print(f"Video source: {self.config.video_source}")
        if self._display:
            print("Display mode: ON (press 'q' to quit)")
        
        try:
            # Initialize components
            self._init_components()
            
            # Open video source
            self._open_video_source()
            print(f"Video opened: {self._frame_width}x{self._frame_height} @ {self._fps:.1f} FPS")
            
            # Start GPU memory monitoring
            self.memory_manager.start_monitoring()
            
            # Initialize thread pool if threading enabled
            if self.config.enable_threading:
                self._executor = ThreadPoolExecutor(max_workers=2)
            
            self._running = True
            self._frame_index = 0
            
            # Frame timing for rate control
            frame_time = 1.0 / self._fps
            last_frame_time = time.time()
            
            # Pending futures for async processing
            pending_futures: List[Future] = []
            
            # Create display window if display mode is enabled
            if self._display:
                cv2.namedWindow("Railway Wagon Inspection", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Railway Wagon Inspection", 1280, 720)
            
            while self._running and not self._shutdown_event.is_set():
                # Read frame
                success, frame = self._read_frame()
                
                if not success:
                    # End of video or read error
                    if self._video_capture.get(cv2.CAP_PROP_POS_FRAMES) >= \
                       self._video_capture.get(cv2.CAP_PROP_FRAME_COUNT) - 1:
                        print("End of video reached.")
                    break
                
                # Check GPU memory and adjust if needed
                if self.memory_manager.is_memory_pressure():
                    self.memory_manager.clear_cache()
                    # Clear deblur manager cache if memory pressure
                    if self._deblur_manager:
                        self._deblur_manager.clear_all_caches()
                
                # Process frame (synchronous for now - NAFNet needs sequential processing)
                records, tracked_wagons, raw_detections = self._process_frame(frame, self._frame_index)
                
                # Display frame if display mode is enabled
                if self._display:
                    display_frame = self._draw_display_frame(frame, self._frame_index, tracked_wagons, raw_detections)
                    cv2.imshow("Railway Wagon Inspection", display_frame)
                    
                    # Check for quit key
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:  # 'q' or ESC
                        print("\nDisplay closed by user.")
                        break
                
                self._frame_index += 1
                
                # Frame rate control (skip if display mode to allow real-time viewing)
                if not self._display:
                    elapsed = time.time() - last_frame_time
                    if elapsed < frame_time:
                        time.sleep(frame_time - elapsed)
                last_frame_time = time.time()
                
                # Progress update every 100 frames
                if self._frame_index % 100 == 0:
                    wagon_count = self._tracker.get_wagon_count()
                    mem_summary = self.memory_manager.get_memory_summary()
                    cache_size = self._deblur_manager.get_cache_size() if self._deblur_manager else 0
                    print(f"Processed {self._frame_index} frames, {wagon_count} wagons | Cache: {cache_size} | {mem_summary}")
            
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        except Exception as e:
            print(f"Pipeline error: {e}")
            raise
        finally:
            # Close display window if it was opened
            if self._display:
                cv2.destroyAllWindows()
            self.shutdown()

    def shutdown(self) -> None:
        """Gracefully shutdown the pipeline and release resources."""
        print("Shutting down pipeline...")
        
        self._running = False
        self._shutdown_event.set()
        
        # Stop GPU memory monitoring
        self.memory_manager.stop_monitoring()
        
        # Close video capture
        if self._video_capture is not None:
            self._video_capture.release()
            self._video_capture = None
        
        # Shutdown thread pool
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        
        # Clear deblur manager caches
        if self._deblur_manager is not None:
            self._deblur_manager.clear_all_caches()
        
        # Flush and close logger
        if self._logger is not None:
            self._logger.close()
        
        # Clear GPU memory
        self.memory_manager.clear_cache()
        
        # Print summary
        if self._tracker is not None:
            wagon_count = self._tracker.get_wagon_count()
            print(f"Pipeline complete. Total wagons counted: {wagon_count}")
            print(f"Total frames processed: {self._frame_index}")
            print(self.memory_manager.get_memory_summary())
    
    def get_wagon_count(self) -> int:
        """Get the current wagon count.
        
        Returns:
            Number of wagons that have crossed the counting line
        """
        if self._tracker is None:
            return 0
        return self._tracker.get_wagon_count()
    
    def is_running(self) -> bool:
        """Check if the pipeline is currently running.
        
        Returns:
            True if pipeline is running, False otherwise
        """
        return self._running


def setup_signal_handlers(pipeline: RailwayWagonPipeline) -> None:
    """Setup signal handlers for graceful shutdown.
    
    Args:
        pipeline: Pipeline instance to shutdown on signal
    """
    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}, initiating shutdown...")
        pipeline.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def main(config_path: Optional[str] = None) -> None:
    """Main entry point for the pipeline.
    
    Args:
        config_path: Optional path to configuration JSON file.
                    If None, uses default configuration.
    """
    try:
        # Load configuration
        if config_path:
            print(f"Loading configuration from: {config_path}")
            config = load_config(config_path)
        else:
            print("Using default configuration")
            config = load_config()
        
        # Create and run pipeline
        pipeline = RailwayWagonPipeline(config)
        
        # Setup signal handlers
        setup_signal_handlers(pipeline)
        
        # Run the pipeline
        pipeline.run()
        
    except ConfigurationError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)
    except ModelNotFoundError as e:
        print(f"Model not found: {e}")
        print("Please ensure all required model files are present.")
        sys.exit(1)
    except ModelLoadError as e:
        print(f"Model load error: {e}")
        sys.exit(1)
    except VideoSourceError as e:
        print(f"Video source error: {e}")
        sys.exit(1)
    except StreamConnectionError as e:
        print(f"Stream connection error: {e}")
        sys.exit(1)
    except GPUMemoryError as e:
        print(f"GPU memory error: {e}")
        print("Try reducing batch size or closing other GPU applications.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Railway Wagon Inspection Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration JSON file"
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to video file or stream URL (overrides config)"
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show video output window with annotations"
    )
    parser.add_argument(
        "--line",
        type=float,
        default=None,
        help="Counting line position (0.0-1.0, fraction of frame width/height). Default: 0.5"
    )
    parser.add_argument(
        "--orientation",
        type=str,
        choices=["vertical", "horizontal"],
        default=None,
        help="Counting line orientation: 'vertical' for left-right wagon movement, 'horizontal' for up-down movement. Default: vertical"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Handle video override
    if args.video:
        config.video_source = args.video
    
    # Handle counting line override
    if args.line is not None:
        if 0.0 <= args.line <= 1.0:
            config.counting_line_position = args.line
        else:
            print(f"Error: --line must be between 0.0 and 1.0, got {args.line}")
            sys.exit(1)
    
    # Handle orientation override
    if args.orientation is not None:
        config.counting_line_orientation = args.orientation
    
    try:
        pipeline = RailwayWagonPipeline(config, display=args.display)
        setup_signal_handlers(pipeline)
        pipeline.run()
    except ConfigurationError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)
    except ModelNotFoundError as e:
        print(f"Model not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
