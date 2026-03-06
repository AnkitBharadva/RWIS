"""
Benchmark script to measure average performance metrics.
Run this to get FPS, latency, and resource usage statistics.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU mode

import cv2
import time
import numpy as np
import psutil
from datetime import datetime
from typing import List, Dict
import json

from config import PipelineConfig, load_config
from pipelines.wagon_detector import WagonDetector
from pipelines.damage_detector import DamageDetector
from pipelines.ocr_pipeline import OCRPipeline
from pipelines.blur_detector import BlurDetector
from pipelines.mprnet_wrapper import MPRNetDeblur
from pipelines.deblur_manager import DeblurManager
from utils.clahe import CLAHEEnhancer


class PerformanceBenchmark:
    """Benchmark the pipeline performance."""
    
    def __init__(self, video_path: str = "1.ts", num_frames: int = None):
        """Initialize benchmark.
        
        Args:
            video_path: Path to test video
            num_frames: Number of frames to process for benchmark (None = entire video)
        """
        self.video_path = video_path
        self.num_frames = num_frames
        self.metrics = {
            'fps': [],
            'latency_ms': [],
            'cpu_percent': [],
            'memory_mb': [],
            'detection_time_ms': [],
            'ocr_time_ms': [],
            'deblur_time_ms': []
        }
        
    def run_benchmark(self) -> Dict:
        """Run the benchmark and collect metrics."""
        print("=" * 70)
        print("Railway Wagon Inspection System - Performance Benchmark")
        print("=" * 70)
        print(f"\nVideo: {self.video_path}")
        
        # Open video to get total frame count
        cap_temp = cv2.VideoCapture(self.video_path)
        if not cap_temp.isOpened():
            print(f"\n✗ Failed to open video: {self.video_path}")
            return None
        
        total_video_frames = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_temp.release()
        
        if self.num_frames is None:
            self.num_frames = total_video_frames
            print(f"Frames to process: ALL ({total_video_frames} frames)")
        else:
            print(f"Frames to process: {self.num_frames} of {total_video_frames}")
        
        print(f"Mode: CPU (sm_120 compatibility)")
        print("\nInitializing components...")
        
        # Initialize components
        try:
            wagon_detector = WagonDetector(
                model_path="models/damage_detector.pt",
                confidence_threshold=0.25
            )
            print("✓ Wagon detector loaded")
        except Exception as e:
            print(f"✗ Wagon detector failed: {e}")
            wagon_detector = None
        
        try:
            damage_detector = DamageDetector(
                model_path="models/wagon_detector.pt",
                confidence_threshold=0.5
            )
            print("✓ Damage detector loaded")
        except Exception as e:
            print(f"✗ Damage detector failed: {e}")
            damage_detector = None
        
        try:
            ocr_pipeline = OCRPipeline(gpu_enabled=False, language='en')
            if ocr_pipeline.is_available():
                print("✓ OCR pipeline loaded")
            else:
                print(f"✗ OCR not available: {ocr_pipeline.get_initialization_error()}")
                ocr_pipeline = None
        except Exception as e:
            print(f"✗ OCR pipeline failed: {e}")
            ocr_pipeline = None
        
        try:
            blur_detector = BlurDetector(t1=50.0, t2=100.0)
            print("✓ Blur detector loaded")
        except Exception as e:
            print(f"✗ Blur detector failed: {e}")
            blur_detector = None
        
        try:
            mprnet = MPRNetDeblur(
                model_path="MPRNet/Deblurring/pretrained_models/model_deblurring.pth",
                device='cpu',
                use_fp16=False,
                max_roi_width=256
            )
            mprnet.load_model()
            deblur_manager = DeblurManager(
                mprnet=mprnet,
                blur_detector=blur_detector,
                frame_interval=3,
                max_roi_width=256
            )
            print("✓ Deblur manager loaded")
        except Exception as e:
            print(f"✗ Deblur manager failed: {e}")
            deblur_manager = None
        
        clahe = CLAHEEnhancer()
        print("✓ CLAHE enhancer loaded")
        
        # Open video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"\n✗ Failed to open video: {self.video_path}")
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"\n✓ Video opened: {width}x{height} @ {fps:.1f} FPS")
        
        print("\n" + "=" * 70)
        print("Starting benchmark...")
        print("=" * 70)
        
        process = psutil.Process()
        frame_count = 0
        start_time = time.time()
        
        while frame_count < self.num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_start = time.time()
            
            # Measure CPU and memory
            cpu_percent = process.cpu_percent()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            # Stage 1: CLAHE
            enhanced_frame = clahe.enhance(frame)
            
            # Stage 2: Wagon Detection
            detection_start = time.time()
            detections = []
            if wagon_detector:
                detections = wagon_detector.detect(enhanced_frame)
            detection_time = (time.time() - detection_start) * 1000
            
            # Stage 3: Process ROIs
            ocr_time = 0
            deblur_time = 0
            
            if detections and len(detections) > 0:
                # Take first detection for benchmark
                wagon = detections[0]
                x1, y1, x2, y2 = int(wagon.bbox.x1), int(wagon.bbox.y1), int(wagon.bbox.x2), int(wagon.bbox.y2)
                
                # Clip to frame boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                
                roi = frame[y1:y2, x1:x2]
                
                if roi.size > 0:
                    # Resize ROI to max 256x256 for deblurring
                    h, w = roi.shape[:2]
                    if w > 256 or h > 256:
                        scale = min(256 / w, 256 / h)
                        new_w = int(w * scale)
                        new_h = int(h * scale)
                        roi_resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    else:
                        roi_resized = roi
                    
                    # Deblur
                    if deblur_manager:
                        deblur_start = time.time()
                        processed_roi, _, _ = deblur_manager.process_roi(roi_resized, wagon_id=0, frame_index=frame_count)
                        deblur_time = (time.time() - deblur_start) * 1000
                    else:
                        processed_roi = roi_resized
                    
                    # OCR (every 5th frame)
                    if ocr_pipeline and frame_count % 5 == 0:
                        ocr_start = time.time()
                        ocr_result = ocr_pipeline.extract_text(processed_roi, min_confidence=0.3)
                        ocr_time = (time.time() - ocr_start) * 1000
            
            # Calculate frame metrics
            frame_time = time.time() - frame_start
            frame_fps = 1.0 / frame_time if frame_time > 0 else 0
            latency_ms = frame_time * 1000
            
            # Store metrics
            self.metrics['fps'].append(frame_fps)
            self.metrics['latency_ms'].append(latency_ms)
            self.metrics['cpu_percent'].append(cpu_percent)
            self.metrics['memory_mb'].append(memory_mb)
            self.metrics['detection_time_ms'].append(detection_time)
            self.metrics['ocr_time_ms'].append(ocr_time)
            self.metrics['deblur_time_ms'].append(deblur_time)
            
            frame_count += 1
            
            # Progress update
            if frame_count % 10 == 0:
                print(f"Processed {frame_count}/{self.num_frames} frames | "
                      f"FPS: {frame_fps:.1f} | Latency: {latency_ms:.1f}ms | "
                      f"CPU: {cpu_percent:.1f}% | RAM: {memory_mb:.0f}MB")
        
        cap.release()
        
        total_time = time.time() - start_time
        
        # Calculate averages
        results = {
            'system_info': {
                'cpu': psutil.cpu_count(),
                'ram_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
                'gpu': 'NVIDIA RTX 5050 (CPU mode - sm_120)',
                'video_resolution': f"{width}x{height}",
                'video_fps': fps
            },
            'performance': {
                'avg_fps': np.mean(self.metrics['fps']),
                'min_fps': np.min(self.metrics['fps']),
                'max_fps': np.max(self.metrics['fps']),
                'avg_latency_ms': np.mean(self.metrics['latency_ms']),
                'avg_cpu_percent': np.mean(self.metrics['cpu_percent']),
                'avg_memory_mb': np.mean(self.metrics['memory_mb']),
                'avg_detection_time_ms': np.mean(self.metrics['detection_time_ms']),
                'avg_ocr_time_ms': np.mean([x for x in self.metrics['ocr_time_ms'] if x > 0]) if any(self.metrics['ocr_time_ms']) else 0,
                'avg_deblur_time_ms': np.mean([x for x in self.metrics['deblur_time_ms'] if x > 0]) if any(self.metrics['deblur_time_ms']) else 0
            },
            'summary': {
                'total_frames': frame_count,
                'total_time_seconds': total_time,
                'overall_fps': frame_count / total_time
            }
        }
        
        return results
    
    def print_results(self, results: Dict):
        """Print benchmark results."""
        print("\n" + "=" * 70)
        print("BENCHMARK RESULTS")
        print("=" * 70)
        
        print("\nSystem Information:")
        print(f"  CPU Cores: {results['system_info']['cpu']}")
        print(f"  RAM: {results['system_info']['ram_gb']:.1f} GB")
        print(f"  GPU: {results['system_info']['gpu']}")
        print(f"  Video: {results['system_info']['video_resolution']} @ {results['system_info']['video_fps']:.1f} FPS")
        
        print("\nPerformance Metrics:")
        print(f"  Average FPS: {results['performance']['avg_fps']:.2f}")
        print(f"  Min FPS: {results['performance']['min_fps']:.2f}")
        print(f"  Max FPS: {results['performance']['max_fps']:.2f}")
        print(f"  Average Latency: {results['performance']['avg_latency_ms']:.2f} ms")
        print(f"  Average CPU Usage: {results['performance']['avg_cpu_percent']:.1f}%")
        print(f"  Average Memory: {results['performance']['avg_memory_mb']:.0f} MB")
        
        print("\nComponent Timings:")
        print(f"  Detection: {results['performance']['avg_detection_time_ms']:.2f} ms")
        print(f"  Deblurring: {results['performance']['avg_deblur_time_ms']:.2f} ms")
        print(f"  OCR: {results['performance']['avg_ocr_time_ms']:.2f} ms")
        
        print("\nSummary:")
        print(f"  Total Frames: {results['summary']['total_frames']}")
        print(f"  Total Time: {results['summary']['total_time_seconds']:.2f} seconds")
        print(f"  Overall FPS: {results['summary']['overall_fps']:.2f}")
        
        print("\n" + "=" * 70)
    
    def save_results(self, results: Dict, filename: str = "benchmark_results.json"):
        """Save results to JSON file."""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {filename}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark Railway Wagon Inspection System")
    parser.add_argument("--video", type=str, default="1.ts", help="Path to test video")
    parser.add_argument("--frames", type=int, default=None, help="Number of frames to process (default: entire video)")
    
    args = parser.parse_args()
    
    benchmark = PerformanceBenchmark(video_path=args.video, num_frames=args.frames)
    results = benchmark.run_benchmark()
    
    if results:
        benchmark.print_results(results)
        benchmark.save_results(results)
        
        print("\nTo add these metrics to README.md, copy the performance section above.")
    else:
        print("\n✗ Benchmark failed")
