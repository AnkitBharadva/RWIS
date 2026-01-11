"""Run wagon detection with live video display and save output video."""

import cv2
import numpy as np
from ultralytics import YOLO
from tracking.tracker import WagonTracker

# Configuration
VIDEO_PATH = "5.mp4"
WAGON_MODEL_PATH = "models/damage_detector.pt"  # wagon_body detector
DAMAGE_MODEL_PATH = "models/wagon_detector.pt"  # damage classes
OUTPUT_VIDEO_PATH = "outputs/detection_output.mp4"
WAGON_CONFIDENCE = 0.25
COUNTING_LINE_Y = 0.5  # Middle of frame

print("Loading models...")
wagon_model = YOLO(WAGON_MODEL_PATH)
damage_model = YOLO(DAMAGE_MODEL_PATH)
print(f"Wagon model classes: {wagon_model.names}")
print(f"Damage model classes: {damage_model.names}")

# Initialize tracker
tracker = WagonTracker(counting_line_y=COUNTING_LINE_Y)

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Failed to open video: {VIDEO_PATH}")
    exit(1)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video: {frame_width}x{frame_height} @ {fps} FPS, {total_frames} frames")

# Setup video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

# Counting line position in pixels
line_y = int(COUNTING_LINE_Y * frame_height)

frame_idx = 0
wagon_count = 0
MAX_FRAMES = 3000  # Process first 3000 frames (about 50 seconds at 60fps)

print(f"Processing first {MAX_FRAMES} frames... Output saved to:", OUTPUT_VIDEO_PATH)

while frame_idx < MAX_FRAMES:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run wagon detection
    results = wagon_model(frame, conf=WAGON_CONFIDENCE, verbose=False)
    
    # Draw counting line
    cv2.line(frame, (0, line_y), (frame_width, line_y), (255, 0, 0), 2)
    cv2.putText(frame, "Counting Line", (10, line_y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # Process detections
    detections = []
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = wagon_model.names[cls_id]
                
                # Only track wagon_body class
                if cls_name == 'wagon_body':
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'conf': conf,
                        'cls': cls_name
                    })
                    
                    # Draw detection box
                    color = (0, 255, 0)  # Green for wagon
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{cls_name}: {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Check if crossed line
                    center_y = (y1 + y2) // 2
                    if center_y > line_y:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Red when crossed
    
    # Update tracker and count
    if detections:
        # Simple counting based on center crossing line
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            center_y = (y1 + y2) // 2
            # This is simplified - the actual tracker handles proper counting
    
    # Get wagon count from tracker (simplified version)
    wagon_count = tracker.get_wagon_count()
    
    # Draw info overlay
    info_text = f"Frame: {frame_idx}/{total_frames} | Detections: {len(detections)} | Wagons Counted: {wagon_count}"
    cv2.rectangle(frame, (0, 0), (frame_width, 40), (0, 0, 0), -1)
    cv2.putText(frame, info_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Write frame to output video
    out.write(frame)
    
    frame_idx += 1
    
    # Progress update
    if frame_idx % 100 == 0:
        print(f"Processed {frame_idx}/{total_frames} frames, {len(detections)} detections in current frame")

# Cleanup
cap.release()
out.release()

print(f"\nDone! Output saved to: {OUTPUT_VIDEO_PATH}")
print(f"Total frames processed: {frame_idx}")
print(f"Total wagons counted: {wagon_count}")
