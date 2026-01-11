"""Debug script to diagnose wagon detection issues."""

import cv2
import numpy as np
from ultralytics import YOLO

# Load the wagon detector model
model_path = "models/wagon_detector.pt"
video_path = "5.mp4"  # Change to your video file

print(f"Loading model from: {model_path}")
model = YOLO(model_path)

# Print model info
print(f"\nModel classes: {model.names}")
print(f"Number of classes: {len(model.names)}")

# Open video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Failed to open video: {video_path}")
    exit(1)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"\nVideo info: {frame_width}x{frame_height} @ {fps} FPS, {total_frames} frames")

# Test detection on first few frames with different confidence thresholds
test_frames = [0, 100, 200, 500, 1000]
confidence_thresholds = [0.1, 0.25, 0.5]

for frame_idx in test_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to read frame {frame_idx}")
        continue
    
    print(f"\n--- Frame {frame_idx} ---")
    
    for conf in confidence_thresholds:
        results = model(frame, conf=conf, verbose=False)
        
        total_detections = 0
        for result in results:
            if result.boxes is not None:
                total_detections += len(result.boxes)
                
                # Print detection details
                for i, box in enumerate(result.boxes):
                    cls_id = int(box.cls[0].cpu().numpy())
                    cls_name = model.names.get(cls_id, f"class_{cls_id}")
                    box_conf = float(box.conf[0].cpu().numpy())
                    xyxy = box.xyxy[0].cpu().numpy()
                    print(f"  conf={conf}: {cls_name} ({box_conf:.2f}) at [{xyxy[0]:.0f}, {xyxy[1]:.0f}, {xyxy[2]:.0f}, {xyxy[3]:.0f}]")
        
        if total_detections == 0:
            print(f"  conf={conf}: No detections")

cap.release()
print("\nDone!")
