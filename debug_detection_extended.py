"""Extended debug script to check wagon detection across entire video."""

import cv2
import numpy as np
from ultralytics import YOLO

model_path = "models/wagon_detector.pt"
video_path = "5.mp4"

print(f"Loading model from: {model_path}")
model = YOLO(model_path)
print(f"Model classes: {model.names}")

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Failed to open video: {video_path}")
    exit(1)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total frames: {total_frames}")

# Sample every 500 frames across the entire video
sample_frames = list(range(0, total_frames, 500))
print(f"Testing {len(sample_frames)} frames...")

detection_count = 0
frames_with_detections = []

for frame_idx in sample_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        continue
    
    # Try very low confidence
    results = model(frame, conf=0.05, verbose=False)
    
    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            detection_count += len(result.boxes)
            frames_with_detections.append(frame_idx)
            print(f"Frame {frame_idx}: {len(result.boxes)} detections")
            for box in result.boxes:
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = model.names.get(cls_id, f"class_{cls_id}")
                conf = float(box.conf[0].cpu().numpy())
                print(f"  - {cls_name}: {conf:.3f}")

cap.release()

print(f"\n=== Summary ===")
print(f"Total detections: {detection_count}")
print(f"Frames with detections: {len(frames_with_detections)}")
if frames_with_detections:
    print(f"Detection frames: {frames_with_detections[:20]}...")  # Show first 20
