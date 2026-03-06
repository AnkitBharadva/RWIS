# Model Training Details

This document provides comprehensive details about the training of both YOLO models used in the Railway Wagon Inspection System.

---

## Overview

Two YOLOv11n (nano) models were trained for this project:
1. **Wagon Detector** - Detects wagon bodies and wheels
2. **Damage Detector** - Detects various types of damage on wagons

Both models were trained using transfer learning from YOLOv11n pretrained weights.

---

## 1. Wagon Detector

### Model Architecture
- **Base Model:** YOLOv11n (Ultralytics)
- **Input Size:** 640x640
- **Classes:** 2
  - `wagon_body` - Main body of the wagon
  - `wheel` - Wagon wheels

### Training Configuration

```yaml
# From training/Wagon_detector/args.yaml
task: detect
mode: train
model: yolo11n.pt
data: wagon_dataset.yaml
epochs: 150
batch: 16
imgsz: 640
device: 0 (GPU)
optimizer: AdamW
lr0: 0.01
momentum: 0.937
weight_decay: 0.0005
```

### Training Results

**Final Metrics (Epoch 49):**
- **Precision:** 63.84%
- **Recall:** 73.81%
- **mAP@50:** 70.98%
- **mAP@50-95:** 43.35%

**Training Progress:**
- Started with mAP@50 of 32.07% (epoch 1)
- Reached peak mAP@50 of 72.50% (epoch 30)
- Final stable performance around 71% mAP@50

**Loss Curves:**
- Box Loss: 1.168 → 0.695 (40.5% reduction)
- Classification Loss: 2.536 → 0.651 (74.3% reduction)
- DFL Loss: 1.451 → 1.087 (25.1% reduction)

### Dataset Statistics
- **Training Images:** ~4,230 images
- **Validation Split:** ~20%
- **Augmentations:** 
  - Random scaling
  - Horizontal flip
  - Mosaic augmentation
  - Color jittering

### Training Artifacts
Located in `training/Wagon_detector/`:
- `results.csv` - Complete training metrics
- `results.png` - Training curves visualization
- `confusion_matrix.png` - Class confusion matrix
- `BoxPR_curve.png` - Precision-Recall curve
- `BoxF1_curve.png` - F1 score curve
- `train_batch*.jpg` - Sample training batches
- `val_batch*_pred.jpg` - Validation predictions

---

## 2. Damage Detector

### Model Architecture
- **Base Model:** YOLOv11n (Ultralytics)
- **Input Size:** 640x640
- **Classes:** 7
  - `Bamboo Door` - Bamboo door type
  - `Breakage` - Broken parts or components
  - `Close Door` - Closed door state
  - `Damage Door` - Damaged door
  - `Dent` - Dents on wagon surface
  - `Open Door` - Open door state
  - `Wagon` - General wagon detection

### Training Configuration

```yaml
# From training/Damage_Detector/args.yaml
task: detect
mode: train
model: yolo11n.pt
data: damage_dataset.yaml
epochs: 150
batch: 16
imgsz: 640
device: 0 (GPU)
optimizer: AdamW
lr0: 0.01
momentum: 0.937
weight_decay: 0.0005
```

### Training Results

**Final Metrics (Epoch 150):**
- **Precision:** 71.79%
- **Recall:** 69.05%
- **mAP@50:** 71.78%
- **mAP@50-95:** 55.60%

**Training Progress:**
- Started with mAP@50 of 29.92% (epoch 1)
- Reached peak mAP@50 of 87.49% (epoch 61)
- Final stable performance around 72% mAP@50

**Loss Curves:**
- Box Loss: 1.422 → 0.588 (58.6% reduction)
- Classification Loss: 1.766 → 0.381 (78.4% reduction)
- DFL Loss: 1.535 → 1.015 (33.9% reduction)

**Best Performance (Epoch 50):**
- Precision: 77.63%
- Recall: 82.80%
- mAP@50: 87.24%
- mAP@50-95: 57.98%

### Dataset Statistics
- **Training Images:** ~29,820 images
- **Validation Split:** ~20%
- **Augmentations:**
  - Random scaling
  - Horizontal flip
  - Mosaic augmentation
  - Color jittering
  - Random rotation (±10°)

### Training Artifacts
Located in `training/Damage_Detector/`:
- `results.csv` - Complete training metrics (150 epochs)
- `results.png` - Training curves visualization
- `confusion_matrix.png` - Class confusion matrix
- `BoxPR_curve.png` - Precision-Recall curve
- `BoxF1_curve.png` - F1 score curve
- `train_batch*.jpg` - Sample training batches
- `val_batch*_pred.jpg` - Validation predictions

---

## Training Notebooks

### Wagon Detector Training
**File:** `training/wagon-detector.ipynb`

Contains:
- Data preparation and augmentation
- Model initialization and configuration
- Training loop with validation
- Results visualization
- Model export

### Damage Detector Training
**File:** `training/damage-detection.ipynb`

Contains:
- Data preparation and augmentation
- Model initialization and configuration
- Training loop with validation
- Results visualization
- Model export

---

## Hardware & Environment

### Training Hardware
- **GPU:** NVIDIA GPU (CUDA-enabled)
- **Framework:** PyTorch 2.x
- **Ultralytics:** YOLOv11 (latest)

### Training Time
- **Wagon Detector:** ~1,048 seconds (~17.5 minutes for 49 epochs)
- **Damage Detector:** ~5,781 seconds (~96.4 minutes for 150 epochs)

---

## Model Performance Comparison

| Metric | Wagon Detector | Damage Detector |
|--------|----------------|-----------------|
| Precision | 63.84% | 71.79% |
| Recall | 73.81% | 69.05% |
| mAP@50 | 70.98% | 71.78% |
| mAP@50-95 | 43.35% | 55.60% |
| Training Epochs | 49 | 150 |
| Dataset Size | ~4,230 | ~29,820 |
| Classes | 2 (wagon_body, wheel) | 7 (Bamboo Door, Breakage, Close Door, Damage Door, Dent, Open Door, Wagon) |

---

## Model Deployment

### Exported Models
Both models are exported in PyTorch format (`.pt`) and located in:
- `models/damage_detector.pt` - Actually contains wagon detection (wagon_body, wheel)
- `models/wagon_detector.pt` - Actually contains damage detection classes

**Note:** The model names are swapped in the deployment. This is intentional based on the class mappings.

### Inference Configuration
- **Input Size:** 640x640 (resized from 1280x720 video)
- **Confidence Threshold:** 0.35 (adjustable in dashboard)
- **NMS IoU Threshold:** 0.45
- **Device:** CPU (RTX 5050 not compatible with PyTorch yet)

---

## Training Insights

### Wagon Detector
✅ **Strengths:**
- High recall (73.81%) - catches most wagons
- Good generalization across different wagon types
- Fast inference (~33ms on CPU)

⚠️ **Areas for Improvement:**
- Precision could be higher (63.84%)
- Some false positives on similar objects
- Wheel detection less accurate than wagon body

### Damage Detector
✅ **Strengths:**
- Excellent peak performance (87.24% mAP@50)
- High precision (71.79%) - few false positives
- Good at detecting multiple damage types
- Large diverse dataset (~30k images)

⚠️ **Areas for Improvement:**
- Performance dropped from peak (epoch 50 to 150)
- Could benefit from early stopping
- Some damage types harder to detect than others

---

## Future Improvements

### Short Term
1. **Early Stopping:** Implement early stopping to prevent overfitting
2. **Class Balancing:** Balance damage classes in training data
3. **Hyperparameter Tuning:** Optimize learning rate schedule
4. **Data Augmentation:** Add more realistic augmentations

### Long Term
1. **Larger Model:** Try YOLOv11s or YOLOv11m for better accuracy
2. **Ensemble:** Combine multiple models for better performance
3. **Active Learning:** Continuously improve with production data
4. **GPU Support:** Wait for PyTorch sm_120 support for RTX 5050

---

## Reproducing Training

### Prerequisites
```bash
pip install ultralytics torch torchvision opencv-python
```

### Run Training
```bash
# Wagon Detector
jupyter notebook training/wagon-detector.ipynb

# Damage Detector
jupyter notebook training/damage-detection.ipynb
```

### Monitor Training
```bash
# View results
tensorboard --logdir=runs/detect/train
```

---

## Model Kaggle Links

The trained models are available on Kaggle:

1. **Wagon Detector:** https://www.kaggle.com/models/ankitbharadva/wagon-detection
2. **Damage Detector:** https://www.kaggle.com/models/ankitbharadva/wagon-damage-detection
3. **MPRNet (Deblurring):** https://www.kaggle.com/models/ankitbharadva/nprnet

---

## References

- **YOLOv11:** [Ultralytics Documentation](https://docs.ultralytics.com/)
- **Training Guide:** [YOLO Training Tutorial](https://docs.ultralytics.com/modes/train/)
- **Model Zoo:** [Ultralytics Model Zoo](https://github.com/ultralytics/ultralytics)

---

## Contact

For questions about model training or to request the training datasets:
- Check the Kaggle model pages for dataset links
- Review the training notebooks for detailed implementation

---

**Last Updated:** March 2026
**Models Version:** YOLOv11n
**Framework:** Ultralytics YOLO
