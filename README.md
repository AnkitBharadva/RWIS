# Railway Wagon Inspection System

An end-to-end AI-powered video processing pipeline for railway wagon inspection, featuring real-time wagon counting, damage detection, and OCR-based wagon identification.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Features

- **Real-time Wagon Detection** - YOLOv11n-based detection with ByteTrack multi-object tracking
- **Damage Detection** - Automated detection of wagon damage (dents, breakage, door issues)
- **OCR Text Extraction** - EasyOCR-powered wagon identification with visual feedback
- **Smart Deblurring** - MPRNet ROI-only deblurring for improved OCR accuracy
- **Low-Light Enhancement** - CLAHE and gamma correction for night-time footage
- **Mission Control Dashboard** - Streamlit-based real-time monitoring interface
- **Processing Status Indicators** - Visual feedback for Illumination, Deblur, and OCR status
- **Automatic Frame Saving** - Save OCR frames with JSON metadata

## Quick Start

### Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA support (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/railway-wagon-inspection.git
cd railway-wagon-inspection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# For GPU support (when sm_120 is supported by PyTorch)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Model Downloads

The system requires three custom-trained models for operation. All models must be downloaded and placed in their respective directories before running the pipeline.

| Model | Purpose | Classes | Size | Download |
|-------|---------|---------|------|----------|
| **Wagon Detector** | Detects railway wagons in video frames | `wagon_body`, `wheel` | ~6 MB | [Kaggle](https://www.kaggle.com/models/ankitbharadva/wagon-detection) |
| **Damage Detector** | Identifies damage types on wagon surfaces | `Bamboo Door`, `Breakage`, `Close Door`, `Damage Door`, `Dent`, `Open Door`, `Wagon` | ~6 MB | [Kaggle](https://www.kaggle.com/models/ankitbharadva/wagon-damage-detection) |
| **MPRNet Deblur** | Removes motion blur from ROI regions | N/A (Image restoration) | ~20 MB | [Kaggle](https://www.kaggle.com/models/ankitbharadva/nprnet) |

**Installation Instructions:**

```bash
# Create required directories
mkdir -p models
mkdir -p MPRNet/Deblurring/pretrained_models

# Download and place models
# 1. Wagon Detector → models/damage_detector.pt
# 2. Damage Detector → models/wagon_detector.pt
# 3. MPRNet Deblur → MPRNet/Deblurring/pretrained_models/model_deblurring.pth
```

**Verification:**

```bash
# Verify all models are in place
ls models/damage_detector.pt
ls models/wagon_detector.pt
ls MPRNet/Deblurring/pretrained_models/model_deblurring.pth
```

### Running the Dashboard

```bash
# Start the Mission Control Dashboard (local access only)
streamlit run dashboard/app.py

# Or use the launcher script
python run_dashboard.py

# Enable network access (access from phone/tablet on same WiFi)
streamlit run dashboard/app.py --server.address 0.0.0.0 --server.port 8501
```

**Network Access:**
1. Find your computer's IP address: `ipconfig` (Windows) or `ifconfig` (Linux/Mac)
2. Look for IPv4 Address (e.g., `192.168.1.100`)
3. Access from any device on the same network: `http://192.168.1.100:8501`
4. If blocked by firewall, allow Python through Windows Defender Firewall

### Running the Pipeline

```bash
# Run with video file
python main.py --video path/to/video.mp4 --display

# Run with RTSP stream
python main.py --video rtsp://192.168.1.100:554/stream

# Run with configuration file
python main.py --config config.json
```

## Architecture

### System Overview

```
┌────────────────────────────────────────────────────────────────────────────┐
│                           Mission Control Dashboard                        │
│                        (Streamlit WebSocket Server)                        │
├────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Enhanced Metrics Row                             │   │
│  │  FPS | Latency | Objects | Wagons | Damage | Illum | Deblur | OCR   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Dual Video Display                               │   │
│  │  ┌─────────────────────┐    ┌─────────────────────┐                 │   │
│  │  │    Raw Input        │    │  Processed Output   │                 │   │
│  │  │                     │    │  + OCR Bounding Boxes│                │   │
│  │  │                     │    │  + Text Overlays     │                │   │
│  │  └─────────────────────┘    └─────────────────────┘                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Detection Log / OCR Log                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────┘
                                    ↕ WebSocket
                    ┌───────────────────────────────────┐
                    │  Client Devices (Thin Clients)    │
                    │  • Desktop Browser                │
                    │  • Mobile Phone (Same WiFi)       │
                    │  • Tablet                         │
                    └───────────────────────────────────┘
```

### Client-Server Architecture

**Server Side (Your Computer):**
- **Streamlit WebSocket Server** - Runs on port 8501
- **Python Backend** - Processes video frames, runs ML models
- **ML Models** - YOLO (wagon/damage detection), MPRNet (deblurring), EasyOCR (text recognition)
- **Session State** - Maintains state for each connected client
- **All heavy computation** happens on the server (CPU/GPU)

**Network Communication:**
- **Initial Connection** - HTTP handshake on port 8501
- **Upgrade to WebSocket** - Full-duplex bidirectional communication
- **Client → Server** - UI interactions (button clicks, slider changes)
- **Server → Client** - UI updates, processed frames (base64 encoded), metrics

**Data Flow:**
```
Mobile/Desktop Browser → WebSocket → Streamlit Server → Python Backend
                                            ↓
                                    ML Models (CPU/GPU)
                                            ↓
                                Process Frame + Detect
                                            ↓
Mobile/Desktop Browser ← WebSocket ← Annotated Frame + Metrics (base64)
```

**Why It's Fast:**
- Only **processed frames** are transmitted (not raw video)
- **Delta updates** - Only changed UI elements are sent
- **WebSocket** keeps connection alive (no HTTP overhead)
- **Browser caching** for static assets (CSS, JS)
- **Thin client model** - All ML processing on server

**Key Technologies:**
- **Streamlit** - Web framework with built-in WebSocket support
- **Tornado** - Async web server (Streamlit's backend)
- **WebSocket Protocol** - Full-duplex communication (RFC 6455)
- **Base64 Encoding** - For image transmission over WebSocket
- **Session State** - Server-side state management per client

## Processing Pipeline

1. **Frame Capture** - Video source to OpenCV frame capture
2. **CLAHE Enhancement** - L-channel enhancement for low-light conditions
3. **Wagon Detection** - YOLOv11n detects wagons
4. **Wagon Tracking** - ByteTrack assigns unique IDs
5. **ROI Extraction** - Extract wagon regions of interest
6. **Blur Detection** - Laplacian variance blur scoring
7. **Conditional Deblurring** - MPRNet on blurry ROIs (N-th frame)
8. **OCR Extraction** - EasyOCR text recognition
9. **OCR Visualization** - Bounding boxes and text overlays
10. **Damage Detection** - YOLOv11n damage classification
11. **Frame Saving** - Save OCR frames with metadata
12. **Logging** - CSV/JSON output

## Project Structure

```
railway-wagon-inspection/
├── config.py                    # Pipeline configuration
├── main.py                      # Main pipeline controller
├── run_dashboard.py             # Dashboard launcher
├── requirements.txt             # Python dependencies
│
├── pipelines/                   # Detection and processing modules
│   ├── wagon_detector.py        # YOLOv11n wagon detection
│   ├── damage_detector.py       # YOLOv11n damage detection
│   ├── blur_detector.py         # Blur detection
│   ├── ocr_pipeline.py          # EasyOCR text extraction
│   ├── deblur_manager.py        # N-th frame deblur coordination
│   └── mprnet_wrapper.py        # MPRNet deblurring wrapper
│
├── tracking/                    # Multi-object tracking
│   └── tracker.py               # ByteTrack wagon tracker
│
├── utils/                       # Utility modules
│   ├── data_models.py           # Data classes and enums
│   ├── clahe.py                 # CLAHE enhancement
│   ├── roi_utils.py             # ROI extraction and resizing
│   ├── logger.py                # CSV/JSON logging
│   └── settings_manager.py      # Settings persistence
│
├── dashboard/                   # Streamlit dashboard
│   ├── app.py                   # Main dashboard application
│   ├── ocr_visualization.py     # OCR bounding box visualization
│   ├── ocr_interval_controller.py # OCR frame interval control
│   ├── ocr_frame_saver.py       # OCR frame saving with metadata
│   └── processing_indicators.py # Processing status indicators
│
├── models/                      # YOLO model weights
│   ├── damage_detector.pt
│   └── wagon_detector.pt
│
├── MPRNet/                      # MPRNet deblurring model
│   └── Deblurring/pretrained_models/
│
├── outputs/                     # Output directory
│   ├── debug_frames/
│   └── ocr_frames/
│
└── tests/                       # Unit and property tests (355+ tests)
```

## Configuration

Create a `config.json` file:

```json
{
  "video_source": "path/to/video.mp4",
  "blur_threshold": 100.0,
  "wagon_confidence_threshold": 0.25,
  "damage_confidence_threshold": 0.5,
  "counting_line_position": 0.5,
  "counting_line_orientation": "vertical",
  "ocr_gpu_enabled": true,
  "ocr_frame_interval": 5,
  "use_fp16": true,
  "output_dir": "outputs",
  "log_format": ["csv", "json"]
}
```

## Dashboard Features

### Processing Status Indicators

| Status | Color | Description |
|--------|-------|-------------|
| APPLIED | 🟢 Green | Processing was applied to current frame |
| ACTIVE | 🟢 Green | Processing is currently active |
| NORMAL | ⚪ Gray | No processing needed |
| SKIPPED | 🟡 Yellow | Processing was skipped |
| OFF | 🔴 Red | Processing is disabled |
| ERROR | 🔴 Red | Processing error occurred |

### OCR Visualization

- **Cyan bounding boxes** around detected text
- **Text overlays** with confidence percentages
- **Orange text** for low confidence (< 50%)
- **Semi-transparent backgrounds** for readability

### OCR Frame Interval

Control OCR execution frequency via slider (1-30):
- Lower values = more frequent OCR (higher GPU usage)
- Higher values = less frequent OCR (lower GPU usage)
- Default: 5 (OCR runs every 5th frame)

## Testing

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_ocr_visualization.py

# Run property-based tests with statistics
pytest --hypothesis-show-statistics
```

## Performance Benchmarks

### Test System Specifications
- **CPU**: AMD Ryzen 7 (16 cores)
- **RAM**: 15.3 GB
- **GPU**: NVIDIA GeForce RTX 5050 Laptop (CPU mode - sm_120 compatibility))(Unable to Utilize due to PyTorch compactibilty)
- **OS**: Windows 11
- **Test Video**: 1280x720 @ 60 FPS, 20,858 frames

### Performance Metrics (CPU Mode)

| Metric | Value | Notes |
|--------|-------|-------|
| **Average FPS** | 18.45 | Real-time processing speed |
| **Peak FPS** | 33.15 | Maximum achieved framerate |
| **Average Latency** | 152 ms | Per-frame processing time |
| **CPU Usage** | 809% | Multi-core utilization (16 cores) |
| **Memory Usage** | 1.5 GB | RAM consumption |

### Component Performance Breakdown

| Component | Average Time | Description |
|-----------|--------------|-------------|
| **Wagon Detection** | 33.75 ms | YOLOv11n wagon detection |
| **Deblurring** | 205.20 ms | MPRNet ROI deblurring (N-th frame) |
| **OCR** | 106.99 ms | EasyOCR text extraction (every 5th frame) |
| **Total Pipeline** | 152.15 ms | End-to-end processing |

### Throughput
- **Processed**: 20,858 frames in 53.5 minutes
- **Overall FPS**: 6.49 (with all features enabled)
- **Real-time Capability**: 18.45 FPS average (suitable for 15-20 FPS video streams)

### Performance Notes
- Running in **CPU mode** due to RTX 5050 sm_120 architecture (PyTorch compatibility)
- **GPU mode** (when supported) expected to provide 5-10x speedup
- Deblurring runs every 3rd frame with caching (3x speedup)
- OCR runs every 5th frame (5x speedup)
- Multi-core CPU utilization: 809% (efficient use of 16 cores)

### Optimization Impact

| Feature | Speedup | Description |
|---------|---------|-------------|
| N-th Frame Deblurring | 3x | Cache results for intermediate frames |
| OCR Frame Interval | 5x | Run OCR every 5th frame |
| ROI-only Processing | 10x | Process small regions vs full frame |
| Multi-threading | 8x | Parallel processing on 16 cores |

## GPU Memory Management

Optimized for NVIDIA RTX 3050 (6 GB VRAM):

- **ROI-only deblurring** - Max 256x256 pixels
- **FP16 inference** - 50% memory reduction
- **N-th frame execution** - Cached results for intermediate frames
- **OCR frame interval** - Configurable OCR frequency
- **Automatic cache clearing** - On wagon exit

## Output Files

### CSV/JSON Logs
```
outputs/logs.csv
outputs/logs.json
```

### OCR Frames with Metadata
```
outputs/ocr_frames/ocr_20260111_103045_123456_100_1.jpg
outputs/ocr_frames/ocr_20260111_103045_123456_100_1.json
```

### Metadata JSON Structure
```json
{
  "timestamp": "2026-01-11T10:30:45.123456",
  "frame_index": 100,
  "wagon_id": 1,
  "detections": [
    {
      "text": "ABC123",
      "confidence": 0.95,
      "bbox": {"x1": 10, "y1": 20, "x2": 100, "y2": 50}
    }
  ],
  "deblur_applied": true,
  "illumination_applied": false
}
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [YOLOv11](https://github.com/ultralytics/ultralytics) - Object detection
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) - Text recognition
- [MPRNet](https://github.com/swz30/MPRNet) - Image deblurring
- [ByteTrack](https://github.com/ifzhang/ByteTrack) - Multi-object tracking
- [Streamlit](https://streamlit.io/) - Dashboard framework
