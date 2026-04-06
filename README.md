---
title: Robot Vision System
emoji: 🤖
colorFrom: red
colorTo: red
sdk: gradio
sdk_version: "4.44.1"
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
---

# 🤖 Robot Vision System — Open-Source Object & Hazard Detection

[![GitHub](https://img.shields.io/badge/GitHub-robocam_vision-black?logo=github)](https://github.com/DebangaBaruah/robocam_vision)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/prob12/robocam)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)

A powerful, easy-to-use **real-time computer vision system** for detecting objects, fire, smoke, and water falling using YOLOv3 deep learning and OpenCV. Perfect for robotics, security monitoring, industrial safety, and emergency response applications.

**Live Demo**: [🚀 Try on Hugging Face Spaces](https://huggingface.co/spaces/prob12/robocam)

## ✨ Features

- 📹 **Live Webcam Detection**: Real-time object detection directly from your camera
- 📤 **Video Upload & Processing**: Batch process videos with detailed statistics
- 🟢 **80 COCO Classes**: Detect person, car, bicycle, dog, cat, and 76 more objects
- 🔴 **Fire Detection**: Real-time fire hazard recognition using color segmentation
- 🟡 **Smoke Detection**: Motion-based smoke identification with gray-level heuristics
- 🔵 **Water Falling Detection**: Background subtraction for detecting falling water/rain
- ⚡ **Adjustable Confidence**: Fine-tune detection sensitivity with confidence threshold slider
- 🎛️ **Toggle Detectors**: Enable/disable individual detectors on the fly
- 🎨 **Live Bounding Boxes**: Color-coded boxes for each detection type
- 📊 **Detailed Statistics**: Processing time and detection counts per frame

## 🎯 What Gets Detected?

| Detector | Method | Color | Use Case |
|---|---|---|---|
| **General Objects (80 COCO)** | YOLOv3 ResNet | 🟢 Green | People, vehicles, animals, items |
| **Fire** | HSV color segmentation | 🔴 Red | Flame/fire hazard detection |
| **Smoke** | Frame diff + gray heuristic | 🟡 Yellow | Fire/hazard early warning |
| **Water Falling** | MOG2 background subtraction | 🔵 Blue | Rain, water sprays, leaks |

## 🚀 Quick Start

### Option 1: Run Locally

**Prerequisites**: Python 3.10+ and pip

```bash
# Clone the repository
git clone https://github.com/DebangaBaruah/robocam_vision.git
cd robocam_vision/robot_vision_app

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Then open **http://127.0.0.1:7860** in your browser.

### Option 2: Try Online

No installation needed! Use the [live demo on Hugging Face Spaces](https://huggingface.co/spaces/prob12/robocam).

## 📖 Usage Guide

### Live Webcam Detection

1. Navigate to the **"📹 Live Webcam Detection"** tab
2. Allow camera access when prompted
3. Adjust the **Confidence Threshold** slider (lower = more detections)
4. Toggle detectors on/off as needed
5. Watch real-time detection with bounding boxes

### Video Upload Processing

1. Go to **"📤 Video Upload Processing"** tab
2. Upload a video file (MP4, AVI, MOV, etc.)
3. Configure:
   - **Confidence Threshold**: Detection sensitivity
   - **Skip Frames**: Process every Nth frame (faster but less precise)
   - **Detectors**: Enable/disable specific detection types
4. Click **"▶ Run Detection"**
5. Download the annotated output video when done

## 🛠️ Configuration

### Adjustable Parameters

**Confidence Threshold** (0.05 - 0.95)
- Lower values → more detections (more false positives)
- Higher values → fewer detections (fewer false positives)
- **Recommended default**: 0.25

**Skip Frames** (0 - 5)
- 0 = Process every frame (slowest, highest quality)
- 2-3 = Good balance of speed and accuracy
- 5 = Skip most frames (fastest, lower quality)

## 📂 Project Structure

```
robocam_vision/
├── robot_vision_app/
│   ├── app.py                      # Gradio UI (main entry point)
│   ├── requirements.txt            # Python dependencies
│   ├── README.md                   # This file
│   ├── LICENSE                     # MIT License
│   ├── models/                     # YOLOv3 model files (auto-downloaded)
│   │   ├── yolov3.cfg             # Network architecture
│   │   ├── yolov3.weights         # Pre-trained weights (~237 MB)
│   │   └── coco.names             # 80 object class names
│   ├── outputs/                    # Generated output videos
│   └── utils/
│       ├── __init__.py
│       ├── detectors.py            # Detection classes
│       ├── robot_vision_system.py  # Main orchestrator
│       └── model_downloader.py     # Auto-download models
├── run_app.ps1                     # PowerShell launcher
├── run_app.bat                     # Batch file launcher
└── .gitignore                      # Git ignore rules
```

## 📋 Requirements

- **Python**: 3.10 or higher
- **Memory**: 4GB minimum (8GB+ recommended for video processing)
- **Disk Space**: 1GB for models + output videos
- **GPU**: Optional (CPU works fine for real-time detection)

## 📦 Dependencies

```
Gradio (>=4.0.0)          # Web UI framework
OpenCV (>=4.8.0)          # Computer vision
NumPy (>=1.24.0)          # Numerical computing
```

Full list in [requirements.txt](requirements.txt)

## 🔧 Advanced Usage

### Custom Confidence Thresholds

Edit the default confidence in `app.py`:

```python
webcam_confidence = gr.Slider(
    minimum=0.05, maximum=0.95, value=0.25, step=0.05,  # Change 0.25 to your preference
    label="Confidence Threshold"
)
```

### Disable Specific Detectors

Uncheck the detector toggles in the UI, or modify the initialization in `app.py`:

```python
enable_objects=True   # Set to False to skip object detection
enable_fire=True      # Set to False to skip fire detection
enable_smoke=True     # Set to False to skip smoke detection
enable_water=True     # Set to False to skip water detection
```

## 📊 Performance Notes

- **Real-time webcam**: 20-30 FPS on CPU (Intel i5/i7)
- **Video processing**: ~5-10 mins per minute of video on CPU
- **First run**: Auto-downloads YOLOv3 weights (~237 MB) — may take 2-5 minutes
- **GPU acceleration**: Supported via OpenCV CUDA (requires NVIDIA GPU + CUDA toolkit)

## 🤝 Contributing

Contributions are welcome! Here's how:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

## 🐛 Issues & Bug Reports

Found a bug? Please open an [issue on GitHub](https://github.com/DebangaBaruah/robocam_vision/issues) with:
- Description of the problem
- Steps to reproduce
- Expected vs. actual behavior
- Your system info (OS, Python version, etc.)

## 📝 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) file for details.

## 👨‍💻 Author & Credits

**Created by**: Debanga Baruah

**Built with**:
- [YOLOv3](https://github.com/pjreddie/darknet) — Object detection
- [Gradio](https://www.gradio.app/) — Web UI
- [OpenCV](https://opencv.org/) — Computer vision
- [NumPy](https://numpy.org/) — Numerical computing

## 📞 Support

- **GitHub Issues**: [Report bugs](https://github.com/DebangaBaruah/robocam_vision/issues)
- **Discussions**: [Ask questions](https://github.com/DebangaBaruah/robocam_vision/discussions)
- **Live Demo**: [Try it now](https://huggingface.co/spaces/prob12/robocam)

## 🌟 Star Us!

If you find this project helpful, please give it a ⭐ on GitHub!

---

**Disclaimer**: This system is for educational and monitoring purposes. For critical safety applications, always use professionally-certified fire/water detection systems.

    ├── detectors.py          # Detection classes
    ├── robot_vision_system.py # Main orchestrator
    └── model_downloader.py   # Model file management
```
    └── model_downloader.py   # Auto-download helper
```

## License

MIT — free to use, modify, and deploy.
