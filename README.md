# 📸 Advanced OpenCV Programs

This repository contains a collection of advanced computer vision programs built using **OpenCV** in Python. These scripts demonstrate various real-time and complex image/video processing techniques, object detection/tracking, and integration with deep learning models.

---

## 🚀 Features

- Real-time object detection using YOLOv10 and OpenCV DNN
- Object tracking with MeanShift and CamShift
- Image segmentation using PyTorch models converted to ONNX
- Video analytics using pre-trained models
- Drawing and image annotation utilities
- Integration with webcam/video input

---

## 🛠️ Requirements

- Python 3.8+
- OpenCV (`opencv-python`, `opencv-contrib-python`)
- NumPy
- PyTorch (for model export)
- ONNX and `onnxruntime` (for ONNX inference)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📂 Project Structure

```
advanced-opencv/
│
├── object_detection/
│   ├── yolo_opencv.py         # YOLOv10 with OpenCV DNN
│   └── utils.py               # Utility functions for drawing, preprocessing
│
├── object_tracking/
│   ├── meanshift_tracker.py   # Meanshift tracking demo
│   └── camshift_tracker.py    # Camshift tracking demo
│
├── segmentation/
│   ├── convert_to_onnx.py     # Export PyTorch segmentation model to ONNX
│   └── segment_opencv.py      # Run segmentation using ONNX + OpenCV
│
├── assets/                    # Test images and videos
│
└── README.md
```

---

## 🧪 How to Run

**1. Object Detection (YOLO):**
```bash
python object_detection/yolo_opencv.py
```

**2. Object Tracking (MeanShift/CamShift):**
```bash
python object_tracking/meanshift_tracker.py
python object_tracking/camshift_tracker.py
```

**3. Segmentation with ONNX Model:**
```bash
python segmentation/segment_opencv.py
```

---

## 📸 Sample Outputs

- Bounding boxes over real-time webcam feed
- Smooth object tracking with visual feedback
- Color-coded segmentation overlay

---

## 📘 References

- [OpenCV Docs](https://docs.opencv.org/)
- [YOLOv10](https://github.com/WongKinYiu/yolov10)
- [ONNX](https://onnx.ai/)

---
