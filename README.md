<div align="center">

# 🤟 VNU-IS: Real-time Sign Language Translation System

**A Computer Vision & Deep Learning approach for continuous sign language recognition.**

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.9.0-00BFA5.svg?style=for-the-badge&logo=google&logoColor=white)](https://google.github.io/mediapipe/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8.svg?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

*Dự án cuối kỳ học phần Computer Vision - VNU International School.*

</div>

---

## 📖 Table of Contents
- [Project Overview](#-project-overview)
- [System Architecture](#-system-architecture)
- [Dataset Strategy](#-dataset-strategy)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#-usage)
- [Core Team](#-core-team)
- [License](#-license)

---

## 🎯 Project Overview
Hệ thống **Real-time Sign Language Translation** được thiết kế nhằm thu hẹp khoảng cách giao tiếp bằng cách dịch ngôn ngữ ký hiệu thành văn bản theo thời gian thực. 

Thay vì sử dụng các mô hình Convolutional Neural Networks (CNN) truyền thống tốn nhiều tài nguyên để xử lý chuỗi hình ảnh, hệ thống áp dụng kỹ thuật **Landmark Extraction** (trích xuất điểm neo) kết hợp với **Recurrent Neural Networks (LSTM)**. Phương pháp này giảm thiểu 90% khối lượng tính toán, cho phép triển khai (inference) mượt mà trên các thiết bị máy tính cá nhân tiêu chuẩn (Edge Devices) mà không yêu cầu GPU chuyên dụng.

---

## ⚙️ System Architecture

Quy trình hoạt động (Pipeline) của hệ thống được chia làm 3 module độc lập:

1. **Vision Module (OpenCV & MediaPipe Holistic):** Bắt luồng video trực tiếp từ Webcam. Sử dụng MediaPipe để trích xuất hệ tọa độ đa chiều (3D coordinates) của khuôn mặt, thân trên và hai bàn tay.
2. **Preprocessing Module:** Xử lý các điểm nhiễu, chuẩn hóa tọa độ và gom cụm dữ liệu thành các chuỗi (sequences) có độ dài cố định (30 frames/sequence).
3. **Inference Module (TensorFlow/Keras):** Đưa dữ liệu chuỗi thời gian vào mô hình LSTM đã được huấn luyện (Pre-trained weights) để tính toán phân phối xác suất (Softmax) và trả về từ vựng tương ứng.

---

## 📂 Dataset Strategy
Để tối ưu hóa chi phí huấn luyện, dự án sử dụng tập con (subset) của bộ dữ liệu **WLASL (Word-Level American Sign Language)**.
- **Classes:** 5 từ vựng cơ bản (Hello, Thanks, I Love You, Please, Sorry).
- **Data Engineering:** Toàn bộ dữ liệu video thô (.mp4) được chạy qua batch-script để chuyển đổi thành các mảng NumPy (`.npy`).
- *Lưu ý: Dữ liệu thô và mô hình trọng số (`.h5`) được bỏ qua (ignored) trên repository này theo tiêu chuẩn quản lý mã nguồn.*

---

## 🗂 Project Structure

```text
Sign-Language-Recognition/
├── assets/                 # UI assets & documentation images
├── data/                   # [Local Only] Dataset directory
│   ├── raw/                # Source video files
│   └── processed/          # Extracted landmarks (.npy matrices)
├── models/                 # [Local Only] Compiled model weights (.h5)
├── notebooks/              # Jupyter notebooks for EDA and Model Training
├── src/                    # Core source code
│   ├── config.py           # Global constants & hyperparameters
│   ├── data_collection.py  # Automation script for dataset generation
│   ├── preprocessing.py    # Train-test split & label encoding
│   ├── model.py            # LSTM network architecture definition
│   └── utils.py            # Helper functions for rendering & math
├── .gitignore              # Git exclusion rules
├── app.py                  # Main entry point for Real-time Inference
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
