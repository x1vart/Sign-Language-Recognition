<div align="center">

#   Real-time Sign Language Translation System

**A Computer Vision & Deep Learning approach for continuous sign language recognition.**

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.x-00BFA5.svg?style=for-the-badge&logo=google&logoColor=white)](https://google.github.io/mediapipe/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8.svg?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

*Dự án cuối kỳ học phần Computer Vision - VNU International School.*

</div>

---

## 📖 Table of Contents
- [Project Overview](#-project-overview)
- [System Architecture](#-system-architecture)
- [Dataset & Sentences](#-dataset--sentences)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Usage](#-usage)

---

## 🎯 Project Overview
Hệ thống **Real-time Sign Language Translation** được thiết kế nhằm thu hẹp khoảng cách giao tiếp bằng cách dịch ngôn ngữ ký hiệu thành các **câu hoàn chỉnh** theo thời gian thực. 

Thay vì chỉ nhận diện từng từ đơn lẻ, hệ thống áp dụng kỹ thuật **Landmark Extraction** (trích xuất điểm neo) kết hợp với **Recurrent Neural Networks (LSTM, GRU)** và thuật toán logic ghép câu. Phương pháp này giảm thiểu 90% khối lượng tính toán, cho phép ứng dụng chạy mượt mà trên Edge Devices mà không cần GPU chuyên dụng, đồng thời cung cấp giao diện trực quan cho phép người dùng chuyển đổi linh hoạt giữa nhiều mô hình mạng Nơ-ron khác nhau.

---

## ⚙️ System Architecture

Quy trình hoạt động (Pipeline) của hệ thống được chia làm 3 module chính:

1. **Vision Module (OpenCV & MediaPipe Holistic):** Bắt luồng video trực tiếp từ Webcam. Trích xuất 1662 tọa độ đa chiều (3D coordinates) của khuôn mặt, thân trên và hai bàn tay.
2. **Preprocessing Module:** Khử nhiễu và đóng gói luồng dữ liệu thời gian thực thành các "cửa sổ trượt" (Sliding Windows) với độ dài cố định 30 frames/sequence.
3. **Inference & UI Module (TensorFlow/Keras):** 
   - Đưa chuỗi tọa độ vào các mô hình (`LSTM`, `GRU`, `Dense`) để dự đoán phân phối xác suất.
   - Thuật toán ổn định (ngăn chặn lặp từ) sẽ ghép các từ đơn lẻ thành 1 câu hoàn chỉnh.
   - Giao diện OpenCV được thiết kế theo phong cách UI hiện đại để hiển thị kết quả.

---

## 📂 Dataset & Sentences
Hệ thống hỗ trợ thư viện gồm **18 từ vựng** cơ bản để có thể dịch mượt mà **5 câu giao tiếp hoàn chỉnh** sau:
1. `hello how are you today` (Xin chào hôm nay bạn thế nào)
2. `i am fine thank you` (Tôi khỏe cảm ơn bạn)
3. `please can you help me` (Làm ơn bạn có thể giúp tôi không)
4. `i love you very much` (Tôi yêu bạn rất nhiều)
5. `sorry i am late today` (Xin lỗi hôm nay tôi đến trễ)

**Danh sách 18 từ vựng:** `hello, how, are, you, today, i, am, fine, thank, please, can, help, me, love, very, much, sorry, late`

> *Lưu ý: Bạn có thể tự thu thập dữ liệu thông qua Webcam bằng file `data_collection.py`, hoặc tự động xử lý tập dữ liệu WLASL bằng file `auto_process.py`. Dữ liệu thô và trọng số mô hình (`.h5`) bị loại trừ khỏi Git.*

---

## 🗂 Project Structure

```text
Sign-Language-Recognition/
├── data/                   # [Ignored] Dữ liệu video thô và đã xử lý (.npy)
├── models/                 # [Ignored] Trọng số mô hình đã train (.h5)
├── logs/                   # [Ignored] Lịch sử huấn luyện Tensorboard
├── src/                    
│   ├── config.py           # Cấu hình danh sách từ vựng & tham số model
│   ├── data_collection.py  # Script mở webcam thu thập dữ liệu train
│   ├── auto_process.py     # Script bóc tách dữ liệu từ file .mp4 có sẵn
│   ├── preprocessing.py    # Xử lý Label Encoding & Train-test split
│   ├── model.py            # Kiến trúc các model mạng (LSTM, GRU, Dense)
│   └── utils.py            # Hàm xử lý MediaPipe và vẽ Landmarks
├── .gitignore              # Quy tắc loại trừ file
├── app.py                  # Ứng dụng chính (Giao diện nhận diện Real-time)
├── requirements.txt        # Danh sách thư viện Python
└── README.md               # Tài liệu dự án
```

---

## 🚀 Usage

**1. Cài đặt thư viện:**
```bash
pip install -r requirements.txt
```

**2. Thu thập dữ liệu (Chọn 1 trong 2 cách):**
- Tự quay bằng Webcam: `python src/data_collection.py`
- Dùng video .mp4 có sẵn: `python src/auto_process.py`

**3. Khởi chạy Ứng dụng Thời gian thực:**
```bash
python app.py
```
*Tương tác trên giao diện:*
- Bấm `1`, `2`, `3` để chuyển đổi qua lại giữa các mô hình (LSTM, GRU, Dense).
- Bấm `C` để xóa câu dịch hiện tại trên màn hình.
- Bấm `Q` để thoát.
