import cv2
import numpy as np
import mediapipe as mp
import os
import sys

# Thêm đường dẫn src vào PYTHONPATH nếu muốn chạy app.py ở thư mục gốc
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.config import ACTIONS, SEQUENCE_LENGTH
from src.model import build_model
from src.utils import mediapipe_detection, draw_styled_landmarks, extract_keypoints

def run_app():
    """
    Hàm chính chạy ứng dụng nhận diện thời gian thực.
    Yêu cầu:
    1. Khởi tạo mô hình LSTM và load trọng số file models/lstm_model.h5.
    2. Mở kết nối Webcam.
    3. Duyệt từng frame hình ảnh qua MediaPipe để trích xuất tọa độ.
    4. Lưu trữ tọa độ vào mảng tạm, đủ SEQUENCE_LENGTH frame thì tiến hành dự đoán.
    5. Xử lý làm mượt kết quả (chống nhiễu) và hiển thị kết quả text lên màn hình OpenCV.
    """
    # TODO: Load mô hình đã huấn luyện
    
    # TODO: Khởi tạo camera và vòng lặp thu thập - nhận diện
    
    pass

if __name__ == '__main__':
    run_app()
