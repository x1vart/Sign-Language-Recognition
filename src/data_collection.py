import os
import cv2
import numpy as np
import mediapipe as mp

# TODO: Sửa lại đường dẫn import nếu cần
from src.config import DATA_PATH, ACTIONS, NO_SEQUENCES, SEQUENCE_LENGTH
from src.utils import mediapipe_detection, draw_styled_landmarks, extract_keypoints

def collect_data():
    """
    Hàm thu thập dữ liệu video từ webcam và lưu trữ dưới dạng file numpy.
    Yêu cầu:
    1. Kiểm tra và tạo cấu trúc thư mục chứa dữ liệu (nếu chưa tồn tại).
       VD: data/processed/hello/0, data/processed/hello/1, ...
    2. Khởi tạo Webcam và MediaPipe Holistic.
    3. Dùng vòng lặp lồng nhau (ACTIONS -> NO_SEQUENCES -> SEQUENCE_LENGTH) để chụp.
    4. Khi chụp từng frame, trích xuất keypoints và lưu thành file .npy.
    5. Cần hiển thị text lên màn hình báo chờ để người dùng chuẩn bị trước mỗi chuỗi (sequence) mới.
    """
    # TODO: Viết logic tạo thư mục
    
    # TODO: Viết logic mở camera và thu thập dữ liệu
    
    pass

if __name__ == '__main__':
    collect_data()
