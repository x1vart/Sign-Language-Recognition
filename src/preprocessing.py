import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# TODO: Sửa lại đường dẫn import nếu cần
from src.config import DATA_PATH, ACTIONS, NO_SEQUENCES, SEQUENCE_LENGTH

def load_data():
    """
    Hàm đọc dữ liệu đã thu thập từ ổ cứng và tiến hành chia tập Train/Test.
    Yêu cầu:
    1. Đọc các file .npy đã lưu trong DATA_PATH.
    2. Ghép nối thành các sequences hoàn chỉnh với kích thước (NO_SEQUENCES, SEQUENCE_LENGTH, số lượng keypoints).
    3. Tạo mảng nhãn (labels) tương ứng và chuyển đổi sang dạng One-Hot Encoding.
    4. Dùng train_test_split để chia dữ liệu (gợi ý: lấy 5-10% cho test_size).
    """
    # TODO: Cài đặt logic load dữ liệu và gán nhãn
    
    # TODO: Chia dữ liệu và trả về
    X_train, X_test, y_train, y_test = None, None, None, None
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    # Kiểm tra thử hàm load_data()
    # X_train, X_test, y_train, y_test = load_data()
    pass
