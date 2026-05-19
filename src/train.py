import os
from tensorflow.keras.callbacks import TensorBoard

# TODO: Sửa lại đường dẫn import nếu cần
from src.preprocessing import load_data
from src.model import build_model

def train():
    """
    Hàm thực thi quá trình huấn luyện mô hình.
    Yêu cầu:
    1. Gọi load_data() để lấy tập dữ liệu huấn luyện.
    2. Gọi build_model() để tạo lập mô hình.
    3. (Tùy chọn) Cài đặt TensorBoard để lưu log huấn luyện.
    4. Chạy hàm model.fit() với tập train.
    5. Lưu file trọng số (.h5) sinh ra vào thư mục models/.
    """
    # TODO: Cài đặt logic train ở đây
    
    pass

if __name__ == '__main__':
    train()
