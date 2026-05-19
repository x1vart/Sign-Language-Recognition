from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# TODO: Sửa lại đường dẫn import nếu cần
from src.config import ACTIONS

def build_model(input_shape=(30, 1662)):
    """
    Hàm định nghĩa kiến trúc mạng nơ-ron LSTM.
    Yêu cầu:
    1. Khởi tạo model kiểu Sequential.
    2. Thêm các lớp LSTM (chú ý tham số return_sequences và input_shape).
    3. Thêm các lớp Dense (Fully Connected).
    4. Lớp đầu ra (output layer) phải có số node bằng số lượng ACTIONS, dùng hàm kích hoạt softmax.
    5. Cấu hình (compile) mô hình với optimizer và loss function phù hợp.
    """
    model = Sequential()
    
    # TODO: Cài đặt kiến trúc các lớp mạng ở đây
    
    return model

if __name__ == '__main__':
    # TODO: Gọi thử build_model() và in ra model.summary() để xem số lượng parameters.
    pass
