import os

# Đường dẫn đến thư mục chứa dữ liệu đã được xử lý (trích xuất đặc trưng/keypoints)
DATA_PATH = os.path.join('data', 'processed')

# Các hành động (ngôn ngữ ký hiệu) cần nhận diện
ACTIONS = ['hello', 'thanks', 'iloveyou']

# Số lượng video (sequence) sẽ thu thập cho mỗi hành động
NO_SEQUENCES = 30

# Số lượng frame (khung hình) cho mỗi video (sequence)
SEQUENCE_LENGTH = 30
