import os
import numpy as np

# 1. Đường dẫn các thư mục cần thiết
DATA_PATH = os.path.join('data', 'processed')
MODELS_PATH = os.path.join('models')
LOGS_PATH = os.path.join('logs')

# 2. Các hành động (từ vựng) cần nhận diện
# Đã chốt 5 từ cơ bản theo file README
ACTIONS = np.array(['hello', 'thanks', 'iloveyou', 'please', 'sorry'])

# 3. Thông số kỹ thuật cho mô hình
NO_SEQUENCES = 30     # Số lượng video thu thập cho mỗi hành động
SEQUENCE_LENGTH = 30  # Số lượng frame (khung hình) bắt buộc cho mỗi video
