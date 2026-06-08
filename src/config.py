import os
import numpy as np

# 1. Đường dẫn các thư mục cần thiết
DATA_PATH = os.path.join('data', 'processed')
MODELS_PATH = os.path.join('models')
LOGS_PATH = os.path.join('logs')

# 2. Các hành động (từ vựng) cần nhận diện
_actions = []
if os.path.exists(DATA_PATH):
    _actions = [d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]

if len(_actions) > 0:
    ACTIONS = np.array(sorted(_actions))
else:
    # THÊM TỪ 'idle' ĐỂ CHỐNG NHIỄU KHI HẠ TAY
    ACTIONS = np.array([
        'hello', 'how', 'are', 'you', 'today',
        'i', 'am', 'fine', 'thank',
        'please', 'can', 'help', 'me',
        'love', 'very', 'much',
        'sorry', 'late', 'idle' 
    ])

# 3. Thông số kỹ thuật cho mô hình
NO_SEQUENCES = 30     # Số lượng video (chuỗi) thu thập cho mỗi hành động
SEQUENCE_LENGTH = 30  # Số lượng frame (khung hình) bắt buộc cho mỗi video