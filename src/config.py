import os
import numpy as np

# 1. Đường dẫn các thư mục cần thiết
DATA_PATH = os.path.join('data', 'processed')
MODELS_PATH = os.path.join('models')
LOGS_PATH = os.path.join('logs')

# 2. Các hành động (từ vựng) cần nhận diện
# Bộ 18 từ vựng ghép thành 5 câu hoàn chỉnh (mỗi câu 4-5 từ):
# 1. "hello how are you today" (5 từ)
# 2. "i am fine thank you" (5 từ)
# 3. "please can you help me" (5 từ)
# 4. "i love you very much" (5 từ)
# 5. "sorry i am late today" (5 từ)
ACTIONS = np.array([
    'hello', 'how', 'are', 'you', 'today',
    'i', 'am', 'fine', 'thank',
    'please', 'can', 'help', 'me',
    'love', 'very', 'much',
    'sorry', 'late'
])

# 3. Thông số kỹ thuật cho mô hình
NO_SEQUENCES = 30     # Số lượng video (chuỗi) thu thập cho mỗi hành động
SEQUENCE_LENGTH = 30  # Số lượng frame (khung hình) bắt buộc cho mỗi video
