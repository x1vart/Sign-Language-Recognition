import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Đảm bảo có thể import từ src nếu chạy độc lập
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.config import DATA_PATH, ACTIONS, SEQUENCE_LENGTH

def load_data():
    """
    Đọc dữ liệu file .npy, gán nhãn One-Hot, chia Train/Test
    """
    label_map = {label: num for num, label in enumerate(ACTIONS)}
    sequences, labels = [], []
    
    # Duyệt qua các hành động
    for action in ACTIONS:
        action_path = os.path.join(DATA_PATH, action)
        if not os.path.exists(action_path):
            continue
            
        # Lấy tất cả các thư mục con (mỗi thư mục con là 1 video/sequence)
        sequence_dirs = [d for d in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, d))]
        
        for sequence_dir in sequence_dirs:
            window = []
            sequence_path = os.path.join(action_path, sequence_dir)
            
            # Kiểm tra xem đủ số lượng frame chưa
            files_in_seq = [f for f in os.listdir(sequence_path) if f.endswith('.npy')]
            if len(files_in_seq) < SEQUENCE_LENGTH:
                continue
                
            try:
                for frame_num in range(SEQUENCE_LENGTH):
                    res = np.load(os.path.join(sequence_path, f"{frame_num}.npy"))
                    window.append(res)
                
                sequences.append(window)
                labels.append(label_map[action])
            except Exception as e:
                print(f"[-] Lỗi đọc frame ở {sequence_path}: {e}")
                
    if len(sequences) == 0:
        return None, None, None, None
        
    X = np.array(sequences)
    y = to_categorical(labels, num_classes=len(ACTIONS))
    
    # Chia 10% tập test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()
    if X_train is not None:
        print(f"[*] Dữ liệu Train X: {X_train.shape}, Y: {y_train.shape}")
        print(f"[*] Dữ liệu Test  X: {X_test.shape}, Y: {y_test.shape}")
    else:
        print("[-] Không tìm thấy dữ liệu!")
