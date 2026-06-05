import os
import sys

# Đảm bảo có thể import từ src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tensorflow.keras.callbacks import TensorBoard
from src.preprocessing import load_data
from src.model import build_model, get_available_models
from src.config import MODELS_PATH, LOGS_PATH

def train():
    """
    Thực thi quá trình huấn luyện toàn bộ các mô hình và lưu lại tệp h5
    """
    os.makedirs(MODELS_PATH, exist_ok=True)
    os.makedirs(LOGS_PATH, exist_ok=True)

    print("[*] Đang load và xử lý dữ liệu...")
    X_train, X_test, y_train, y_test = load_data()
    
    if X_train is None or len(X_train) == 0:
        print("\n[LỖI] Không tìm thấy dữ liệu. Bạn hãy chạy `python src/data_collection.py` hoặc `python src/auto_process.py` để lấy dữ liệu trước nhé!")
        return

    print(f"[+] Dữ liệu huấn luyện: {X_train.shape[0]} mẫu, Dữ liệu kiểm thử: {X_test.shape[0]} mẫu.")
    available_models = get_available_models()
    
    # Train lần lượt từng kiến trúc mô hình (LSTM, GRU, Dense)
    for m_name in available_models:
        print(f"\n{'='*50}")
        print(f"[*] ĐANG HUẤN LUYỆN MÔ HÌNH: {m_name}")
        print(f"{'='*50}")
        
        # Build model architecture
        model = build_model(model_type=m_name)
        
        # Callbacks
        tb_callback = TensorBoard(log_dir=os.path.join(LOGS_PATH, m_name))
        
        # Training
        model.fit(
            X_train, y_train, 
            epochs=100, 
            callbacks=[tb_callback], 
            validation_data=(X_test, y_test)
        )
        
        # Lưu file weights (.h5) khớp với app.py
        save_path = os.path.join(MODELS_PATH, f'{m_name.lower()}_model.h5')
        model.save_weights(save_path)
        print(f"\n[+] ĐÃ LƯU THÀNH CÔNG trọng số cho {m_name} tại: {save_path}")

    print("\n[*] Đã huấn luyện xong toàn bộ các model! Bạn có thể chạy `python app.py` để trải nghiệm.")

if __name__ == '__main__':
    train()
