import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.config import ACTIONS, MODELS_PATH, LOGS_PATH
from src.preprocessing import load_data
from src.model import build_model, get_available_models

RESULTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'results')

def plot_training_curves(model_name):
    history_path = os.path.join(LOGS_PATH, f'{model_name.lower()}_history.json')
    if not os.path.exists(history_path):
        print(f"[-] Không tìm thấy file lịch sử huấn luyện cho {model_name}. Hãy chạy lại lệnh `python src/train.py` để sinh ra dữ liệu biểu đồ.")
        return
    
    with open(history_path, 'r', encoding='utf-8') as f:
        history = json.load(f)
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot Accuracy
    if 'categorical_accuracy' in history and 'val_categorical_accuracy' in history:
        ax1.plot(history['categorical_accuracy'], label='Train Accuracy')
        ax1.plot(history['val_categorical_accuracy'], label='Val Accuracy')
    elif 'accuracy' in history and 'val_accuracy' in history:
        ax1.plot(history['accuracy'], label='Train Accuracy')
        ax1.plot(history['val_accuracy'], label='Val Accuracy')
        
    ax1.set_title(f'{model_name} - Tốc độ học (Accuracy)')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot Loss
    if 'loss' in history and 'val_loss' in history:
        ax2.plot(history['loss'], label='Train Loss')
        ax2.plot(history['val_loss'], label='Val Loss')
    ax2.set_title(f'{model_name} - Lỗi (Loss)')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_PATH, f'{model_name}_training_curves.png')
    plt.savefig(save_path)
    plt.close()
    print(f"[+] Đã xuất Biểu đồ huấn luyện (Training Curves) thành công tại: {save_path}")

def evaluate_and_analyze(model_name, model, X_test, y_test):
    # Predict
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ACTIONS)
    fig, ax = plt.subplots(figsize=(12, 10))
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical')
    plt.title(f'Ma trận nhầm lẫn (Confusion Matrix) - {model_name}')
    
    cm_save_path = os.path.join(RESULTS_PATH, f'{model_name}_confusion_matrix.png')
    plt.savefig(cm_save_path)
    plt.close()
    print(f"[+] Đã xuất Confusion Matrix thành công tại: {cm_save_path}")
    
    # Failure Case Analysis
    print(f"\n--- Phân tích Lỗi nhận diện (Failure Cases) của {model_name} ---")
    incorrect_indices = np.where(y_pred_classes != y_true_classes)[0]
    
    if len(incorrect_indices) == 0:
        print("[+] Tuyệt vời! Mô hình đoán đúng 100% trên tập Test.")
    else:
        for idx in incorrect_indices:
            true_label = ACTIONS[y_true_classes[idx]]
            pred_label = ACTIONS[y_pred_classes[idx]]
            confidence = y_pred[idx][y_pred_classes[idx]] * 100
            print(f"  - Mẫu số {idx}: Thực tế là '{true_label}' --> AI đoán nhầm thành '{pred_label}' (Độ tự tin: {confidence:.2f}%)")
    print("-" * 50)

def main():
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    print("[*] Đang nạp tập dữ liệu kiểm thử (Test Set)...")
    X_train, X_test, y_train, y_test = load_data()
    
    if X_test is None or len(X_test) == 0:
        print("[-] Không tìm thấy dữ liệu kiểm thử.")
        return
        
    available_models = get_available_models()
    for m_name in available_models:
        print(f"\n{'='*50}")
        print(f"[*] TIẾN HÀNH ĐÁNH GIÁ MÔ HÌNH: {m_name}")
        print(f"{'='*50}")
        
        # 1. Vẽ đồ thị huấn luyện
        plot_training_curves(m_name)
        
        # 2. Phân tích Confusion Matrix & Lỗi sai
        weight_path = os.path.join(MODELS_PATH, f'{m_name.lower()}_model.h5')
        if not os.path.exists(weight_path):
            print(f"[-] Không tìm thấy file trọng số {weight_path}. Bỏ qua.")
            continue
            
        model = build_model(model_type=m_name)
        model.load_weights(weight_path)
        
        evaluate_and_analyze(m_name, model, X_test, y_test)
        
    print("\n[*] QUÁ TRÌNH PHÂN TÍCH ĐÃ HOÀN TẤT! Bạn có thể xem các ảnh biểu đồ tại thư mục /results.")

if __name__ == '__main__':
    main()
