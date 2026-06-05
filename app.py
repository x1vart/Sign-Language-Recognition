import cv2
import numpy as np
import os
import sys

# Thêm đường dẫn src vào PYTHONPATH nếu muốn chạy app.py ở thư mục gốc
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.config import ACTIONS, SEQUENCE_LENGTH, MODELS_PATH
from src.model import build_model, get_available_models
from src.utils import mediapipe_detection, draw_styled_landmarks, extract_keypoints, mp_holistic

def draw_beautiful_ui(image, sentence, current_model_name, available_models):
    """
    Vẽ giao diện (Overlay) lên ảnh chứa thông tin các models và câu dịch được.
    """
    h, w, _ = image.shape
    
    # Tạo overlay mờ để UI nhìn rõ hơn
    overlay = image.copy()
    
    # Thanh hiển thị model ở trên cùng
    cv2.rectangle(overlay, (0, 0), (w, 60), (40, 40, 40), -1)
    
    # Thanh hiển thị câu dịch ở dưới cùng
    cv2.rectangle(overlay, (0, h - 100), (w, h), (30, 30, 30), -1)
    
    # Áp dụng overlay với độ trong suốt (alpha)
    alpha = 0.75
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    
    # Text: Chọn Model
    cv2.putText(image, "MODELS (Bam 1-3 de chuyen):", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    for idx, model_name in enumerate(available_models):
        color = (0, 255, 0) if model_name == current_model_name else (150, 150, 150)
        thickness = 2 if model_name == current_model_name else 1
        cv2.putText(image, f"{idx+1}.{model_name}", (20 + idx * 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness, cv2.LINE_AA)
        
    # Text: Câu dịch
    display_text = " ".join(sentence)
    if not display_text:
        display_text = "Dang cho tin hieu ngon ngu ky hieu..."
        
    cv2.putText(image, "Cau dich:", (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(image, display_text, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Nút hướng dẫn
    cv2.putText(image, "Bam 'C' de xoa cau | Bam 'Q' de thoat", (w - 380, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
    
    return image

def run_app():
    # 1. Tải danh sách models và khởi tạo
    available_models = get_available_models()
    current_model_idx = 0
    
    models = {}
    print("[*] Đang khởi tạo các mô hình mạng...")
    for m_name in available_models:
        model = build_model(model_type=m_name)
        # Cố gắng load trọng số nếu có sẵn
        weight_path = os.path.join(MODELS_PATH, f'{m_name.lower()}_model.h5')
        if os.path.exists(weight_path):
            try:
                model.load_weights(weight_path)
                print(f"[+] Đã load trọng số thành công cho mô hình: {m_name}")
            except Exception as e:
                print(f"[-] Không thể load trọng số cho {m_name}: {e}")
        else:
            print(f"[-] Chưa có file train ({weight_path}) cho mô hình {m_name}. Mô hình sẽ dự đoán ngẫu nhiên.")
        models[m_name] = model

    # 2. Biến trạng thái để theo dõi chuỗi, câu và dự đoán
    sequence = []      # Lưu trữ tọa độ của 30 frames gần nhất
    sentence = []      # Lưu trữ câu dịch (danh sách các từ)
    predictions = []   # Lưu trữ lịch sử dự đoán để ổn định
    threshold = 0.7    # Ngưỡng tin cậy (Confidence)

    # 3. Mở Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[LỖI] Không thể kết nối với Webcam. Hãy kiểm tra lại camera của bạn.")
        return

    # Tùy chỉnh độ phân giải camera (HD)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n[*] Ứng dụng đã sẵn sàng. Đang mở Webcam...")

    # Sử dụng MediaPipe Holistic
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Lật ảnh theo trục dọc giống như gương để dễ thao tác
            frame = cv2.flip(frame, 1)

            # Đưa qua MediaPipe nhận diện
            image, results = mediapipe_detection(frame, holistic)
            
            # Vẽ các điểm landmarks lên cơ thể, tay, mặt
            draw_styled_landmarks(image, results)
            
            # Trích xuất keypoints và đưa vào chuỗi (sequence)
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-SEQUENCE_LENGTH:] # Giữ lại tối đa 30 frames gần nhất
            
            # Lấy mô hình đang được chọn hiện tại
            current_model_name = available_models[current_model_idx]
            current_model = models[current_model_name]
            
            # Tiến hành dự đoán khi đã thu thập đủ 30 frames
            if len(sequence) == SEQUENCE_LENGTH:
                # Đưa vào model (yêu cầu batch_size=1 nên dùng np.expand_dims)
                res = current_model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                pred_idx = np.argmax(res)
                predictions.append(pred_idx)
                
                # Logic: Chỉ chấp nhận hành động nếu 10 frame gần nhất có chung dự đoán
                # (Chống nhiễu do tay di chuyển qua lại)
                if len(predictions) >= 10 and len(set(predictions[-10:])) == 1:
                    if res[pred_idx] > threshold:
                        action = ACTIONS[pred_idx]
                        
                        # Chỉ thêm vào câu nếu hành động này khác với hành động vừa được thêm trước đó
                        if len(sentence) > 0:
                            if action != sentence[-1]:
                                sentence.append(action)
                        else:
                            sentence.append(action)
                            
                # Giới hạn hiển thị 5 từ gần nhất để câu không quá dài tràn màn hình
                if len(sentence) > 5: 
                    sentence = sentence[-5:]
            
            # 4. Vẽ UI và hiển thị text lên màn hình
            image = draw_beautiful_ui(image, sentence, current_model_name, available_models)
            
            cv2.imshow('Sign Language Translation App', image)
            
            # 5. Xử lý thao tác bàn phím
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):    # Thoát
                break
            elif key == ord('c'):  # Xóa câu dịch
                sentence = []
            elif key == ord('1'):  # Chọn mô hình 1 (LSTM)
                current_model_idx = 0
            elif key == ord('2') and len(available_models) >= 2:  # Chọn mô hình 2 (GRU)
                current_model_idx = 1
            elif key == ord('3') and len(available_models) >= 3:  # Chọn mô hình 3 (Dense)
                current_model_idx = 2

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # Tạo thư mục models nếu chưa có
    os.makedirs(MODELS_PATH, exist_ok=True)
    run_app()
