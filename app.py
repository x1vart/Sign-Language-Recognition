import cv2
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.config import ACTIONS, SEQUENCE_LENGTH, MODELS_PATH
from src.model import build_model, get_available_models
from src.utils import mediapipe_detection, draw_styled_landmarks, extract_keypoints, mp_holistic, normalize_frame

def draw_beautiful_ui(image, sentence, current_model_name, available_models):
    h, w, _ = image.shape
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (40, 40, 40), -1)
    cv2.rectangle(overlay, (0, h - 100), (w, h), (30, 30, 30), -1)
    image = cv2.addWeighted(overlay, 0.75, image, 0.25, 0)
    
    cv2.putText(image, "MODELS (Bam 1-3 de chuyen):", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    for idx, model_name in enumerate(available_models):
        color = (0, 255, 0) if model_name == current_model_name else (150, 150, 150)
        thickness = 2 if model_name == current_model_name else 1
        cv2.putText(image, f"{idx+1}.{model_name}", (20 + idx * 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness, cv2.LINE_AA)
        
    display_text = " ".join(sentence) if sentence else "Dang cho tin hieu ngon ngu ky hieu..."
    cv2.putText(image, "Cau dich:", (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(image, display_text, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, "Bam 'C' de xoa cau | Bam 'Q' de thoat", (w - 380, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
    return image

def run_app():
    available_models = get_available_models()
    current_model_idx = 0
    models = {}
    for m_name in available_models:
        model = build_model(model_type=m_name, input_shape=(SEQUENCE_LENGTH, 258))
        weight_path = os.path.join(MODELS_PATH, f'{m_name.lower()}_model.h5')
        if os.path.exists(weight_path): model.load_weights(weight_path)
        models[m_name] = model

    sequence, sentence, predictions = [], [], []
    threshold = 0.65 

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)
            
            sequence.append(extract_keypoints(results))
            sequence = sequence[-SEQUENCE_LENGTH:]
            current_model = models[available_models[current_model_idx]]
            
            if len(sequence) == SEQUENCE_LENGTH:
                # KIỂM TRA: Nếu không có tay trong khung hình -> Reset và bỏ qua dự đoán để chống nhiễu từ 'late'
                if not (results.left_hand_landmarks or results.right_hand_landmarks):
                    sequence = []
                    predictions = []
                    image = draw_beautiful_ui(image, sentence, available_models[current_model_idx], available_models)
                    cv2.imshow('Sign Language Translation App', image)
                    if cv2.waitKey(10) & 0xFF == ord('q'): break
                    continue

                # CHUẨN HOÁ DỮ LIỆU ĐỂ DỰ ĐOÁN CHÍNH XÁC
                normalized_sequence = [normalize_frame(frame) for frame in sequence]
                res = current_model.predict(np.expand_dims(normalized_sequence, axis=0), verbose=0)[0]
                pred_idx = np.argmax(res)
                predictions.append(pred_idx)
                
                print(f"[*] AI đang đoán: {ACTIONS[pred_idx]:<10} | Độ tự tin: {res[pred_idx]:.2f}")
                
                if len(predictions) >= 10 and predictions[-10:].count(pred_idx) >= 6:
                    if res[pred_idx] > threshold:
                        action = ACTIONS[pred_idx]
                        if action != 'idle':
                            if len(sentence) == 0 or action != sentence[-1]:
                                sentence.append(action)
                        sequence = []
                        predictions = []
                            
                if len(sentence) > 5: sentence = sentence[-5:]
            
            image = draw_beautiful_ui(image, sentence, available_models[current_model_idx], available_models)
            cv2.imshow('Sign Language Translation App', image)
            
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'): break
            elif key == ord('c'): sentence = []
            elif key == ord('1'): current_model_idx = 0
            elif key == ord('2') and len(available_models) >= 2: current_model_idx = 1
            elif key == ord('3') and len(available_models) >= 3: current_model_idx = 2

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    os.makedirs(MODELS_PATH, exist_ok=True)
    run_app()