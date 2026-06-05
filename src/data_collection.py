import os
import cv2
import numpy as np
import sys
import time

# Thêm thư mục src vào path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.config import DATA_PATH, ACTIONS, NO_SEQUENCES, SEQUENCE_LENGTH
from src.utils import mediapipe_detection, draw_styled_landmarks, extract_keypoints, mp_holistic

def collect_data():
    """
    Hàm thu thập dữ liệu video từ webcam và lưu trữ dưới dạng file numpy.
    Đã được hoàn thiện đầy đủ.
    """
    # 1. Tạo cấu trúc thư mục
    for action in ACTIONS: 
        for sequence in range(NO_SEQUENCES):
            try: 
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except Exception as e:
                pass
                
    print(f"[*] Đã tạo xong cấu trúc thư mục tại {DATA_PATH}")

    # 2. Khởi tạo Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[LỖI] Không thể kết nối với Webcam.")
        return
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # 3. Thu thập dữ liệu
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in ACTIONS:
            print(f"\n---> CHUẨN BỊ THU THẬP TỪ: '{action}' <---")
            
            for sequence in range(NO_SEQUENCES):
                # Hiệu ứng tạm dừng 2 giây trước khi bắt đầu thu sequence mới
                for i in range(2, 0, -1):
                    ret, frame = cap.read()
                    frame = cv2.flip(frame, 1)
                    cv2.putText(frame, f"Chuan bi thu '{action}' Video {sequence}/{NO_SEQUENCES}", (120, 200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame, f"Bat dau sau {i} giay...", (120, 250), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow('Data Collection', frame)
                    cv2.waitKey(1000)
                    
                # Tiến hành thu thập đúng SEQUENCE_LENGTH frames
                for frame_num in range(SEQUENCE_LENGTH):
                    ret, frame = cap.read()
                    frame = cv2.flip(frame, 1)
                    
                    # Phát hiện
                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)
                    
                    # Hiển thị text trạng thái thu thập
                    cv2.putText(image, f"THU THAP: '{action}' | Video: {sequence} | Frame: {frame_num}", (15, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
                    cv2.imshow('Data Collection', image)
                    
                    # Trích xuất và lưu trữ keypoints
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)

                    # Thoát giữa chừng nếu bấm q
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        print("[*] Đã hủy thu thập.")
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                    
    cap.release()
    cv2.destroyAllWindows()
    print("\n[*] ĐÃ HOÀN TẤT THU THẬP DỮ LIỆU!")

if __name__ == '__main__':
    collect_data()
