import os
import cv2
import numpy as np
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.config import DATA_PATH, ACTIONS, NO_SEQUENCES, SEQUENCE_LENGTH
from src.utils import mediapipe_detection, draw_styled_landmarks, extract_keypoints, mp_holistic

def collect_data(target_actions=None):
    actions_to_collect = target_actions if target_actions else ACTIONS
    # 1. Tạo trước cây thư mục
    for action in actions_to_collect: 
        for sequence in range(NO_SEQUENCES):
            try: os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except: pass

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in actions_to_collect:
            print(f"\n---> KIỂM TRA DỮ LIỆU TỪ: '{action}' <---")
            for sequence in range(NO_SEQUENCES):
                
                # --- TÍNH NĂNG TẠM DỪNG & TIẾP TỤC (RESUME) ---
                sequence_path = os.path.join(DATA_PATH, action, str(sequence))
                if os.path.exists(sequence_path):
                    # Đếm số lượng file .npy trong thư mục
                    files_in_seq = [f for f in os.listdir(sequence_path) if f.endswith('.npy')]
                    
                    # Nếu đã có đủ 30 khung hình, tự động bỏ qua video này
                    if len(files_in_seq) == SEQUENCE_LENGTH:
                        print(f"[*] Bỏ qua: '{action}' - Video {sequence}/{NO_SEQUENCES} (Đã thu thập đủ)")
                        continue # Lệnh continue giúp vòng lặp bỏ qua các bước dưới và nhảy sang sequence tiếp theo
                # -----------------------------------------------

                # Nếu chưa có đủ dữ liệu, bật đếm ngược để thu thập
                for i in range(2, 0, -1):
                    ret, frame = cap.read()
                    frame = cv2.flip(frame, 1)
                    cv2.putText(frame, f"Chuan bi thu '{action}' Video {sequence}/{NO_SEQUENCES}", (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame, f"Bat dau sau {i} giay...", (120, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow('Data Collection', frame)
                    cv2.waitKey(1000)
                    
                # Tiến hành ghi hình đủ SEQUENCE_LENGTH (30 frames)
                for frame_num in range(SEQUENCE_LENGTH):
                    ret, frame = cap.read()
                    frame = cv2.flip(frame, 1)
                    
                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)
                    
                    cv2.putText(image, f"THU THAP: '{action}' | Video: {sequence} | Frame: {frame_num}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
                    cv2.imshow('Data Collection', image)
                    
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)
                    
                    # BẤM Q ĐỂ LƯU TRẠNG THÁI VÀ THOÁT
                    if cv2.waitKey(10) & 0xFF == ord('q'): 
                        print("\n[*] Đã tạm dừng! Lần sau chạy lại lệnh, hệ thống sẽ tự động tiếp tục từ vị trí này.")
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                        
    cap.release()
    cv2.destroyAllWindows()
    print("\n[*] ĐÃ HOÀN TẤT THU THẬP TOÀN BỘ DỮ LIỆU CỦA CÁC TỪ!")

if __name__ == '__main__':
    args = sys.argv[1:]
    target_actions = args if len(args) > 0 else None
    if target_actions:
        print(f"[*] Chế độ chỉ thu thập các từ: {target_actions}")
    collect_data(target_actions)