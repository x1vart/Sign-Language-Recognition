import os
import glob
import numpy as np
import cv2
import json
import shutil
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.config import DATA_PATH, ACTIONS, SEQUENCE_LENGTH
from src.utils import mediapipe_detection, extract_keypoints, mp_holistic

WLASL_MAPPING = {
    'hello': 'hello', 'how': 'how', 'are': 'are', 'you': 'you', 'do': 'do',
    'i': 'i', 'am': 'am', 'fine': 'fine', 'thank': 'thank', 'please': 'please',
    'can': 'can', 'help': 'help', 'me': 'me', 'love': 'love', 'very': 'very',
    'much': 'much', 'sorry': 'sorry', 'ok': 'ok'
}

def setup_wlasl_data():
    wlasl_base_dir = r"D:\download\cuoiki\Sign-Language-Recognition-main\dataset"
    json_path = os.path.join(wlasl_base_dir, "WLASL_v0.3.json")
    videos_dir = os.path.join(wlasl_base_dir, "videos")
    raw_dir = os.path.join('data', 'raw')
    
    if not os.path.exists(json_path): return
        
    print("[*] Đang tiến hành lấy dữ liệu từ WLASL dataset...")
    with open(json_path, 'r', encoding='utf-8') as f:
        wlasl_data = json.load(f)
        
    target_glosses = list(WLASL_MAPPING.values())
    video_to_action = {}
    
    for entry in wlasl_data:
        if entry['gloss'] in target_glosses:
            action = next(k for k, v in WLASL_MAPPING.items() if v == entry['gloss'])
            for inst in entry['instances']:
                video_to_action[inst['video_id']] = action
                
    copied_count = 0
    for video_id, action in video_to_action.items():
        src_video = os.path.join(videos_dir, f"{video_id}.mp4")
        if os.path.exists(src_video):
            action_raw_dir = os.path.join(raw_dir, action)
            os.makedirs(action_raw_dir, exist_ok=True)
            dest_video = os.path.join(action_raw_dir, f"{video_id}.mp4")
            if not os.path.exists(dest_video):
                shutil.copy2(src_video, dest_video)
                copied_count += 1
                
    if copied_count > 0:
        print(f"[*] Đã copy thành công {copied_count} file video vào data/raw/")

def extract_features_from_videos():
    raw_dir = os.path.join('data', 'raw')
    video_paths = glob.glob(os.path.join(raw_dir, '**', '*.mp4'), recursive=True)
    
    if len(video_paths) == 0:
        return
        
    print(f"\n[*] Bắt đầu trích xuất đặc trưng cho {len(video_paths)} video...")
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for idx, video_path in enumerate(video_paths, 1):
            action_name = os.path.basename(os.path.dirname(video_path))
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            
            action_processed_dir = os.path.join(DATA_PATH, action_name, base_name)
            
            if os.path.exists(action_processed_dir) and len(os.listdir(action_processed_dir)) == SEQUENCE_LENGTH:
                continue
                
            os.makedirs(action_processed_dir, exist_ok=True)
            
            cap = cv2.VideoCapture(video_path)
            frame_num = 0
            
            while cap.isOpened() and frame_num < SEQUENCE_LENGTH:
                ret, frame = cap.read()
                if not ret: break
                    
                image, results = mediapipe_detection(frame, holistic)
                keypoints = extract_keypoints(results)
                
                npy_path = os.path.join(action_processed_dir, str(frame_num))
                np.save(npy_path, keypoints)
                frame_num += 1
                
            cap.release()
            
            # Sửa lỗi Inhomogeneous Shape (Đổi 1662 thành 258)
            while frame_num < SEQUENCE_LENGTH:
                keypoints = np.zeros(258) 
                npy_path = os.path.join(action_processed_dir, str(frame_num))
                np.save(npy_path, keypoints)
                frame_num += 1
                
            print(f"[Xử lý {idx}/{len(video_paths)}] Đã xong: {action_name}/{base_name}.mp4")

def main():
    os.makedirs(os.path.join('data', 'raw'), exist_ok=True)
    os.makedirs(DATA_PATH, exist_ok=True)
    setup_wlasl_data()
    extract_features_from_videos()
    print("\n[*] Hoàn tất script auto_process!")

if __name__ == "__main__":
    main()