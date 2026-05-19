import sys
import io

# Đảm bảo hiển thị đúng tiếng Việt trên Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import glob
import numpy as np
import cv2
import json
import shutil

# Import cấu hình
try:
    from src.config import DATA_PATH, ACTIONS
except ImportError:
    DATA_PATH = os.path.join('data', 'processed')
    ACTIONS = np.array(['hello', 'thank you', 'love', 'please', 'sorry'])

# Bản đồ ánh xạ từ action trong config sang từ khóa trong WLASL dataset
WLASL_MAPPING = {
    'hello': 'hello',
    'thank you': 'thank you',
    'love': 'love',
    'please': 'please',
    'sorry': 'sorry'
}

def process_frame(frame):
    """
    Hàm giả định trích xuất đặc trưng (MediaPipe).
    """
    return np.random.rand(1662)

def setup_wlasl_data():
    """
    Tự động đọc file WLASL JSON và copy các video thuộc 5 actions vào data/raw/
    """
    wlasl_base_dir = r"C:\Users\Xivart\Downloads\wlasl-processed"
    json_path = os.path.join(wlasl_base_dir, "WLASL_v0.3.json")
    videos_dir = os.path.join(wlasl_base_dir, "videos")
    
    raw_dir = os.path.join('data', 'raw')
    
    if not os.path.exists(json_path):
        print(f"[*] Không tìm thấy thư mục tải dataset tại {wlasl_base_dir}.")
        return
        
    print("[*] Đang đọc file mapping WLASL JSON...")
    with open(json_path, 'r', encoding='utf-8') as f:
        wlasl_data = json.load(f)
        
    # Lấy các từ khóa cần tìm
    target_glosses = list(WLASL_MAPPING.values())
    
    # Tìm ID video cho mỗi từ khóa
    video_to_action = {}
    for entry in wlasl_data:
        gloss = entry['gloss']
        if gloss in target_glosses:
            # Tìm action tương ứng với gloss
            action = next(k for k, v in WLASL_MAPPING.items() if v == gloss)
            for inst in entry['instances']:
                video_to_action[inst['video_id']] = action
                
    if not video_to_action:
        print("[*] Không tìm thấy video nào khớp với 5 từ khóa.")
        return
        
    print(f"[*] Đã tìm thấy {len(video_to_action)} video cho {len(ACTIONS)} hành động cốt lõi. Đang tiến hành copy vào data/raw/...")
    
    # Copy video
    copied_count = 0
    for video_id, action in video_to_action.items():
        src_video = os.path.join(videos_dir, f"{video_id}.mp4")
        if os.path.exists(src_video):
            # Tạo thư mục cho action trong data/raw/
            action_raw_dir = os.path.join(raw_dir, action)
            os.makedirs(action_raw_dir, exist_ok=True)
            
            dest_video = os.path.join(action_raw_dir, f"{video_id}.mp4")
            if not os.path.exists(dest_video):
                shutil.copy2(src_video, dest_video)
                copied_count += 1
                
    if copied_count > 0:
        print(f"[*] Đã tự động phân loại và copy thành công {copied_count} file video vào hệ thống.")
    else:
        print(f"[*] Các video đã được copy đầy đủ từ trước.")


def main():
    # 1. Khởi tạo đường dẫn
    raw_dir = os.path.join('data', 'raw')
    processed_dir = DATA_PATH
    
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    # 2. Tự động setup, phân loại và copy dữ liệu WLASL
    setup_wlasl_data()
    
    # 3. Quét tất cả video đã được phân loại trong data/raw/
    video_paths = glob.glob(os.path.join(raw_dir, '**', '*.mp4'), recursive=True)
    total_videos = len(video_paths)
    
    if total_videos == 0:
        print(f"[*] Không tìm thấy video .mp4 nào trong '{raw_dir}'.")
        return
        
    print(f"[*] Bắt đầu xử lý trích xuất đặc trưng cho {total_videos} video...")
    
    # 4. Xử lý video
    for idx, video_path in enumerate(video_paths, 1):
        try:
            # Tên file và thư mục cha (action)
            file_name = os.path.basename(video_path)
            base_name = os.path.splitext(file_name)[0]
            
            # Lấy tên thư mục cha để biết đây là hành động gì (VD: hello, thank you,...)
            # video_path có dạng: data/raw/hello/12345.mp4
            parent_dir = os.path.basename(os.path.dirname(video_path))
            
            # Nếu thư mục cha chính là raw_dir thì gán action mặc định, nếu không thì lấy tên thư mục
            if os.path.normpath(os.path.dirname(video_path)) == os.path.normpath(raw_dir):
                action_name = "unclassified"
            else:
                action_name = parent_dir
                
            # Tạo thư mục đích trong data/processed/
            action_processed_dir = os.path.join(processed_dir, action_name)
            os.makedirs(action_processed_dir, exist_ok=True)
            
            save_path = os.path.join(action_processed_dir, f"{base_name}.npy")
            
            if os.path.exists(save_path):
                percent = (idx / total_videos) * 100
                print(f"[Xử lý {idx}/{total_videos} video - Đạt {percent:.1f}%] Đã tồn tại: {action_name}/{base_name}.npy")
                continue
                
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("Không thể mở file video.")
                
            frames_data = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                features = process_frame(frame)
                frames_data.append(features)
                
            cap.release()
            
            if len(frames_data) == 0:
                raise Exception("Video không có khung hình.")
                
            np.save(save_path, np.array(frames_data))
            
            percent = (idx / total_videos) * 100
            print(f"[Xử lý {idx}/{total_videos} video - Đạt {percent:.1f}%] Thành công: {action_name}/{file_name}")
            
        except Exception as e:
            print(f"\n[LỖI] Xử lý video {video_path} thất bại: {e}\n")
            continue

if __name__ == "__main__":
    main()
