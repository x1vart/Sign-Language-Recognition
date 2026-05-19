import cv2
import numpy as np
import mediapipe as mp

# TODO: Khởi tạo các module của MediaPipe (ví dụ: mp.solutions.holistic, mp.solutions.drawing_utils)
mp_holistic = None
mp_drawing = None

def mediapipe_detection(image, model):
    """
    Hàm xử lý ảnh đầu vào và trả về kết quả dự đoán từ MediaPipe.
    Yêu cầu:
    1. Chuyển đổi màu ảnh từ BGR sang RGB.
    2. Khóa cờ ghi của ảnh để tối ưu, rồi đưa qua model dự đoán.
    3. Chuyển đổi lại từ RGB sang BGR.
    """
    # TODO: Viết code ở đây
    results = None
    return image, results

def draw_styled_landmarks(image, results):
    """
    Hàm vẽ các điểm landmark lên khung hình.
    Yêu cầu: Vẽ các điểm kết nối cho khuôn mặt, tư thế (pose), bàn tay trái và phải.
    Gợi ý: Dùng mp_drawing.draw_landmarks.
    """
    # TODO: Viết code ở đây
    pass

def extract_keypoints(results):
    """
    Hàm trích xuất tọa độ của các keypoints và làm phẳng (flatten) thành vector 1D.
    Yêu cầu:
    1. Lấy pose_landmarks, face_landmarks, left_hand_landmarks, right_hand_landmarks.
    2. Nếu điểm nào không nhận diện được, tạo mảng chứa toàn số 0 (zeros) với kích thước tương ứng.
    3. Trả về mảng numpy dài (khoảng 1662 phần tử) gồm tất cả các keypoints ghép lại.
    """
    # TODO: Viết code ở đây
    return np.array([])
