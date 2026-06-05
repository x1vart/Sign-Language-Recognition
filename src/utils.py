import cv2
import numpy as np
import mediapipe as mp

# Khởi tạo MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    """
    Chuyển đổi màu và đưa qua mô hình MediaPipe để trích xuất đặc trưng
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Chuyển BGR sang RGB
    image.flags.writeable = False                  # Khóa ảnh để tăng tốc
    results = model.process(image)                 # Dự đoán
    image.flags.writeable = True                   # Mở khóa ảnh
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Chuyển RGB về BGR
    return image, results

def draw_styled_landmarks(image, results):
    """
    Vẽ các khung xương lên ảnh hiển thị
    """
    # Vẽ mặt (Face)
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
            mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
            mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
        ) 
    # Vẽ cơ thể (Pose)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
            mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
        ) 
    # Vẽ tay trái (Left Hand)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
            mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
            mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
        ) 
    # Vẽ tay phải (Right Hand)  
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        ) 

def extract_keypoints(results):
    """
    Trích xuất tọa độ keypoints và làm phẳng thành vector 1D
    Tổng kích thước: 33*4 (pose) + 468*3 (face) + 21*3 (lh) + 21*3 (rh) = 1662
    """
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    return np.concatenate([pose, face, lh, rh])
