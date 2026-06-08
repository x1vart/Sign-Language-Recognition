import cv2
import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
            mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
        ) 
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
            mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
            mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
        ) 
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        ) 

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    # 258 features
    return np.concatenate([pose, lh, rh])

def normalize_frame(frame):
    """
    Chuẩn hoá toạ độ bằng cách dịch tâm về vị trí mũi (landmark 0).
    Giúp model không bị ảnh hưởng bởi vị trí đứng của người trong khung hình.
    """
    new_frame = frame.copy()
    # Nếu không có mũi (giá trị 0,0) thì bỏ qua
    if new_frame[0] == 0 and new_frame[1] == 0:
        return new_frame
        
    nose_x, nose_y = new_frame[0], new_frame[1]
    
    # Chuẩn hóa Pose (0-132) - nhảy bước 4 (x, y, z, v)
    for i in range(0, 132, 4):
        if new_frame[i] != 0 or new_frame[i+1] != 0:
            new_frame[i] -= nose_x
            new_frame[i+1] -= nose_y
            
    # Chuẩn hóa Left Hand (132-195) - nhảy bước 3 (x, y, z)
    for i in range(132, 195, 3):
        if new_frame[i] != 0 or new_frame[i+1] != 0:
            new_frame[i] -= nose_x
            new_frame[i+1] -= nose_y
            
    # Chuẩn hóa Right Hand (195-258) - nhảy bước 3 (x, y, z)
    for i in range(195, 258, 3):
        if new_frame[i] != 0 or new_frame[i+1] != 0:
            new_frame[i] -= nose_x
            new_frame[i+1] -= nose_y
            
    return new_frame