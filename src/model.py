import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Flatten
from src.config import ACTIONS

def build_model(model_type='LSTM', input_shape=(30, 1662)):
    """
    Hàm định nghĩa kiến trúc mạng nơ-ron để thử nghiệm nhiều model khác nhau.
    Hỗ trợ: 'LSTM', 'GRU', 'Dense'
    """
    model = Sequential()
    
    if model_type == 'LSTM':
        model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=input_shape))
        model.add(LSTM(128, return_sequences=True, activation='relu'))
        model.add(LSTM(64, return_sequences=False, activation='relu'))
    elif model_type == 'GRU':
        model.add(GRU(64, return_sequences=True, activation='relu', input_shape=input_shape))
        model.add(GRU(128, return_sequences=True, activation='relu'))
        model.add(GRU(64, return_sequences=False, activation='relu'))
    elif model_type == 'Dense':
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
    else:
        raise ValueError("Loại mô hình không được hỗ trợ. Hãy chọn 'LSTM', 'GRU', hoặc 'Dense'.")
        
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    # Lớp đầu ra tương ứng với số lượng hành động
    model.add(Dense(ACTIONS.shape[0], activation='softmax'))
    
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    
    return model

def get_available_models():
    """
    Trả về danh sách các kiến trúc mô hình hỗ trợ
    """
    return ['LSTM', 'GRU', 'Dense']

if __name__ == '__main__':
    for m in get_available_models():
        print(f"\n--- Kiến trúc: {m} ---")
        model = build_model(model_type=m)
        model.summary()
