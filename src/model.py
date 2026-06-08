import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Flatten, Dropout
from src.config import ACTIONS, SEQUENCE_LENGTH

def build_model(model_type='LSTM', input_shape=(SEQUENCE_LENGTH, 258)):
    model = Sequential()
    
    if model_type == 'LSTM':
        model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(128, return_sequences=True, activation='relu'))
        model.add(Dropout(0.2))
        model.add(LSTM(64, return_sequences=False, activation='relu'))
    elif model_type == 'GRU':
        model.add(GRU(64, return_sequences=True, activation='relu', input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(GRU(128, return_sequences=True, activation='relu'))
        model.add(Dropout(0.2))
        model.add(GRU(64, return_sequences=False, activation='relu'))
    elif model_type == 'Dense':
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(128, activation='relu'))
    else:
        raise ValueError("Loại mô hình không được hỗ trợ.")
        
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(ACTIONS.shape[0], activation='softmax'))
    
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model

def get_available_models():
    return ['LSTM', 'GRU', 'Dense']

if __name__ == '__main__':
    for m in get_available_models():
        print(f"\n--- Kiến trúc: {m} ---")
        model = build_model(model_type=m)
        model.summary()