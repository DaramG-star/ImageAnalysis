import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# 제스처 라벨 순서 (모델 학습할 때 사용한 것과 동일하게)
gestures = ['fire', 'shot', 'nono', 'hit', 'hi', 'nyan']

# 모델 불러오기
model = load_model('./손 동적 모델 data 수집/gesture_rnn_model.h5')

# MediaPipe 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# 키포인트 추출 함수
def extract_keypoints(results):
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        return np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark]).flatten()
    else:
        return np.zeros(21 * 3)

# 웹캠 시작
cap = cv2.VideoCapture(0)
sequence = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        mp_drawing.draw_landmarks(img, result.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        keypoints = extract_keypoints(result)
        sequence.append(keypoints)
        sequence = sequence[-30:]  # 최근 30프레임만 유지

        if len(sequence) == 30:
            input_data = np.expand_dims(sequence, axis=0)  # shape: (1, 30, 63)
            prediction = model.predict(input_data, verbose=0)[0]
            predicted_label = gestures[np.argmax(prediction)]
            confidence = np.max(prediction)

            # 화면에 표시
            cv2.putText(img, f'{predicted_label} ({confidence:.2f})', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Real-Time Gesture Recognition', img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
