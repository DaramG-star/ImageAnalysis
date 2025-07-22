import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# 🔧 1. 모델 경로를 현재 파일 기준으로 안전하게 설정
base_dir = os.path.dirname(os.path.abspath(__file__))  # 이 파일(test_model.py)의 절대경로
model_path = os.path.join(base_dir, "gesture_model4.h5")

# ✅ 2. 모델 로드
model = tf.keras.models.load_model(model_path)
label_names = ['bad', 'fist', 'good', 'gun', 'heart', 'none', 'ok', 'open_palm', 'promise', 'rock', 'victory']

# ✅ 3. MediaPipe 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# ✅ 4. 웹캠 시작
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            coords = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            base_x, base_y, base_z = coords[0]
            relative = []
            for x, y, z in coords:
                relative.extend([x - base_x, y - base_y, z - base_z])

            if len(relative) == 63:
                input_data = np.array(relative).reshape(1, -1)  # shape: (1, 63)
                predictions = model.predict(input_data, verbose=0)
                predicted_idx = np.argmax(predictions)
                predicted_label = label_names[predicted_idx]
                confidence = np.max(predictions)

                cv2.putText(frame, f"{predicted_label} ({confidence:.2f})", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)

    cv2.imshow("Hand Gesture Prediction", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
