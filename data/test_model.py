import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# 1. 모델 로드
model = tf.keras.models.load_model(r"C:\Users\SSAFY\Desktop\daramstudy\imageanalysis\data\gesture_model4.h5")
label_names = ['bad', 'fist', 'good', 'gun', 'heart', 'none', 'ok', 'open_palm', 'promise', 'rock', 'victory']


# 2. MediaPipe 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# 3. 웹캠 시작
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
            # 손 랜드마크 그리기
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 좌표 → 상대좌표 변환
            coords = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            base_x, base_y, base_z = coords[0]
            relative = []
            for x, y, z in coords:
                relative.extend([x - base_x, y - base_y, z - base_z])

            if len(relative) == 63:
                # 예측
                input_data = np.array(relative).reshape(1, -1)  # (1, 63)
                predictions = model.predict(input_data, verbose=0)
                predicted_idx = np.argmax(predictions)
                predicted_label = label_names[predicted_idx]
                confidence = np.max(predictions)

                # 화면에 표시
                cv2.putText(frame, f"{predicted_label} ({confidence:.2f})", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)

    cv2.imshow("Hand Gesture Prediction", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
