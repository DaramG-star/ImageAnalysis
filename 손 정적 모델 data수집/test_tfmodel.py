import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# 제스처 라벨
label_names = ['bad', 'fist', 'good', 'gun', 'heart', 'none', 'ok', 'open_palm', 'promise', 'rock', 'victory']

# TFLite 모델 로드
interpreter = tf.lite.Interpreter(model_path="stop_hands/gesture_model4.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# MediaPipe 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# 입력 변환 함수 (정규화된 좌표를 상대 좌표로 변환)
def extract_keypoints(hand_landmarks):
    wrist = hand_landmarks.landmark[0]
    keypoints = []
    for lm in hand_landmarks.landmark:
        keypoints.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
    return np.array(keypoints, dtype=np.float32)

# 웹캠 시작
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)

            keypoints = extract_keypoints(hand).reshape(1, -1)  # (1, 63)

            # 입력 데이터 설정
            interpreter.set_tensor(input_details[0]['index'], keypoints)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            predicted_idx = int(np.argmax(output_data))
            predicted_label = label_names[predicted_idx]
            confidence = float(np.max(output_data))

            # 화면에 출력
            cv2.putText(img, f"{predicted_label} ({confidence:.2f})", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Static Gesture (TFLite)", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
