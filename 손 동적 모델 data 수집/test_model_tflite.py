import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os
import time

model_path = "moving_hands/gesture_rnn_model1.tflite"

# 이동 관련 설정
prev_wrist = None
move_threshold = 0.005
confidence_threshold = 0.8

# 제스처 라벨
gestures = ['fire', 'hi', 'hit', 'none', 'nono', 'nyan', 'shot']

# TFLite 모델 로드
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# MediaPipe 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# 이펙트 이미지
fire_img = cv2.imread('moving_hands/fire_effect.png', cv2.IMREAD_UNCHANGED)
overlay_dict = {
    'fire': cv2.resize(fire_img, (100, 100)) if fire_img is not None else None
}

# 키포인트 추출 함수
def extract_keypoints(results):
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        wrist = hand.landmark[0]
        return np.array([[lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z] for lm in hand.landmark]).flatten()
    return np.zeros(21 * 3)

# 이미지 오버레이 함수
def overlay_image(bg, overlay, x, y):
    h, w = overlay.shape[:2]
    if x < 0 or y < 0 or x + w > bg.shape[1] or y + h > bg.shape[0]:
        return bg
    alpha = overlay[:, :, 3] / 255.0
    for c in range(3):
        bg[y:y+h, x:x+w, c] = (1 - alpha) * bg[y:y+h, x:x+w, c] + alpha * overlay[:, :, c]
    return bg

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
        hand = result.multi_hand_landmarks[0]
        wrist = hand.landmark[0]

        # 이동량 계산
        if prev_wrist is not None:
            dx = wrist.x - prev_wrist.x
            dy = wrist.y - prev_wrist.y
            dz = wrist.z - prev_wrist.z
            move_distance = (dx**2 + dy**2 + dz**2) ** 0.5

            if move_distance < move_threshold:
                cv2.putText(img, "No movement", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
                prev_wrist = wrist
                cv2.imshow('TFLite Gesture Recognition', img)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                continue

        prev_wrist = wrist  # 현재 손 위치 저장

        mp_drawing.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)
        keypoints = extract_keypoints(result).astype(np.float32)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            input_data = np.expand_dims(sequence, axis=0).astype(np.float32)

            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])

            predicted_label = gestures[np.argmax(output)]
            confidence = np.max(output)

            # confidence가 낮으면 none으로 처리
            if confidence < confidence_threshold:
                predicted_label = 'none'

            # 결과 표시
            cv2.putText(img, f'{predicted_label} ({confidence:.2f})', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # fire 효과 오버레이
            if predicted_label == 'fire' and overlay_dict['fire'] is not None:
                h, w, _ = img.shape
                cx, cy = int(wrist.x * w), int(wrist.y * h)
                img = overlay_image(img, overlay_dict['fire'], cx - 50, cy - 150)

    cv2.imshow('TFLite Gesture Recognition', img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
