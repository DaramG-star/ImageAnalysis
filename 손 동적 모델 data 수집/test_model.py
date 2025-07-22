import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import time  # ⬅️ shot 지속시간 체크용

# 제스처 라벨 순서
gestures = ['fire', 'hi', 'hit', 'none', 'nono', 'nyan', 'shot']

# 모델 로드
model = load_model('./손 동적 모델 data 수집/gesture_rnn_model.h5')

# MediaPipe 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# 이미지 불러오기
fire_img = cv2.imread(r'C:/Users/SSAFY/Desktop/daramstudy/ImageAnalysis/fire_effect.png', cv2.IMREAD_UNCHANGED)
paw_img = cv2.imread(r'C:/Users/SSAFY/Desktop/daramstudy/ImageAnalysis/cat_paw_real.png', cv2.IMREAD_UNCHANGED)
heart_img = cv2.imread('heart.png', cv2.IMREAD_UNCHANGED)
crack_img = cv2.imread('screen_crack.png', cv2.IMREAD_UNCHANGED)
hi_img = cv2.imread('hi.png', cv2.IMREAD_UNCHANGED)
fist_img = cv2.imread('fist.png', cv2.IMREAD_UNCHANGED)

# 상태 변수
prev_label = None
shot_start_time = None  # shot 시작 시간 기록

# 오버레이 함수
def overlay_image(bg, overlay, x, y):
    h, w = overlay.shape[:2]
    if x < 0 or y < 0 or x + w > bg.shape[1] or y + h > bg.shape[0]:
        return bg
    alpha_overlay = overlay[:, :, 3] / 255.0
    for c in range(3):
        bg[y:y+h, x:x+w, c] = (1 - alpha_overlay) * bg[y:y+h, x:x+w, c] + alpha_overlay * overlay[:, :, c]
    return bg

# 키포인트 추출
def extract_keypoints(results):
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        wrist = hand.landmark[0]
        return np.array([[lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z] for lm in hand.landmark]).flatten()
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
        sequence = sequence[-30:]

        if len(sequence) == 30:
            input_data = np.expand_dims(sequence, axis=0)
            prediction = model.predict(input_data, verbose=0)[0]
            predicted_label = gestures[np.argmax(prediction)]
            confidence = np.max(prediction)

            # 결과 표시
            cv2.putText(img, f'{predicted_label} ({confidence:.2f})', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            h, w, _ = img.shape

            # 🔥 fire
            if predicted_label == 'fire':
                hand = result.multi_hand_landmarks[0]
                wrist = hand.landmark[0]
                cx, cy = int(wrist.x * w), int(wrist.y * h)
                resized_fire = cv2.resize(fire_img, (100, 100))
                img = overlay_image(img, resized_fire, cx - 50, cy - 150)

            # 🐾 nyan
            elif predicted_label == 'nyan' and paw_img is not None:
                hand = result.multi_hand_landmarks[0]
                wrist = hand.landmark[0]
                cx, cy = int(wrist.x * w), int(wrist.y * h)
                resized_paw = cv2.resize(paw_img, (100, 100))
                img = overlay_image(img, resized_paw, cx + 60, cy - 50)

            # hit
            elif predicted_label == 'hit' and fist_img is not None:
                hand = result.multi_hand_landmarks[0]
                wrist = hand.landmark[0]
                cx, cy = int(wrist.x * w), int(wrist.y * h)

                resized_fist = cv2.resize(fist_img, (60, 60))

                offsets = [(-80, -80), (0, -80), (80, -80),
                        (-80, 0),  (0, 0),   (80, 0),
                        (-80, 80), (0, 80),  (80, 80)]

                for dx, dy in offsets:
                    img = overlay_image(img, resized_fist, cx + dx, cy + dy)

            # 🖐️ hi
            if predicted_label == 'hi':
                hand = result.multi_hand_landmarks[0]
                wrist = hand.landmark[0]
                cx, cy = int(wrist.x * w), int(wrist.y * h)

                resized_hi = cv2.resize(hi_img, (100, 100))
                img = overlay_image(img, resized_hi, cx - 50, cy - 150)


            # 🔫 shot
            if predicted_label == 'shot':
                if prev_label != 'shot':
                    shot_start_time = time.time()

                elapsed = time.time() - shot_start_time if shot_start_time else 0
                hand = result.multi_hand_landmarks[0]
                index_tip = hand.landmark[8]
                cx, cy = int(index_tip.x * w), int(index_tip.y * h)

                if heart_img is not None and elapsed > 0:
                    resized_heart = cv2.resize(heart_img, (40, 40))
                    img = overlay_image(img, resized_heart, cx, cy - 60)

                if crack_img is not None and elapsed >= 4.0:
                    resized_crack = cv2.resize(crack_img, (w, h))
                    img = overlay_image(img, resized_crack, 0, 0)

            else:
                shot_start_time = None  # shot 아닌 제스처 되면 초기화

            prev_label = predicted_label  # 마지막에 저장

            

    cv2.imshow('Real-Time Gesture Recognition', img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
