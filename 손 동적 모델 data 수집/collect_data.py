import cv2
import mediapipe as mp
import numpy as np
import os
import time

# MediaPipe 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# 저장 경로 및 제스처 목록
DATA_PATH = os.path.join(os.getcwd(), 'gesture_data')
gestures = ['none', 'nono', 'hit', 'nyan']
sequence_length = 30  # 1 sequence = 30프레임
record_time = 300  # 제스처당 수집 시간 (초)

# 키포인트 추출 함수
def extract_keypoints(results):
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        wrist = hand.landmark[0]
        return np.array([[
            lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z
        ] for lm in hand.landmark]).flatten()
    else:
        return np.zeros(21 * 3)


# 비디오 캡처
cap = cv2.VideoCapture(0)
os.makedirs(DATA_PATH, exist_ok=True)

for label in gestures:
    print(f"\n🖐️ 제스처 '{label}' 수집 시작합니다! 5초 뒤 시작...")
    time.sleep(5)

    data = []
    frame_buffer = []
    start = time.time()

    while time.time() - start < record_time:
        ret, frame = cap.read()
        if not ret:
            break
        img = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            keypoints = extract_keypoints(result)
            frame_buffer.append(keypoints)
            mp_drawing.draw_landmarks(img, result.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

            if len(frame_buffer) == sequence_length:
                data.append(frame_buffer)
                frame_buffer = []
                print(f"  ⏺️ {label} sequence 저장됨 (총 {len(data)})")

        # 화면에 표시
        cv2.putText(img, f"Collecting: {label}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Data Collection', img)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    save_path = os.path.join(DATA_PATH, f'{label}.npy')

    if os.path.exists(save_path):
        existing_data = np.load(save_path)
        data = np.concatenate((existing_data, data), axis=0)

    np.save(save_path, np.array(data))
    print(f"✅ '{label}' 수집 완료! 총 {len(data)}개 sequence 저장됨")


cap.release()
cv2.destroyAllWindows()
