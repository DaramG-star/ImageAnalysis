import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import time
import os

# 제스처 라벨과 이미지 매핑
label_names = ['bad', 'fist', 'good', 'gun', 'heart', 'none', 'ok', 'open_palm', 'promise', 'rock', 'victory']
label_to_image = {
    'bad': 'stop_hands/bad.png',
    'good': 'stop_hands/good.png',
    'gun': 'stop_hands/gun.png',
    'heart': 'stop_hands/heart.png',
    'ok': 'stop_hands/ok.png',
    'promise': 'stop_hands/promise.png',
    'rock': 'stop_hands/rock.png',
    'victory': 'stop_hands/victory.png'
}

# 이미지 미리 로드 (알파채널 포함)
loaded_images = {}
for label, path in label_to_image.items():
    if os.path.exists(path):
        loaded_images[label] = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # 알파채널 포함

# TFLite 모델 로드
interpreter = tf.lite.Interpreter(model_path="stop_hands/gesture_model7.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# MediaPipe 설정 (뼈대 안 그릴거니까 drawing_utils 필요 X)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

def extract_keypoints(hand_landmarks):
    wrist = hand_landmarks.landmark[0]
    keypoints = []
    for lm in hand_landmarks.landmark:
        keypoints.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
    return np.array(keypoints, dtype=np.float32)

def overlay_transparent(background, overlay, x, y):
    """알파채널 있는 이미지를 합성"""
    bh, bw = background.shape[:2]
    h, w = overlay.shape[:2]

    if x >= bw or y >= bh:
        return background

    if x + w > bw:
        w = bw - x
        overlay = overlay[:, :w]

    if y + h > bh:
        h = bh - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        return background

    overlay_img = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_img
    return background

cap = cv2.VideoCapture(0)

last_label = None
label_start_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    current_label = None

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            keypoints = extract_keypoints(hand).reshape(1, -1)

            interpreter.set_tensor(input_details[0]['index'], keypoints)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            predicted_idx = int(np.argmax(output_data))
            current_label = label_names[predicted_idx]

    # 같은 제스처 유지 시간 체크
    if current_label == last_label:
        duration = time.time() - label_start_time
    else:
        label_start_time = time.time()
        duration = 0

    last_label = current_label

    # 1초 이상 유지 & none이 아니면 이미지 출력
        # 1초 이상 유지 & none이 아니면 이미지 출력
    if duration >= 1.0 and current_label in loaded_images and current_label != 'none':
        overlay_img = loaded_images[current_label]

        # --- 이미지 크기 줄이기 (가로 150px로 고정, 세로는 비율 유지) ---
        scale_width = 150
        h, w = overlay_img.shape[:2]
        scale_ratio = scale_width / w
        new_h = int(h * scale_ratio)
        overlay_img_resized = cv2.resize(overlay_img, (scale_width, new_h), interpolation=cv2.INTER_AREA)

        img = overlay_transparent(img, overlay_img_resized, 50, 50)

    cv2.imshow("Gesture Emoji", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
