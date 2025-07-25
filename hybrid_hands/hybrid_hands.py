import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import time, os

# ---------- 공통 설정 ----------
STATIC_LABELS  = ['bad', 'fist', 'good', 'gun', 'heart',
                  'none', 'ok', 'open_palm', 'promise', 'rock', 'victory']
DYNAMIC_LABELS = ['fire', 'hi', 'hit', 'none', 'nono', 'nyan', 'shot']

# 로깅 최소화(TF 진행바 등 숨김)
tf.get_logger().setLevel('ERROR')

# ---------- TFLite 모델 ----------
static_interpreter  = tf.lite.Interpreter(model_path="stop_hands/gesture_model4.tflite")
dynamic_interpreter = tf.lite.Interpreter(model_path="moving_hands/gesture_rnn_model1.tflite")
static_interpreter.allocate_tensors()
dynamic_interpreter.allocate_tensors()
si_in,  si_out  = static_interpreter.get_input_details()[0]['index'],  static_interpreter.get_output_details()[0]['index']
di_in,  di_out  = dynamic_interpreter.get_input_details()[0]['index'],  dynamic_interpreter.get_output_details()[0]['index']

# ---------- MediaPipe ----------
mp_hands   = mp.solutions.hands
hands_proc = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
drawer     = mp.solutions.drawing_utils

# ---------- 파이어 이펙트 ----------
fire_png = cv2.imread('moving_hands/fire_effect.png', cv2.IMREAD_UNCHANGED)
fire_png = cv2.resize(fire_png, (100,100)) if fire_png is not None else None

def overlay_png(bg, fg, x, y):
    if fg is None: return bg
    h,w = fg.shape[:2]
    if x<0 or y<0 or x+w>bg.shape[1] or y+h>bg.shape[0]: return bg
    alpha = fg[:,:,3]/255.0
    for c in range(3):
        bg[y:y+h, x:x+w, c] = bg[y:y+h, x:x+w, c]*(1-alpha) + fg[:,:,c]*alpha
    return bg

# ---------- 랜드마크 → 상대 좌표 ----------
def extract_keypoints(landmarks):
    wrist = landmarks.landmark[0]
    return np.array([[lm.x-wrist.x, lm.y-wrist.y, lm.z-wrist.z] for lm in landmarks.landmark],
                    dtype=np.float32).flatten()          # (63,)

# ---------- 웹캠 ----------
cap             = cv2.VideoCapture(0)
move_threshold  = 0.003          # 움직임 기준
conf_threshold  = 0.80           # 동적 모델 신뢰도
sequence        = []             # 동적 시퀀스(최근 30프레임)
prev_wrist      = None
last_dyn_time   = 0              # 마지막으로 동적 제스처 출력한 시각

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    img = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands_proc.process(rgb)

    dyn_mode = False
    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        drawer.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # 이동량 계산(동적/정적 분기용)
        wrist_now = hand_landmarks.landmark[0]
        if prev_wrist is not None:
            mv = np.linalg.norm([wrist_now.x-prev_wrist.x,
                                 wrist_now.y-prev_wrist.y,
                                 wrist_now.z-prev_wrist.z])
            dyn_mode = mv > move_threshold
        prev_wrist = wrist_now

        # ---------- 동적 ----------
        if dyn_mode:
            sequence.append(extract_keypoints(hand_landmarks))
            sequence = sequence[-30:]
            if len(sequence) == 30:
                inp = np.expand_dims(sequence, axis=0).astype(np.float32)  # (1,30,63)
                dynamic_interpreter.set_tensor(di_in, inp)
                dynamic_interpreter.invoke()
                probs = dynamic_interpreter.get_tensor(di_out)[0]
                dyn_label = DYNAMIC_LABELS[int(np.argmax(probs))]
                conf      = float(np.max(probs))
                if conf < conf_threshold:
                    dyn_label = 'none'
                # 결과 표시
                cv2.putText(img, f'{dyn_label} ({conf:.2f})', (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                last_dyn_time = time.time()

                # fire 이펙트
                if dyn_label == 'fire' and fire_png is not None:
                    h,w,_ = img.shape
                    cx, cy = int(wrist_now.x*w), int(wrist_now.y*h)
                    img = overlay_png(img, fire_png, cx-50, cy-150)

        # ---------- 정적 ----------
        else:
            key = extract_keypoints(hand_landmarks).reshape(1,-1)
            static_interpreter.set_tensor(si_in, key)
            static_interpreter.invoke()
            probs = static_interpreter.get_tensor(si_out)[0]
            sta_label = STATIC_LABELS[int(np.argmax(probs))]
            conf      = float(np.max(probs))
            cv2.putText(img, f'{sta_label} ({conf:.2f})', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            sequence.clear()      # 동적 버퍼 리셋

    else:
        prev_wrist = None
        sequence.clear()

    cv2.imshow('Static & Dynamic Gesture (TFLite)', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
