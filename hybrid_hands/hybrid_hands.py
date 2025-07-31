import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import time, os

# ---------- 공통 설정 ----------
STATIC_LABELS  = ['bad', 'fist', 'good', 'gun', 'heart',
                  'none', 'ok', 'open_palm', 'promise', 'rock', 'victory']
DYNAMIC_LABELS = ['fire', 'hi', 'hit', 'none', 'nono', 'nyan', 'shot']

tf.get_logger().setLevel('ERROR')

# ---------- TFLite 모델 ----------
static_interpreter  = tf.lite.Interpreter(model_path="stop_hands/gesture_model7.tflite")
dynamic_interpreter = tf.lite.Interpreter(model_path="moving_hands/gesture_rnn_model3.tflite")
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

# ---------- Kalman Filter 초기화 ----------
def create_kalman():
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    return kf

kalman_filters = [create_kalman() for _ in range(21)]

def smooth_landmarks(landmarks):
    smoothed = []
    for i, lm in enumerate(landmarks.landmark):
        meas = np.array([[np.float32(lm.x)], [np.float32(lm.y)]])
        kalman_filters[i].correct(meas)
        pred = kalman_filters[i].predict()
        smoothed.append([pred[0, 0], pred[1, 0], lm.z])
    return np.array(smoothed, dtype=np.float32)

# ---------- 랜드마크 → 상대 좌표 ----------
def extract_keypoints(landmarks):
    smoothed = smooth_landmarks(landmarks)
    wrist = smoothed[0]
    rel = smoothed - wrist
    return rel.flatten()  # (63,)

# ---------- 웹캠 ----------
cap             = cv2.VideoCapture(0)
move_threshold  = 0.003
conf_threshold  = 0.70
sequence        = []
prev_wrist      = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands_proc.process(rgb)

    dyn_mode = False
    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        drawer.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # 부드럽게 보정된 랜드마크 얻기
        smoothed_landmarks = smooth_landmarks(hand_landmarks)

        # 이동량 계산 (동적/정적 구분)
        wrist_now = smoothed_landmarks[0]
        if prev_wrist is not None:
            mv = np.linalg.norm(wrist_now[:3] - prev_wrist[:3])
            dyn_mode = mv > move_threshold
        prev_wrist = wrist_now

        # ---------- 동적 ----------
        if dyn_mode:
            sequence.append((smoothed_landmarks - wrist_now).flatten())
            sequence = sequence[-30:]
            if len(sequence) == 30:
                inp = np.expand_dims(sequence, axis=0).astype(np.float32)
                dynamic_interpreter.set_tensor(di_in, inp)
                dynamic_interpreter.invoke()
                probs = dynamic_interpreter.get_tensor(di_out)[0]
                dyn_label = DYNAMIC_LABELS[int(np.argmax(probs))]
                conf      = float(np.max(probs))
                if conf < conf_threshold:
                    dyn_label = 'none'
                cv2.putText(img, f'{dyn_label} ({conf:.2f})', (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

                if dyn_label == 'fire' and fire_png is not None:
                    h,w,_ = img.shape
                    cx, cy = int(wrist_now[0]*w), int(wrist_now[1]*h)
                    img = overlay_png(img, fire_png, cx-50, cy-150)

        # ---------- 정적 ----------
        else:
            key = (smoothed_landmarks - wrist_now).flatten().reshape(1,-1)
            static_interpreter.set_tensor(si_in, key)
            static_interpreter.invoke()
            probs = static_interpreter.get_tensor(si_out)[0]
            sta_label = STATIC_LABELS[int(np.argmax(probs))]
            conf      = float(np.max(probs))
            cv2.putText(img, f'{sta_label} ({conf:.2f})', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            sequence.clear()

    else:
        prev_wrist = None
        sequence.clear()

    cv2.imshow('Static & Dynamic Gesture (Kalman Filter)', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
