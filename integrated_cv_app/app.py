import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time
import os
from flask import Flask, render_template, Response

# --- 1. 초기 설정 및 모델 로드 ---

# 현재 스크립트 파일의 디렉토리 경로를 기준으로 절대 경로를 생성합니다.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Flask 앱 초기화
app = Flask(__name__)

# --- 표정 인식 설정 ---
FACE_LABELS = ['laugh', 'serious', 'surprise', 'yawn', 'none']
EMOJIS = {
    "laugh": ":D",
    "serious": "-_-",
    "surprise": "!!",
    "yawn": "Zz",
    "none": "..."
}
MOUTH_IDX_FACE = [13, 14, 78, 82, 87, 88, 95, 61, 146, 91, 181, 308, 317, 312, 311, 402]
mouth_labels = ['laugh', 'yawn', 'surprise']
# 절대 경로를 사용하여 모델 로드
expression_model_path = os.path.join(BASE_DIR, 'models', 'face_expression_landmark_model1.h5')
expression_model = tf.keras.models.load_model(expression_model_path)


# --- 뷰티 필터 설정 ---
LIPS_IDX_BEAUTY = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
                   308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

# --- 손동작 인식 설정 ---
STATIC_HAND_LABELS  = ['bad', 'fist', 'good', 'gun', 'heart', 'none', 'ok', 'open_palm', 'promise', 'rock', 'victory']
DYNAMIC_HAND_LABELS = ['fire', 'hi', 'hit', 'none', 'nono', 'nyan', 'shot']
tf.get_logger().setLevel('ERROR')

# 절대 경로를 사용하여 모델 로드
static_model_path = os.path.join(BASE_DIR, 'models', 'gesture_model6.tflite')
dynamic_model_path = os.path.join(BASE_DIR, 'models', 'gesture_rnn_model3.tflite')
static_interpreter = tf.lite.Interpreter(model_path=static_model_path)
dynamic_interpreter = tf.lite.Interpreter(model_path=dynamic_model_path)

static_interpreter.allocate_tensors()
dynamic_interpreter.allocate_tensors()
si_in, si_out = static_interpreter.get_input_details()[0]['index'], static_interpreter.get_output_details()[0]['index']
di_in, di_out = dynamic_interpreter.get_input_details()[0]['index'], dynamic_interpreter.get_output_details()[0]['index']

# 절대 경로를 사용하여 이미지 로드
fire_png_path = os.path.join(BASE_DIR, 'assets', 'fire_effect.png')
fire_png = cv2.imread(fire_png_path, cv2.IMREAD_UNCHANGED)
if fire_png is not None:
    fire_png = cv2.resize(fire_png, (150, 150))

# --- MediaPipe 초기화 ---
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
face_mesh_processor = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.7)
hands_processor = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
drawer = mp.solutions.drawing_utils

# --- 2. 헬퍼 함수 (원본 코드에서 가져옴) ---

# 표정 인식 함수
def extract_relative_keypoints(landmarks):
    base = landmarks[1]
    return np.array([[lm.x - base.x, lm.y - base.y] for lm in landmarks]).flatten()

def is_mouth_covered(landmarks, threshold=0.003):
    base, mouth_movement = landmarks[1], 0
    for i in MOUTH_IDX_FACE:
        lm = landmarks[i]
        mouth_movement += abs(lm.x - base.x) + abs(lm.y - base.y)
    return (mouth_movement / len(MOUTH_IDX_FACE)) < threshold

def is_mouth_closed(landmarks, threshold=0.015):
    return abs(landmarks[14].y - landmarks[13].y) < threshold

# 뷰티 필터 함수
def get_skin_mask(image):
    img_float = image.astype(np.float32) / 255.0
    r, g, b = img_float[:,:,2], img_float[:,:,1], img_float[:,:,0]
    mask_rgb = (r > 0.3725) & (g > 0.1568) & (b > 0.0784) & (r > b) & \
               ((np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])) > 0.0588) & \
               (np.abs(r - g) > 0.0588)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    cr, cb = cv2.split(ycrcb)[1:3]
    mask_ycrcb = (133 < cr) & (cr < 173) & (77 < cb) & (cb < 127)
    return (mask_rgb & mask_ycrcb).astype(np.uint8)

def beautify_face(image):
    # 기본값으로 필터 적용
    gamma, contrast, smoothness, saturation = 1.5, 1.2, 30, 1.1
    smoothed = cv2.edgePreservingFilter(image, flags=1, sigma_s=smoothness, sigma_r=0.2)
    skin_mask = get_skin_mask(image)
    skin_mask_blur = np.clip(cv2.GaussianBlur(skin_mask.astype(np.float32), (5, 5), 0), 0, 1)[..., np.newaxis]
    blended = (smoothed.astype(np.float32) * skin_mask_blur + image.astype(np.float32) * (1 - skin_mask_blur)).astype(np.uint8)
    hsv = cv2.cvtColor(blended, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
    beautified = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")
    beautified = cv2.LUT(beautified, table)
    beautified = cv2.convertScaleAbs(beautified, alpha=contrast, beta=0)
    return beautified

def apply_lip_color(frame, face_landmarks, color=(0, 0, 200), alpha=0.4, blur_size=15):
    h, w, _ = frame.shape
    points = np.array([(int(lm.x * w), int(lm.y * h)) for i, lm in enumerate(face_landmarks.landmark) if i in LIPS_IDX_BEAUTY], dtype=np.int32)
    hull = cv2.convexHull(points)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 7)
    mask_f = (mask.astype(float) / 255.0) * alpha
    color_layer = np.full_like(frame, color, dtype=np.uint8)
    blended = frame.astype(float) * (1 - mask_f[..., None]) + color_layer.astype(float) * mask_f[..., None]
    return blended.astype(np.uint8)

# 손동작 인식 함수
def overlay_png(bg, fg, x, y):
    if fg is None: return bg
    h, w = fg.shape[:2]
    if x < 0 or y < 0 or x + w > bg.shape[1] or y + h > bg.shape[0]: return bg
    alpha = fg[:, :, 3] / 255.0
    for c in range(3):
        bg[y:y+h, x:x+w, c] = bg[y:y+h, x:x+w, c] * (1 - alpha) + fg[:, :, c] * alpha
    return bg

def extract_hand_keypoints(landmarks):
    wrist = landmarks.landmark[0]
    return np.array([[lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z] for lm in landmarks.landmark], dtype=np.float32).flatten()

# --- 3. 비디오 스트리밍 및 처리 ---

def generate_frames():
    cap = cv2.VideoCapture(0)
    sequence, prev_wrist, dyn_label = [], None, 'none'
    move_threshold, conf_threshold = 0.003, 0.80

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # 뷰티 필터 우선 적용
        processed_frame = beautify_face(frame)
        
        # 인식용 RGB 이미지 (뷰티 필터 적용 전 원본 사용)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False # 성능 향상
        
        # 얼굴과 손 동시 처리
        face_results = face_mesh_processor.process(rgb_frame)
        hand_results = hands_processor.process(rgb_frame)

        rgb_frame.flags.writeable = True

        # --- 얼굴 처리 ---
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            landmarks_list = face_landmarks.landmark

            # 립 컬러 적용 (뷰티 필터 적용된 프레임에)
            processed_frame = apply_lip_color(processed_frame, face_landmarks)
            
            # 표정 인식
            if is_mouth_covered(landmarks_list):
                label = "mouth covered"
                cv2.putText(processed_frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            else:
                keypoints = extract_relative_keypoints(landmarks_list)
                input_data = keypoints.reshape(1, -1).astype(np.float32)
                pred_probs = expression_model.predict(input_data, verbose=0)
                pred = np.argmax(pred_probs)
                conf = np.max(pred_probs)
                label = FACE_LABELS[pred]

                if conf >= 0.8:
                    if label in mouth_labels and is_mouth_closed(landmarks_list):
                        cv2.putText(processed_frame, "Uncertain (mouth closed)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
                    else:
                        emoji = EMOJIS[label]
                        cv2.putText(processed_frame, f"{emoji} {label} ({conf:.2f})", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(processed_frame, "Uncertain", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)

        # --- 손 처리 ---
        dyn_mode = False
        if hand_results.multi_hand_landmarks:
            hand_landmarks = hand_results.multi_hand_landmarks[0]
            drawer.draw_landmarks(processed_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            wrist_now = hand_landmarks.landmark[0]
            if prev_wrist:
                mv = np.linalg.norm([wrist_now.x - prev_wrist.x, wrist_now.y - prev_wrist.y, wrist_now.z - prev_wrist.z])
                dyn_mode = mv > move_threshold
            prev_wrist = wrist_now

            # 동적 제스처
            if dyn_mode:
                sequence.append(extract_hand_keypoints(hand_landmarks))
                sequence = sequence[-30:]
                if len(sequence) == 30:
                    inp = np.expand_dims(sequence, axis=0).astype(np.float32)
                    dynamic_interpreter.set_tensor(di_in, inp)
                    dynamic_interpreter.invoke()
                    probs = dynamic_interpreter.get_tensor(di_out)[0]
                    dyn_label = DYNAMIC_HAND_LABELS[int(np.argmax(probs))]
                    conf = float(np.max(probs))
                    if conf < conf_threshold: dyn_label = 'none'
                
                if dyn_label != 'none':
                    cv2.putText(processed_frame, f'DYNAMIC: {dyn_label} ({conf:.2f})', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if dyn_label == 'fire' and fire_png is not None:
                        h,w,_ = processed_frame.shape
                        cx, cy = int(wrist_now.x*w), int(wrist_now.y*h)
                        processed_frame = overlay_png(processed_frame, fire_png, cx - 75, cy - 150)
            
            # 정적 제스처
            else:
                key = extract_hand_keypoints(hand_landmarks).reshape(1, -1)
                static_interpreter.set_tensor(si_in, key)
                static_interpreter.invoke()
                probs = static_interpreter.get_tensor(si_out)[0]
                sta_label = STATIC_HAND_LABELS[int(np.argmax(probs))]
                conf = float(np.max(probs))
                cv2.putText(processed_frame, f'STATIC: {sta_label} ({conf:.2f})', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                sequence.clear()
                dyn_label = 'none' # 동적 모드 해제
        else:
            prev_wrist = None
            sequence.clear()
            dyn_label = 'none'

        # --- 최종 프레임 인코딩 및 전송 ---
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# --- 4. Flask 라우트 설정 ---

@app.route('/')
def index():
    """메인 페이지를 렌더링합니다."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """비디오 스트리밍 경로입니다."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- 5. 앱 실행 ---
if __name__ == '__main__':
    app.run(debug=True, threaded=True)
