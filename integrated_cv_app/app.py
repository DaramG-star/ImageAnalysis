import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time
import os
from flask import Flask, render_template, Response

# --- 1. 초기 설정 및 모델/에셋 로드 ---

# 현재 스크립트 파일의 디렉토리 경로를 기준으로 절대 경로를 생성합니다.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
MODELS_DIR = os.path.join(BASE_DIR, "models")


# Flask 앱 초기화
app = Flask(__name__)

# --- 상수 정의 ---
# 표정 인식 설정
FACE_LABELS = ['laugh', 'serious', 'surprise', 'yawn', 'none']
EMOJIS = {
    "laugh": ":D",
    "serious": "-_-",
    "surprise": "!!",
    "yawn": "Zz",
    "none": "..."
}
MOUTH_IDX_FACE = [13, 14, 78, 82, 87, 88, 95, 61, 146, 91, 181, 308, 317, 312, 311, 402]
MOUTH_LABELS_FOR_CLOSED_CHECK = ['laugh', 'yawn', 'surprise']

# 뷰티 필터 설정
LIPS_IDX_BEAUTY = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
                   308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

# 손동작 인식 설정
STATIC_HAND_LABELS = ['bad', 'fist', 'good', 'gun', 'heart', 'none', 'ok', 'open_palm', 'promise', 'rock', 'victory']
DYNAMIC_HAND_LABELS = ['fire', 'hi', 'hit', 'none', 'nono', 'nyan', 'shot']

# TensorFlow 로거 레벨 설정 (경고 메시지 줄이기)
tf.get_logger().setLevel('ERROR')


# --- 헬퍼 함수 ---

def extract_relative_keypoints(landmarks):
    """얼굴 랜드마크를 기준으로 상대적인 키포인트를 추출합니다."""
    base = landmarks[1]
    return np.array([[lm.x - base.x, lm.y - base.y] for lm in landmarks]).flatten()

def is_mouth_covered(landmarks, threshold=0.003):
    """입이 가려졌는지 확인합니다."""
    base = landmarks[1]
    mouth_movement = sum(abs(landmarks[i].x - base.x) + abs(landmarks[i].y - base.y) for i in MOUTH_IDX_FACE)
    return (mouth_movement / len(MOUTH_IDX_FACE)) < threshold

def is_mouth_closed(landmarks, threshold=0.015):
    """입이 닫혔는지 확인합니다."""
    return abs(landmarks[14].y - landmarks[13].y) < threshold

def get_skin_mask(image):
    """이미지에서 피부 영역 마스크를 생성합니다."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 48, 80], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    return cv2.inRange(hsv, lower_skin, upper_skin)

def apply_lip_color(frame, face_landmarks, color=(0, 0, 200), alpha=0.4, blur_size=15):
    """얼굴 랜드마크를 기반으로 입술에 색상을 적용합니다."""
    h, w, _ = frame.shape
    points = np.array([(int(lm.x * w), int(lm.y * h)) for i, lm in enumerate(face_landmarks.landmark) if i in LIPS_IDX_BEAUTY], dtype=np.int32)
    if len(points) < 3: return frame

    hull = cv2.convexHull(points)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    
    if blur_size % 2 == 0: blur_size += 1
    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 7)
    mask_f = (mask.astype(float) / 255.0) * alpha
    
    color_layer = np.full_like(frame, color, dtype=np.uint8)
    blended = frame.astype(float) * (1 - mask_f[..., None]) + color_layer.astype(float) * mask_f[..., None]
    return blended.astype(np.uint8)

def overlay_image(bg, overlay, x, y, size=None):
    """배경 이미지 위에 전경 PNG (알파 채널 포함)를 오버레이합니다."""
    if overlay is None: return bg
    
    overlay_h, overlay_w = overlay.shape[:2]
    if size:
        overlay = cv2.resize(overlay, (size, size))
        overlay_h, overlay_w = overlay.shape[:2]

    y1, y2 = max(0, y), min(bg.shape[0], y + overlay_h)
    x1, x2 = max(0, x), min(bg.shape[1], x + overlay_w)

    if y1 >= y2 or x1 >= x2: return bg

    alpha = overlay[:, :, 3] / 255.0
    alpha_mask = alpha[y1-y:y2-y, x1-x:x2-x]
    alpha_mask_3ch = cv2.merge([alpha_mask, alpha_mask, alpha_mask])

    bg_roi = bg[y1:y2, x1:x2]
    overlay_content = overlay[y1-y:y2-y, x1-x:x2-x, :3]
    
    blended_roi = (overlay_content * alpha_mask_3ch) + (bg_roi * (1 - alpha_mask_3ch))
    bg[y1:y2, x1:x2] = blended_roi.astype(bg.dtype)
    return bg

def extract_hand_keypoints(landmarks):
    """손 랜드마크를 기준으로 상대적인 키포인트를 추출합니다."""
    wrist = landmarks.landmark[0]
    return np.array([[lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z] for lm in landmarks.landmark], dtype=np.float32).flatten()


# --- 2. 비디오 처리 클래스 ---

class VideoProcessor:
    def __init__(self):
        # 모델 로드
        try:
            self.expression_interpreter = tf.lite.Interpreter(model_path=os.path.join(MODELS_DIR, 'expression_landmark_model1.tflite'))
            self.expression_interpreter.allocate_tensors()
            self.ei_in, self.ei_out = self.expression_interpreter.get_input_details()[0]['index'], self.expression_interpreter.get_output_details()[0]['index']
        except Exception as e:
            print(f"오류: 표정 모델 로드 실패. {e}")
            self.expression_interpreter = None

        try:
            self.static_interpreter = tf.lite.Interpreter(model_path=os.path.join(MODELS_DIR, 'gesture_model7.tflite'))
            self.static_interpreter.allocate_tensors()
            self.si_in, self.si_out = self.static_interpreter.get_input_details()[0]['index'], self.static_interpreter.get_output_details()[0]['index']
        except Exception as e:
            print(f"오류: 정적 제스처 모델 로드 실패. {e}")
            self.static_interpreter = None
        
        try:
            self.dynamic_interpreter = tf.lite.Interpreter(model_path=os.path.join(MODELS_DIR, 'gesture_rnn_model4.tflite'))
            self.dynamic_interpreter.allocate_tensors()
            self.di_in, self.di_out = self.dynamic_interpreter.get_input_details()[0]['index'], self.dynamic_interpreter.get_output_details()[0]['index']
        except Exception as e:
            print(f"오류: 동적 제스처 모델 로드 실패. {e}")
            self.dynamic_interpreter = None

        # 동적 제스처 PNG 이미지 로드
        self.dyn_images = {label: cv2.imread(os.path.join(ASSETS_DIR, f'{filename}.png'), cv2.IMREAD_UNCHANGED)
                           for label, filename in [('fire', 'fire_effect'), ('hi', 'hi'), ('hit', 'fist'), 
                                                   ('nono', 'nono'), ('nyan', 'nyan2'), ('shot', 'heart')]}

        # MediaPipe 초기화
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.face_mesh_processor = self.mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.7)
        self.hands_processor = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.drawer = mp.solutions.drawing_utils

        # 상태 변수 초기화
        self.sequence = []
        self.prev_wrist = None
        self.dyn_label = 'none'
        self.dyn_label_start_time = 0
        self.dyn_label_duration = 2.0 # 2초간 표시
        self.move_threshold = 0.004
        self.conf_threshold = 0.85

    def process_frame(self, frame):
        # 뷰티 필터
        processed_frame = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)
        skin_mask = get_skin_mask(frame)
        skin_mask_blur = np.clip(cv2.GaussianBlur(skin_mask.astype(np.float32), (5, 5), 0), 0, 1)[..., np.newaxis]
        processed_frame = (processed_frame.astype(np.float32) * skin_mask_blur + frame.astype(np.float32) * (1 - skin_mask_blur)).astype(np.uint8)

        # 인식용 RGB 이미지
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        face_results = self.face_mesh_processor.process(rgb_frame)
        hand_results = self.hands_processor.process(rgb_frame)

        rgb_frame.flags.writeable = True

        # --- 얼굴 처리 ---
        if face_results.multi_face_landmarks and self.expression_interpreter:
            face_landmarks = face_results.multi_face_landmarks[0]
            landmarks_list = face_landmarks.landmark
            processed_frame = apply_lip_color(processed_frame, face_landmarks)
            
            if is_mouth_covered(landmarks_list):
                cv2.putText(processed_frame, "Mouth Covered", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            else:
                keypoints = extract_relative_keypoints(landmarks_list)
                input_data = keypoints.reshape(1, -1).astype(np.float32)
                self.expression_interpreter.set_tensor(self.ei_in, input_data)
                self.expression_interpreter.invoke()
                pred_probs = self.expression_interpreter.get_tensor(self.ei_out)
                conf = np.max(pred_probs)
                if conf >= 0.8:
                    label = FACE_LABELS[np.argmax(pred_probs)]
                    if label in MOUTH_LABELS_FOR_CLOSED_CHECK and is_mouth_closed(landmarks_list):
                        cv2.putText(processed_frame, "Uncertain", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
                    else:
                        cv2.putText(processed_frame, f"{EMOJIS[label]} {label} ({conf:.2f})", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # --- 손 처리 ---
        current_time = time.time()
        if current_time - self.dyn_label_start_time > self.dyn_label_duration:
            self.dyn_label = 'none'

        if hand_results.multi_hand_landmarks:
            hand_landmarks = hand_results.multi_hand_landmarks[0]
            self.drawer.draw_landmarks(processed_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            wrist_now = hand_landmarks.landmark[0]
            index_tip = hand_landmarks.landmark[8]
            move_dist = 0
            if self.prev_wrist:
                move_dist = np.linalg.norm([wrist_now.x - self.prev_wrist.x, wrist_now.y - self.prev_wrist.y, wrist_now.z - self.prev_wrist.z])
            self.prev_wrist = wrist_now

            # 동적/정적 모드 결정 및 처리
            if move_dist > self.move_threshold: # 동적 모드
                self.sequence.append(extract_hand_keypoints(hand_landmarks))
                self.sequence = self.sequence[-30:]
                if len(self.sequence) == 30 and self.dynamic_interpreter:
                    inp = np.expand_dims(self.sequence, axis=0).astype(np.float32)
                    self.dynamic_interpreter.set_tensor(self.di_in, inp)
                    self.dynamic_interpreter.invoke()
                    probs = self.dynamic_interpreter.get_tensor(self.di_out)[0]
                    conf = float(np.max(probs))
                    if conf > self.conf_threshold:
                        label = DYNAMIC_HAND_LABELS[int(np.argmax(probs))]
                        if label != 'none':
                            self.dyn_label = label
                            self.dyn_label_start_time = current_time
            else: # 정적 모드
                self.sequence.clear()
                if self.static_interpreter:
                    key = extract_hand_keypoints(hand_landmarks).reshape(1, -1)
                    self.static_interpreter.set_tensor(self.si_in, key)
                    self.static_interpreter.invoke()
                    probs = self.static_interpreter.get_tensor(self.si_out)[0]
                    conf = float(np.max(probs))
                    if conf > self.conf_threshold:
                        sta_label = STATIC_HAND_LABELS[int(np.argmax(probs))]
                        if sta_label != 'none':
                            cv2.putText(processed_frame, f'Static: {sta_label} ({conf:.2f})', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        else:
            self.prev_wrist = None
            self.sequence.clear()

        # 동적 제스처 효과 오버레이
        if self.dyn_label != 'none' and hand_results.multi_hand_landmarks:
            h, w, _ = processed_frame.shape
            wrist = hand_results.multi_hand_landmarks[0].landmark[0]
            index_tip = hand_results.multi_hand_landmarks[0].landmark[8]
            cx_wrist, cy_wrist = int(wrist.x * w), int(wrist.y * h)
            cx_index, cy_index = int(index_tip.x * w), int(index_tip.y * h)
            
            overlay = self.dyn_images.get(self.dyn_label)
            if overlay is not None:
                if self.dyn_label == 'fire':
                    processed_frame = overlay_image(processed_frame, overlay, cx_wrist - 75, cy_wrist - 150, size=150)
                elif self.dyn_label == 'hi':
                    processed_frame = overlay_image(processed_frame, overlay, cx_wrist - 60, cy_wrist - 180, size=120)
                elif self.dyn_label == 'hit':
                    processed_frame = overlay_image(processed_frame, overlay, cx_wrist - 60, cy_wrist - 60, size=120)
                elif self.dyn_label == 'nono':
                    processed_frame = overlay_image(processed_frame, overlay, cx_wrist - 150, cy_wrist - 50, size=120)
                elif self.dyn_label == 'nyan':
                    processed_frame = overlay_image(processed_frame, overlay, cx_wrist + 80, cy_wrist - 50, size=120)
                elif self.dyn_label == 'shot':
                    processed_frame = overlay_image(processed_frame, overlay, cx_index - 60, cy_index - 100, size=120)

        # 최종 프레임 인코딩
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        return buffer.tobytes()


# --- 3. 비디오 스트리밍 및 처리 ---

def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("오류: 비디오 스트림을 열 수 없습니다.")
        return

    processor = VideoProcessor()
    if not all([processor.expression_interpreter, processor.static_interpreter, processor.dynamic_interpreter]):
        print("필수 AI 모델 중 하나 이상을 로드하는 데 실패했습니다. 애플리케이션을 종료합니다.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("오류: 프레임을 가져오지 못했습니다.")
            break
        
        frame_bytes = processor.process_frame(frame)
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

@app.route('/favicon.ico')
def favicon():
    """브라우저의 favicon.ico 요청을 처리합니다 (404 방지)."""
    return ('', 204)

# --- 5. 앱 실행 ---
if __name__ == '__main__':
    os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'assets'), exist_ok=True)
    app.run(debug=True, threaded=True, host='0.0.0.0')
