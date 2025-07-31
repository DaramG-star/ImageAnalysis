import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time
import os
from flask import Flask, render_template, Response, send_from_directory, abort

# --- 1. 초기 설정 및 모델 로드 ---

# 현재 스크립트 파일의 디렉토리 경로를 기준으로 절대 경로를 생성합니다.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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

# --- 헬퍼 함수 (원본 코드에서 가져옴) ---

def extract_relative_keypoints(landmarks):
    """얼굴 랜드마크를 기준으로 상대적인 키포인트를 추출합니다."""
    # 랜드마크 1번(코 끝)을 기준으로 상대적인 위치를 계산
    base = landmarks[1]
    return np.array([[lm.x - base.x, lm.y - base.y] for lm in landmarks]).flatten()

def is_mouth_covered(landmarks, threshold=0.003):
    """입이 가려졌는지 확인합니다."""
    base = landmarks[1] # 코 끝 랜드마크
    mouth_movement = 0
    for i in MOUTH_IDX_FACE:
        lm = landmarks[i]
        mouth_movement += abs(lm.x - base.x) + abs(lm.y - base.y)
    return (mouth_movement / len(MOUTH_IDX_FACE)) < threshold

def is_mouth_closed(landmarks, threshold=0.015):
    """입이 닫혔는지 확인합니다 (윗입술과 아랫입술 랜드마크 간의 거리)."""
    return abs(landmarks[14].y - landmarks[13].y) < threshold

def get_skin_mask(image):
    """이미지에서 피부 영역 마스크를 생성합니다."""
    # RGB 및 YCrCb 색 공간을 사용하여 피부 마스크 생성
    img_float = image.astype(np.float32) / 255.0
    r, g, b = img_float[:,:,2], img_float[:,:,1], img_float[:,:,0]
    mask_rgb = (r > 0.3725) & (g > 0.1568) & (b > 0.0784) & (r > b) & \
               ((np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])) > 0.0588) & \
               (np.abs(r - g) > 0.0588)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    cr, cb = cv2.split(ycrcb)[1:3]
    mask_ycrcb = (133 < cr) & (cr < 173) & (77 < cb) & (cb < 127)
    return (mask_rgb & mask_ycrcb).astype(np.uint8)

def apply_lip_color(frame, face_landmarks, color=(0, 0, 200), alpha=0.4, blur_size=15):
    """얼굴 랜드마크를 기반으로 입술에 색상을 적용합니다."""
    h, w, _ = frame.shape
    # 입술 랜드마크를 추출하여 다각형 생성
    points = np.array([(int(lm.x * w), int(lm.y * h)) for i, lm in enumerate(face_landmarks.landmark) if i in LIPS_IDX_BEAUTY], dtype=np.int32)
    
    # 랜드마크가 충분하지 않으면 아무것도 하지 않음
    if len(points) < 3:
        return frame

    hull = cv2.convexHull(points)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255) # 입술 영역을 마스크로 채움
    
    # 마스크를 블러 처리하여 자연스러운 블렌딩 효과
    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 7)
    mask_f = (mask.astype(float) / 255.0) * alpha
    
    # 색상 레이어를 생성하고 원본 프레임과 블렌딩
    color_layer = np.full_like(frame, color, dtype=np.uint8)
    blended = frame.astype(float) * (1 - mask_f[..., None]) + color_layer.astype(float) * mask_f[..., None]
    return blended.astype(np.uint8)

def overlay_png(bg, fg, x, y):
    """배경 이미지 위에 전경 PNG (알파 채널 포함)를 오버레이합니다."""
    if fg is None: return bg
    h, w = fg.shape[:2]
    
    # 오버레이할 영역이 배경 이미지 내에 있는지 확인
    x_start = max(0, x)
    y_start = max(0, y)
    x_end = min(bg.shape[1], x + w)
    y_end = min(bg.shape[0], y + h)

    # 실제 오버레이할 영역의 크기 계산
    target_w = x_end - x_start
    target_h = y_end - y_start

    if target_w <= 0 or target_h <= 0:
        return bg # 유효한 오버레이 영역이 없으면 원본 배경 반환

    # 전경 이미지에서 실제 사용될 부분
    fg_x_start = x_start - x
    fg_y_start = y_start - y
    fg_x_end = fg_x_start + target_w
    fg_y_end = fg_y_start + target_h
    
    fg_cropped = fg[fg_y_start:fg_y_end, fg_x_start:fg_x_end]
    alpha = fg_cropped[:, :, 3] / 255.0

    # 블렌딩 수행
    for c in range(3): # RGB 채널에 대해
        bg[y_start:y_end, x_start:x_end, c] = bg[y_start:y_end, x_start:x_end, c] * (1 - alpha) + fg_cropped[:, :, c] * alpha
    return bg

def extract_hand_keypoints(landmarks):
    """손 랜드마크를 기준으로 상대적인 키포인트를 추출합니다."""
    # 손목 랜드마크를 기준으로 상대적인 위치를 계산 (x, y, z)
    wrist = landmarks.landmark[0]
    return np.array([[lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z] for lm in landmarks.landmark], dtype=np.float32).flatten()

# --- 2. 비디오 처리 클래스 ---

class VideoProcessor:
    def __init__(self):
        # 모델 로드
        # expression_landmark_model1.tflite는 TFLite 모델이므로 tf.lite.Interpreter를 사용합니다.
        try:
            self.expression_interpreter = tf.lite.Interpreter(model_path=os.path.join(BASE_DIR, 'models', 'expression_landmark_model.tflite'))
            self.expression_interpreter.allocate_tensors()
            self.ei_in, self.ei_out = self.expression_interpreter.get_input_details()[0]['index'], self.expression_interpreter.get_output_details()[0]['index']
        except ValueError as e:
            print(f"오류: 표정 모델(expression_landmark_model1.tflite)을 로드할 수 없습니다. 파일이 유효한 TFLite 모델인지 확인하세요. 오류: {e}")
            self.expression_interpreter = None # 모델 로드 실패 시 None으로 설정
        
        try:
            self.static_interpreter = tf.lite.Interpreter(model_path=os.path.join(BASE_DIR, 'models', 'gesture_model6.tflite'))
            self.static_interpreter.allocate_tensors()
            self.si_in, self.si_out = self.static_interpreter.get_input_details()[0]['index'], self.static_interpreter.get_output_details()[0]['index']
        except ValueError as e:
            print(f"오류: 정적 제스처 모델(gesture_model6.tflite)을 로드할 수 없습니다. 오류: {e}")
            self.static_interpreter = None
                         
        try:
            self.dynamic_interpreter = tf.lite.Interpreter(model_path=os.path.join(BASE_DIR, 'models', 'gesture_rnn_model3.tflite'))
            self.dynamic_interpreter.allocate_tensors()
            self.di_in, self.di_out = self.dynamic_interpreter.get_input_details()[0]['index'], self.dynamic_interpreter.get_output_details()[0]['index']
        except ValueError as e:
            print(f"오류: 동적 제스처 모델(gesture_rnn_model3.tflite)을 로드할 수 없습니다. 오류: {e}")
            self.dynamic_interpreter = None


        # PNG 이미지 로드
        self.fire_png = cv2.imread(os.path.join(BASE_DIR, 'assets', 'fire_effect.png'), cv2.IMREAD_UNCHANGED)
        if self.fire_png is not None:
            self.fire_png = cv2.resize(self.fire_png, (150, 150))
        else:
            print(f"경고: fire_effect.png를 {os.path.join(BASE_DIR, 'assets', 'fire_effect.png')}에서 찾을 수 없습니다.")

        # MediaPipe 초기화
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.face_mesh_processor = self.mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.7)
        self.hands_processor = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
        self.drawer = mp.solutions.drawing_utils

        # 상태 변수 초기화
        self.sequence = []
        self.prev_wrist = None
        self.dyn_label = 'none'
        self.move_threshold = 0.003
        self.conf_threshold = 0.80
        # self.frame_counter = 0 # 프레임 건너뛰기 로직 제거를 위해 주석 처리 또는 제거
        # self.ai_process_interval = 3 # 프레임 건너뛰기 로직 제거를 위해 주석 처리 또는 제거

    def process_frame(self, frame):
        """단일 프레임을 처리하고 결과를 반환합니다."""
        # self.frame_counter += 1 # 프레임 건너뛰기 로직 제거를 위해 주석 처리 또는 제거
        
        # 뷰티 필터 우선 적용 (모든 프레임에 적용)
        # cv2.edgePreservingFilter 대신 cv2.bilateralFilter 사용 (더 빠름)
        # d=9 (픽셀 주변의 지름), sigmaColor=75 (색상 차이), sigmaSpace=75 (공간 차이)
        processed_frame = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)
        
        # 피부 마스크를 사용하여 블렌딩 (원본 로직 유지)
        skin_mask = get_skin_mask(frame) # 원본 프레임에서 마스크 생성
        skin_mask_blur = np.clip(cv2.GaussianBlur(skin_mask.astype(np.float32), (5, 5), 0), 0, 1)[..., np.newaxis]
        # 원본 프레임과 필터링된 프레임을 피부 마스크를 기반으로 블렌딩
        processed_frame = (processed_frame.astype(np.float32) * skin_mask_blur + frame.astype(np.float32) * (1 - skin_mask_blur)).astype(np.uint8)

        # 립 컬러 적용
        # 뷰티 필터 적용 후 랜드마크를 사용하여 립 컬러 적용
        
        face_landmarks = None # 얼굴 랜드마크 초기화
        hand_landmarks = None # 손 랜드마크 초기화
        
        # AI 처리는 이제 모든 프레임에서 수행됩니다.
        # if self.frame_counter % self.ai_process_interval == 0: # 프레임 건너뛰기 로직 제거
        
        # 인식용 RGB 이미지 (뷰티 필터 적용 전 원본 사용)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False # 성능 향상

        # 얼굴과 손 동시 처리
        face_results = self.face_mesh_processor.process(rgb_frame)
        hand_results = self.hands_processor.process(rgb_frame)

        rgb_frame.flags.writeable = True

        # --- 얼굴 처리 ---
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            landmarks_list = face_landmarks.landmark

            # 립 컬러 적용 (뷰티 필터 적용된 프레임에)
            processed_frame = apply_lip_color(processed_frame, face_landmarks)
            
            # 표정 인식
            if is_mouth_covered(landmarks_list):
                label = "입 가림" # "mouth covered" 번역
                cv2.putText(processed_frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            else:
                if self.expression_interpreter: # 모델이 성공적으로 로드된 경우에만 추론
                    keypoints = extract_relative_keypoints(landmarks_list)
                    input_data = keypoints.reshape(1, -1).astype(np.float32)
                    
                    # TFLite Interpreter를 사용하여 표정 모델 추론
                    self.expression_interpreter.set_tensor(self.ei_in, input_data)
                    self.expression_interpreter.invoke()
                    pred_probs = self.expression_interpreter.get_tensor(self.ei_out)

                    pred = np.argmax(pred_probs)
                    conf = np.max(pred_probs)
                    label = FACE_LABELS[pred]

                    if conf >= 0.8:
                        if label in MOUTH_LABELS_FOR_CLOSED_CHECK and is_mouth_closed(landmarks_list):
                            cv2.putText(processed_frame, "불확실 (입 닫힘)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2) # "Uncertain (mouth closed)" 번역
                        else:
                            emoji = EMOJIS[label]
                            cv2.putText(processed_frame, f"{emoji} {label} ({conf:.2f})", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        cv2.putText(processed_frame, "불확실", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2) # "Uncertain" 번역
                else:
                    cv2.putText(processed_frame, "표정 모델 로드 실패", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


        # --- 손 처리 ---
        dyn_mode = False
        if hand_results.multi_hand_landmarks:
            hand_landmarks = hand_results.multi_hand_landmarks[0]
            self.drawer.draw_landmarks(processed_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            wrist_now = hand_landmarks.landmark[0]
            if self.prev_wrist:
                mv = np.linalg.norm([wrist_now.x - self.prev_wrist.x, wrist_now.y - self.prev_wrist.y, wrist_now.z - self.prev_wrist.z])
                dyn_mode = mv > self.move_threshold
            self.prev_wrist = wrist_now

            # 동적 제스처
            if dyn_mode:
                if self.dynamic_interpreter: # 모델이 성공적으로 로드된 경우에만 추론
                    self.sequence.append(extract_hand_keypoints(hand_landmarks))
                    self.sequence = self.sequence[-30:] # 최근 30프레임 유지
                    if len(self.sequence) == 30:
                        inp = np.expand_dims(self.sequence, axis=0).astype(np.float32)
                        self.dynamic_interpreter.set_tensor(self.di_in, inp)
                        self.dynamic_interpreter.invoke()
                        probs = self.dynamic_interpreter.get_tensor(self.di_out)[0]
                        self.dyn_label = DYNAMIC_HAND_LABELS[int(np.argmax(probs))]
                        conf = float(np.max(probs))
                        if conf < self.conf_threshold: self.dyn_label = 'none'
                    
                    if self.dyn_label != 'none':
                        cv2.putText(processed_frame, f'동적: {self.dyn_label} ({conf:.2f})', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) # "DYNAMIC:" 번역
                        if self.dyn_label == 'fire' and self.fire_png is not None:
                            h,w,_ = processed_frame.shape
                            cx, cy = int(wrist_now.x*w), int(wrist_now.y*h)
                            processed_frame = overlay_png(processed_frame, self.fire_png, cx - 75, cy - 150)
                    else:
                        # 동적 제스처가 'none'일 경우, 정적 제스처 텍스트를 지우기 위해 빈 문자열 출력
                        cv2.putText(processed_frame, '', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                else:
                    cv2.putText(processed_frame, "동적 제스처 모델 로드 실패", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 정적 제스처
            else:
                if self.static_interpreter: # 모델이 성공적으로 로드된 경우에만 추론
                    key = extract_hand_keypoints(hand_landmarks).reshape(1, -1)
                    self.static_interpreter.set_tensor(self.si_in, key)
                    self.static_interpreter.invoke()
                    probs = self.static_interpreter.get_tensor(self.si_out)[0]
                    sta_label = STATIC_HAND_LABELS[int(np.argmax(probs))]
                    conf = float(np.max(probs))
                    cv2.putText(processed_frame, f'정적: {sta_label} ({conf:.2f})', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2) # "STATIC:" 번역
                    self.sequence.clear()
                    self.dyn_label = 'none' # 동적 모드 해제
                else:
                    cv2.putText(processed_frame, "정적 제스처 모델 로드 실패", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            self.prev_wrist = None
            self.sequence.clear()
            self.dyn_label = 'none'
            # 손이 감지되지 않을 때 텍스트 지우기
            cv2.putText(processed_frame, '', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # --- 최종 프레임 인코딩 및 전송 ---
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        return frame_bytes

# --- 3. 비디오 스트리밍 및 처리 ---

def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("오류: 비디오 스트림을 열 수 없습니다.") # "Error: Could not open video stream." 번역
        return

    processor = VideoProcessor()
    # 모델 로드 실패 시 스트림 중단
    if processor.expression_interpreter is None or \
       processor.static_interpreter is None or \
       processor.dynamic_interpreter is None:
        print("필수 AI 모델 중 하나 이상을 로드하는 데 실패했습니다. 애플리케이션을 종료합니다.")
        return # 더 이상 프레임을 생성하지 않고 종료

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("오류: 프레임을 가져오지 못했습니다.") # "Error: Failed to grab frame." 번역
            break
        
        # 프레임 처리
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
    """브라우저의 favicon.ico 요청을 처리합니다."""
    # 투명한 16x16 픽셀 SVG 아이콘을 base64로 인코딩
    # 이는 'static' 폴더나 실제 파일을 필요로 하지 않으므로 배포가 더 간단합니다.
    icon_svg = """<svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="8" cy="8" r="7" fill="#63b3ed"/></svg>"""
    encoded_icon = icon_svg.encode('utf-8')
    return Response(encoded_icon, mimetype='image/svg+xml')

# --- 5. 앱 실행 ---
if __name__ == '__main__':
    # 모델 및 assets 폴더가 없으면 생성 (개발 편의성)
    os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'assets'), exist_ok=True)
    
    app.run(debug=True, threaded=True, host='0.0.0.0') # 외부 접속 허용
