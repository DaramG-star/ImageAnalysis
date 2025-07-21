from channels.generic.websocket import WebsocketConsumer
import json
import base64
import cv2
import numpy as np
import mediapipe as mp
import time
from tensorflow.keras.models import load_model

# 1. 모델 로드
model = load_model("models/gesture_model.h5")  # 너가 만든 h5 파일 경로
labels = ["bad", "fist", "good", "heart", "none", "ok", "open_palm", "promise", "victory"]

# 2. MediaPipe 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.7, min_tracking_confidence=0.7)

class GestureConsumer(WebsocketConsumer):
    def connect(self):
        self.accept()

    def disconnect(self, close_code):
        pass

    def receive(self, text_data):
        data = json.loads(text_data)
        img_data = base64.b64decode(data['image'].split(',')[1])
        frame = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        frame = cv2.flip(frame, 1)

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        gesture_result = ""

        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0].landmark
            base_x, base_y, base_z = landmarks[0].x, landmarks[0].y, landmarks[0].z

            relative_landmarks = []
            for lm in landmarks:
                relative_landmarks.extend([
                    lm.x - base_x,
                    lm.y - base_y,
                    lm.z - base_z
                ])

            # numpy로 reshape 후 예측
            input_data = np.array([relative_landmarks], dtype=np.float32)
            predictions = model.predict(input_data)
            predicted_index = np.argmax(predictions[0])
            gesture_result = f"{labels[predicted_index]} ({predictions[0][predicted_index]:.2f})"

        # 클라이언트로 결과 전송
        self.send(text_data=json.dumps({
            'gesture': gesture_result
        }))
