import cv2
import mediapipe as mp
import numpy as np
import math
import json
import os

# Mediapipe FaceMesh 설정
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7
)

# 회전 함수
def rotate_image(image, angle):
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(0, 0, 0, 0))

# PNG 오버레이 함수
def overlay_transparent(background, overlay, x, y):
    h, w = overlay.shape[:2]
    if x < 0: x = 0
    if y < 0: y = 0
    if x + w > background.shape[1]:
        w = background.shape[1] - x
        overlay = overlay[:, :w]
    if y + h > background.shape[0]:
        h = background.shape[0] - y
        overlay = overlay[:h, :]

    if overlay.shape[2] < 4:
        return background

    b, g, r, a = cv2.split(overlay)
    mask = a / 255.0

    for c in range(3):
        background[y:y+h, x:x+w, c] = \
            (1 - mask) * background[y:y+h, x:x+w, c] + mask * overlay[:h, :w, c]
    return background

# 각도 계산
def get_angle(p1, p2):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    return math.degrees(math.atan2(dy, dx))

# 필터 설정 (초기값)
filter_cfg = {
    "path": "OpenCV-Face-Filters/moustache.png",
    "anchor": 13,      # 윗입술 중앙
    "scale_w": 0.5,
    "scale_h": 0.3,
    "offset_x": 0.0,
    "offset_y": 0.0,
    "rotate": True
}

# 트랙바 콜백 (아무 동작 X)
def nothing(x): pass

# 윈도우 및 트랙바 생성
cv2.namedWindow("FaceMesh Filter")
cv2.createTrackbar("X Offset", "FaceMesh Filter", 0, 200, nothing)   # -100~100
cv2.createTrackbar("Y Offset", "FaceMesh Filter", 0, 200, nothing)
cv2.createTrackbar("Scale W", "FaceMesh Filter", 50, 200, nothing)   # 0.1~2.0
cv2.createTrackbar("Scale H", "FaceMesh Filter", 30, 200, nothing)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    # 트랙바 값 읽기
    filter_cfg["offset_x"] = (cv2.getTrackbarPos("X Offset", "FaceMesh Filter") - 100) / 100
    filter_cfg["offset_y"] = (cv2.getTrackbarPos("Y Offset", "FaceMesh Filter") - 100) / 100
    filter_cfg["scale_w"] = cv2.getTrackbarPos("Scale W", "FaceMesh Filter") / 100
    filter_cfg["scale_h"] = cv2.getTrackbarPos("Scale H", "FaceMesh Filter") / 100

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape

        left_eye = (int(landmarks[33].x * w), int(landmarks[33].y * h))
        right_eye = (int(landmarks[263].x * w), int(landmarks[263].y * h))
        roll_angle = get_angle(left_eye, right_eye)

        face_width = int(abs(right_eye[0] - left_eye[0]) * 2)
        face_height = int(face_width * 0.6)

        anchor = landmarks[filter_cfg["anchor"]]
        anchor_x, anchor_y = int(anchor.x * w), int(anchor.y * h)

        img = cv2.imread(filter_cfg["path"], cv2.IMREAD_UNCHANGED)
        if img is not None:
            new_w = int(face_width * filter_cfg["scale_w"])
            new_h = int(face_height * filter_cfg["scale_h"])
            img = cv2.resize(img, (new_w, new_h))

            if filter_cfg["rotate"]:
                img = rotate_image(img, -roll_angle)

            pos_x = int(anchor_x + filter_cfg["offset_x"] * face_width)
            pos_y = int(anchor_y + filter_cfg["offset_y"] * face_height)

            frame = overlay_transparent(frame, img, pos_x, pos_y)

    cv2.imshow("FaceMesh Filter", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        # 현재 설정 저장
        with open("filter_config.json", "w", encoding="utf-8") as f:
            json.dump(filter_cfg, f, indent=4)
        print("✅ 설정이 filter_config.json 파일에 저장됨!")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
