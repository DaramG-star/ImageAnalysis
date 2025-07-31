import cv2
import mediapipe as mp
import numpy as np
import math

# Mediapipe FaceMesh 설정
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7
)

# 필터 이미지 로드
moustache = cv2.imread("OpenCV-Face-Filters/moustache.png", cv2.IMREAD_UNCHANGED)
hat = cv2.imread("OpenCV-Face-Filters/cowboy_hat.png", cv2.IMREAD_UNCHANGED)

# 이미지를 회전시키는 함수
def rotate_image(image, angle):
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(0, 0, 0, 0))

# PNG 이미지를 배경 위에 오버레이하는 함수
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

    b, g, r, a = cv2.split(overlay)
    mask = a / 255.0

    for c in range(3):
        background[y:y+h, x:x+w, c] = \
            (1 - mask) * background[y:y+h, x:x+w, c] + mask * overlay[:h, :w, c]
    return background

# 두 점 사이 각도 계산
def get_angle(p1, p2):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    return math.degrees(math.atan2(dy, dx))

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape

        # 눈 좌표
        left_eye = (int(landmarks[33].x * w), int(landmarks[33].y * h))
        right_eye = (int(landmarks[263].x * w), int(landmarks[263].y * h))

        # 코 중앙
        nose = (int(landmarks[1].x * w), int(landmarks[1].y * h))

        # 윗입술 중앙 (콧수염 위치 기준점)
        upper_lip = (int(landmarks[13].x * w), int(landmarks[13].y * h))

        # 얼굴 회전 각도
        roll_angle = get_angle(left_eye, right_eye)

        # 얼굴 너비/높이 추정
        face_width = int(abs(right_eye[0] - left_eye[0]) * 2)
        face_height = int(face_width * 0.6)

        # ✅ 모자 필터
        rotated_hat = rotate_image(
            cv2.resize(hat, (face_width, face_height)),
            -roll_angle
        )
        frame = overlay_transparent(
            frame, rotated_hat,
            nose[0] - face_width // 2,
            nose[1] - face_height * 2
        )

        # ✅ 콧수염 필터 (윗입술 기준으로 위치 조정)
        rotated_mst = rotate_image(
            cv2.resize(moustache, (face_width // 2, face_height // 3)),
            -roll_angle
        )
        mst_x = upper_lip[0] - (face_width // 4)
        mst_y = upper_lip[1] - (face_height // 6)  # 살짝 위로 올림
        frame = overlay_transparent(frame, rotated_mst, mst_x, mst_y)

    cv2.imshow("FaceMesh Filter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()