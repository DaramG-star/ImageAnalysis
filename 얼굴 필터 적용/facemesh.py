import cv2
import mediapipe as mp

# MediaPipe 관련 모듈 초기화
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# Face Mesh 모델 설정
# static_image_mode=False : 비디오 스트림 처리용
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True, # 눈, 입술 주변 랜드마크 정교화
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# 웹캠 시작
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # 성능 향상을 위해 이미지를 읽기 전용으로 설정
    frame.flags.writeable = False
    # BGR 이미지를 RGB로 변환
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Face Mesh 처리 실행
    results = face_mesh.process(rgb_frame)

    # 다시 이미지를 쓰기 가능으로 변경
    frame.flags.writeable = True

    # 얼굴이 감지되면 메쉬 그리기
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # mp_drawing.draw_landmarks 함수를 사용하여 메쉬를 그림
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                # FACEMESH_TESSELLATION: 얼굴 전체의 삼각형 메쉬 연결 정보
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None, # 랜드마크 점은 그리지 않음
                # 기본으로 제공되는 메쉬 그리기 스타일 사용
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

    # 좌우 반전하여 일반적인 웹캠 뷰로 출력
    cv2.imshow('MediaPipe Face Mesh', cv2.flip(frame, 1))

    if cv2.waitKey(5) & 0xFF == 27: # ESC 키를 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()