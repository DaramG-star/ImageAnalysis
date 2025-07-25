import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2, numpy as np, mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True,
                                  max_num_faces=1,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

LEFT  = [33,133,160,159,158,157,173]
RIGHT = [362,263,387,386,385,384,398]

cv2.namedWindow('EyeFilter');
cv2.createTrackbar('Strength','EyeFilter',50,200,lambda x: None)
cv2.createTrackbar('Feather','EyeFilter',10,50,lambda x: None)

# 'enlarge' 함수를 아래 코드로 교체해주세요.
def enlarge(frame, lmks, idx, scale, feather):
    h, w = frame.shape[:2]
    pts = np.array([(int(lmks[i].x*w), int(lmks[i].y*h)) for i in idx], np.int32)

    x, y, ew, eh = cv2.boundingRect(pts)
    if ew==0 or eh==0: return

    eye = frame[y:y+eh, x:x+ew].copy()

    # ----- (1) 1채널 마스크 (Feather 트랙바 값 적용) -----
    mask = np.zeros((eh, ew), np.uint8)
    cv2.fillConvexPoly(mask, pts-[x,y], 255)
    
    # GaussianBlur의 커널 크기는 홀수여야 함
    if feather > 0:
        if feather % 2 == 0: feather += 1
        mask = cv2.GaussianBlur(mask, (feather, feather), 0)

    # ----- (2) 스케일 조절 -----
    new_w, new_h = int(ew*scale), int(eh*scale)
    if new_w > w or new_h > h:
        shrink = min(w/ new_w, h/ new_h) * 0.9
        new_w, new_h = int(new_w*shrink), int(new_h*shrink)
    
    eye_big  = cv2.resize(eye,  (new_w,new_h), interpolation=cv2.INTER_CUBIC)
    mask_big = cv2.resize(mask, (new_w,new_h), interpolation=cv2.INTER_LINEAR)

    # 마스크를 0~1 범위의 3채널 부동소수점 형태로 변환 (알파 블렌딩 준비)
    mask_alpha = cv2.cvtColor(mask_big, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0

    # ----- (3) 알파 블렌딩으로 직접 합성 -----
    cx, cy = x+ew//2, y+eh//2
    x_start, y_start = cx - new_w//2, cy - new_h//2
    x_end, y_end = x_start + new_w, y_start + new_h

    # 프레임 경계를 벗어나지 않도록 좌표 보정
    if x_start < 0 or y_start < 0 or x_end > w or y_end > h:
        return # 간단히 처리를 건너뜀 (더 정교한 클리핑도 가능)

    try:
        # 합성할 영역 추출
        frame_roi = frame[y_start:y_end, x_start:x_end]
        
        # 알파 블렌딩 공식 적용: blended = src * alpha + dst * (1 - alpha)
        blended = (eye_big * mask_alpha + frame_roi * (1 - mask_alpha)).astype(np.uint8)
        
        # 원본 프레임에 결과 덮어쓰기
        frame[y_start:y_end, x_start:x_end] = blended

    except Exception as e:
        print('alpha blend err:', e) # 디버깅용


cap = cv2.VideoCapture(0)
while cap.isOpened():
    ok, frame = cap.read()
    if not ok: break

    # 좌우반전 추가 (일반적인 웹캠 화면)
    frame = cv2.flip(frame, 1)

    res = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if res.multi_face_landmarks:
        lmk     = res.multi_face_landmarks[0].landmark
        val     = cv2.getTrackbarPos('Strength','EyeFilter')
        feather = cv2.getTrackbarPos('Feather','EyeFilter')
        scl     = 1.0 + val/100
        
        # 함수 호출 시 feather 값 전달
        enlarge(frame, lmk, LEFT,  scl, feather)
        enlarge(frame, lmk, RIGHT, scl, feather)

    cv2.imshow('EyeFilter', frame)
    if cv2.waitKey(1)&0xFF == ord('q'): break
cap.release(); cv2.destroyAllWindows()