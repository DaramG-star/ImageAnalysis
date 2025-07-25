import cv2
import numpy as np

def adjust_gamma(image, gamma=1.0):
    """
    감마 보정을 사용하여 이미지 밝기를 조절합니다.
    gamma < 1.0 이면 이미지가 어두워지고, gamma > 1.0 이면 밝아집니다.
    """
    # 감마 값이 0이 되는 것을 방지하여 오류를 막습니다.
    if gamma == 0:
        gamma = 0.01
        
    inv_gamma = 1.0 / gamma
    # 룩업 테이블 생성
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # 룩업 테이블을 사용하여 감마 보정 적용
    return cv2.LUT(image, table)


# ⭐️ gamma 값을 인자로 받도록 함수 수정
def beautify_face(image, gamma=1.5):
    """
    주어진 이미지에 대해 뷰티 필터를 적용합니다.
    (밝기 조절 부분이 감마 보정으로 변경되었습니다)
    """

    # 1. Bilateral Filter 적용
    bilateral = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    # 2. Canny Edge Detection 적용
    edges = cv2.Canny(image, 100, 200)

    # 3. 필터 결합 (피부 영역에만 블러 적용)
    img_float = image.astype(np.float32) / 255.0
    r, g, b = img_float[:,:,2], img_float[:,:,1], img_float[:,:,0]
    
    skin_mask = (r > 0.3725) & (g > 0.1568) & (b > 0.0784) & (r > b) & \
                ((np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)) > 0.0588) & \
                (np.abs(r - g) > 0.0588)
    
    non_edge_mask = edges < 50
    final_mask = (skin_mask & non_edge_mask).astype(np.uint8)
    final_mask_3channel = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)

    beautified_image = np.where(final_mask_3channel == 1, bilateral, image)
    
    # 4. 밝기 및 채도 조절
    # 채도를 10% 증가시킵니다.
    hsv = cv2.cvtColor(beautified_image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.1, 0, 255)
    beautified_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # ⭐️ 인자로 받은 gamma 값을 사용하여 밝기 조절
    final_result = adjust_gamma(beautified_image, gamma=gamma)

    return final_result


# --- 메인 실행 부분 ---
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        exit()
        
    window_name = 'Beautify Face Demo - Original vs. Result'
    # ⭐️ 윈도우를 먼저 생성합니다.
    cv2.namedWindow(window_name)

    # ⭐️ 'Gamma'라는 이름의 조절 바를 생성합니다.
    # 범위는 0-300, 기본값은 150 (gamma 1.5에 해당)
    cv2.createTrackbar('Gamma', window_name, 150, 300, lambda x:None)

    print("프로그램 종료: 'q' 키를 누르세요.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        # ⭐️ 조절 바에서 현재 감마 값을 읽어옵니다.
        gamma_val_int = cv2.getTrackbarPos('Gamma', window_name)
        # 정수 값을 float으로 변환 (150 -> 1.5)
        gamma_val_float = gamma_val_int / 100.0

        # ⭐️ 뷰티 필터에 감마 값을 전달합니다.
        beautified_frame = beautify_face(frame, gamma=gamma_val_float)
        
        combined_view = np.hstack((frame, beautified_frame))
        cv2.imshow(window_name, combined_view)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()