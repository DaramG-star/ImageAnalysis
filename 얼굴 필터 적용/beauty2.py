import cv2
import numpy as np

def adjust_gamma(image, gamma=1.0):
    if gamma <= 0:
        gamma = 0.01
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

def adjust_contrast(image, alpha=1.0):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=0)

def get_skin_mask(image):
    img_float = image.astype(np.float32) / 255.0
    r, g, b = img_float[:,:,2], img_float[:,:,1], img_float[:,:,0]

    mask_rgb = (r > 0.3725) & (g > 0.1568) & (b > 0.0784) & \
               (r > b) & ((np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])) > 0.0588) & \
               (np.abs(r - g) > 0.0588)

    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    mask_ycrcb = (133 < cr) & (cr < 173) & (77 < cb) & (cb < 127)

    return (mask_rgb & mask_ycrcb).astype(np.uint8)

def beautify_face(image, gamma=1.5, contrast=1.2, smoothness=30, saturation=1.1):
    smoothed = cv2.edgePreservingFilter(image, flags=1, sigma_s=smoothness, sigma_r=0.2)

    skin_mask = get_skin_mask(image)
    skin_mask_blur = cv2.GaussianBlur(skin_mask.astype(np.float32), (5, 5), 0)
    skin_mask_blur = np.clip(skin_mask_blur, 0, 1)[..., np.newaxis]

    blended = (smoothed.astype(np.float32) * skin_mask_blur +
               image.astype(np.float32) * (1 - skin_mask_blur)).astype(np.uint8)

    hsv = cv2.cvtColor(blended, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
    beautified = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    beautified = adjust_gamma(beautified, gamma)
    beautified = adjust_contrast(beautified, contrast)

    return beautified


# --- 메인 실행 ---
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 웹캠을 열 수 없습니다.")
        exit()

    window_name = 'Real-time Beautify Filter'
    cv2.namedWindow(window_name)

    # 트랙바 추가
    cv2.createTrackbar('Gamma', window_name, 150, 300, lambda x: None)       # 1.0~3.0
    cv2.createTrackbar('Contrast', window_name, 120, 300, lambda x: None)    # 1.2~3.0
    cv2.createTrackbar('Smoothness', window_name, 30, 100, lambda x: None)   # sigma_s
    cv2.createTrackbar('Saturation', window_name, 110, 200, lambda x: None)  # 1.1~2.0

    print("🎥 실시간 뷰티필터 실행 중... 종료: q")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 프레임을 읽을 수 없습니다.")
            break

        gamma_val = cv2.getTrackbarPos('Gamma', window_name) / 100.0
        contrast_val = cv2.getTrackbarPos('Contrast', window_name) / 100.0
        smooth_val = cv2.getTrackbarPos('Smoothness', window_name)
        saturation_val = cv2.getTrackbarPos('Saturation', window_name) / 100.0

        beautified = beautify_face(frame,
                                   gamma=gamma_val,
                                   contrast=contrast_val,
                                   smoothness=smooth_val,
                                   saturation=saturation_val)

        combined = np.hstack((frame, beautified))
        cv2.imshow(window_name, combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
