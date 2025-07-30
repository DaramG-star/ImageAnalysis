import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

LIPS_IDX = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
            308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / max(gamma, 0.01)
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

def adjust_contrast(image, alpha=1.0):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=0)

def get_skin_mask(image):
    img_float = image.astype(np.float32) / 255.0
    r, g, b = img_float[:,:,2], img_float[:,:,1], img_float[:,:,0]
    mask_rgb = (r > 0.37) & (g > 0.15) & (b > 0.07) & (r > b) & \
               ((np.max(img_float, axis=2) - np.min(img_float, axis=2)) > 0.05) & \
               (np.abs(r - g) > 0.05)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    mask_ycrcb = (133 < cr) & (cr < 173) & (77 < cb) & (cb < 127)
    return (mask_rgb & mask_ycrcb).astype(np.uint8)

def beautify_face(image, gamma=1.5, contrast=1.2, smoothness=20, saturation=1.1):
    smoothed = cv2.bilateralFilter(image, d=7, sigmaColor=smoothness, sigmaSpace=smoothness)
    skin_mask = get_skin_mask(image)
    mask_blur = cv2.GaussianBlur(skin_mask.astype(np.float32), (3, 3), 0)[..., np.newaxis]
    blended = (smoothed.astype(np.float32) * mask_blur + image.astype(np.float32) * (1 - mask_blur)).astype(np.uint8)
    hsv = cv2.cvtColor(blended, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
    beautified = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return adjust_contrast(adjust_gamma(beautified, gamma), contrast)

def draw_lip(frame, landmarks, alpha=0.5, blur_size=15, color=(0,0,255)):
    h, w, _ = frame.shape
    points = np.array([(int(lm.x * w), int(lm.y * h)) for i, lm in enumerate(landmarks) if i in LIPS_IDX], dtype=np.int32)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(points), 255)
    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 7)
    mask_f = (mask.astype(float) / 255.0) * alpha
    color_layer = np.full_like(frame, color, dtype=np.uint8)
    return (frame.astype(float) * (1 - mask_f[..., None]) + color_layer.astype(float) * mask_f[..., None]).astype(np.uint8)

# ---------------- 메인 ----------------
cap = cv2.VideoCapture(0)
window_name = 'Beautify + Lip Color'
cv2.namedWindow(window_name)

cv2.createTrackbar('Gamma', window_name, 150, 300, lambda x: None)
cv2.createTrackbar('Contrast', window_name, 120, 300, lambda x: None)
cv2.createTrackbar('Smoothness', window_name, 20, 100, lambda x: None)
cv2.createTrackbar('Saturation', window_name, 110, 200, lambda x: None)
cv2.createTrackbar('Lip Alpha', window_name, 40, 100, lambda x: None)
cv2.createTrackbar('Lip Blur', window_name, 15, 30, lambda x: None)

frame_count, skip_rate = 0, 2
results = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    if frame_count % skip_rate == 0:  
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

    gamma_val = cv2.getTrackbarPos('Gamma', window_name) / 100.0
    contrast_val = cv2.getTrackbarPos('Contrast', window_name) / 100.0
    smooth_val = cv2.getTrackbarPos('Smoothness', window_name)
    saturation_val = cv2.getTrackbarPos('Saturation', window_name) / 100.0
    lip_alpha = cv2.getTrackbarPos('Lip Alpha', window_name) / 100.0
    lip_blur = cv2.getTrackbarPos('Lip Blur', window_name)

    beautified = beautify_face(frame, gamma=gamma_val, contrast=contrast_val,
                               smoothness=smooth_val, saturation=saturation_val)

    final_frame = beautified
    if results and results.multi_face_landmarks:
        final_frame = draw_lip(beautified, results.multi_face_landmarks[0].landmark,
                               alpha=lip_alpha, blur_size=lip_blur)

    combined = np.hstack((frame, final_frame))
    cv2.imshow(window_name, combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
