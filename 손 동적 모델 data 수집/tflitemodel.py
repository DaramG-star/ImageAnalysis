import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import time

model_path = "moving_hands/gesture_rnn_model5.tflite"
gestures = ['fire', 'hi', 'hit', 'none', 'nono', 'nyan', 'shot']

images = {
    'fire': cv2.imread('moving_hands/fire_effect.png', cv2.IMREAD_UNCHANGED),
    'hi': cv2.imread('moving_hands/hi.png', cv2.IMREAD_UNCHANGED),
    'hit': cv2.imread('moving_hands/fist.png', cv2.IMREAD_UNCHANGED),
    'nono': cv2.imread('moving_hands/nono.png', cv2.IMREAD_UNCHANGED),
    'nyan': cv2.imread('moving_hands/nyan2.png', cv2.IMREAD_UNCHANGED),
    'shot': cv2.imread('moving_hands/heart.png', cv2.IMREAD_UNCHANGED)
}

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

def overlay_image(bg, overlay, x, y):
    if overlay is None:
        return bg
    h, w = overlay.shape[:2]
    # Ensure overlay coordinates are within the background image boundaries
    x = max(0, min(x, bg.shape[1] - w))
    y = max(0, min(y, bg.shape[0] - h))

    alpha = overlay[:, :, 3] / 255.0
    for c in range(3):
        bg[y:y+h, x:x+w, c] = (1 - alpha) * bg[y:y+h, x:x+w, c] + alpha * overlay[:, :, c]
    return bg

def extract_keypoints(results):
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        wrist = hand.landmark[0]
        return np.array([[lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z] for lm in hand.landmark]).flatten()
    return np.zeros(21 * 3)

cap = cv2.VideoCapture(0)
sequence = []

display_label = None
display_start_time = 0
display_duration = 2.0 # Display image for 2 seconds

last_gesture_time = {} # To store the last time each gesture was recognized
gesture_cooldown = 5.0 # 5 seconds cooldown for the same gesture

confidence_threshold = 0.8
move_threshold = 0.004
prev_wrist = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    current_time = time.time()

    # Check if a display_label is active and its display duration has passed
    if display_label is not None and current_time - display_start_time > display_duration:
        display_label = None # Deactivate display

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        wrist = hand.landmark[0]
        index_tip = hand.landmark[8]

        # Calculate hand movement
        if prev_wrist is not None:
            dx = wrist.x - prev_wrist.x
            dy = wrist.y - prev_wrist.y
            dz = wrist.z - prev_wrist.z
            move_distance = (dx**2 + dy**2 + dz**2) ** 0.5
        else:
            move_distance = 0

        prev_wrist = wrist

        # If hand is barely moving, skip recognition
        if move_distance < move_threshold:
            # If there's an active display, show it
            if display_label in images and images[display_label] is not None:
                h, w, _ = img.shape
                cx_wrist, cy_wrist = int(wrist.x * w), int(wrist.y * h)
                cx_index, cy_index = int(index_tip.x * w), int(index_tip.y * h)
                overlay_and_display(img, display_label, cx_wrist, cy_wrist, cx_index, cy_index, images)
            cv2.imshow("Gesture Recognition with Hold", img)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            continue # Skip further processing if not moving

        keypoints = extract_keypoints(result).astype(np.float32)
        sequence.append(keypoints)
        sequence = sequence[-30:] # Keep only the last 30 frames

        if len(sequence) == 30:
            input_data = np.expand_dims(sequence, axis=0).astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])

            predicted_label = gestures[np.argmax(output)]
            confidence = np.max(output)

            # Only consider valid gestures with sufficient confidence
            if confidence < confidence_threshold or predicted_label == 'none':
                predicted_label = 'none' # Treat low confidence as 'none'

            # 1. Handle display_label (image holding)
            if display_label is None and predicted_label != 'none':
                # Check for cooldown for the predicted gesture
                if predicted_label not in last_gesture_time or \
                   (current_time - last_gesture_time[predicted_label] > gesture_cooldown):
                    display_label = predicted_label
                    display_start_time = current_time
                    last_gesture_time[predicted_label] = current_time # Update last recognized time

            # 2. If a display_label is currently active, prevent new recognition
            # This is implicitly handled by the `if display_label is None` check above.
            # If `display_label` is not None, the new `predicted_label` won't set `display_label` again.

    # Function to overlay image based on display_label
    def overlay_and_display(img_frame, label, wrist_x, wrist_y, index_x, index_y, image_dict):
        if label in image_dict and image_dict[label] is not None:
            overlay = cv2.resize(image_dict[label], (120, 120))
            if label == 'fire':
                img_frame = overlay_image(img_frame, overlay, wrist_x - 60, wrist_y - 150)
            elif label == 'hi':
                img_frame = overlay_image(img_frame, overlay, wrist_x - 60, wrist_y - 180)
            elif label == 'hit':
                offsets = [(-60, -60), (0, -60), (60, -60),
                           (-60, 0), (0, 0), (60, 0),
                           (-60, 60), (0, 60), (60, 60)]
                small = cv2.resize(image_dict[label], (60, 60))
                for dx, dy in offsets:
                    img_frame = overlay_image(img_frame, small, wrist_x + dx, wrist_y + dy)
            elif label == 'nono':
                img_frame = overlay_image(img_frame, overlay, wrist_x - 150, wrist_y - 50)
            elif label == 'nyan':
                img_frame = overlay_image(img_frame, overlay, wrist_x + 80, wrist_y - 50)
            elif label == 'shot':
                img_frame = overlay_image(img_frame, overlay, index_x - 20, index_y - 100)
        return img_frame

    # Always display the current 'display_label' if it's active
    if display_label in images and images[display_label] is not None:
        h, w, _ = img.shape
        # Ensure wrist and index_tip are available for display positioning
        if result.multi_hand_landmarks:
            wrist = result.multi_hand_landmarks[0].landmark[0]
            index_tip = result.multi_hand_landmarks[0].landmark[8]
            cx_wrist, cy_wrist = int(wrist.x * w), int(wrist.y * h)
            cx_index, cy_index = int(index_tip.x * w), int(index_tip.y * h)
            img = overlay_and_display(img, display_label, cx_wrist, cy_wrist, cx_index, cy_index, images)
    
    cv2.imshow('Gesture Recognition with Hold', img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()