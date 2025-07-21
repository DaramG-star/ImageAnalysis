import cv2
import os
import csv
import mediapipe as mp

label_names = ["bad", "fist", "good", "heart", "none", "ok", "open_palm", "promise", "victory", "gun", "rock"]
csv_path = "hand_landmarks.csv"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# CSV 헤더 만들기
header = ["label"]
for i in range(21):
    header += [f"x{i}", f"y{i}", f"z{i}"]

# 새로 시작할 경우 CSV 생성
if not os.path.exists(csv_path):
    with open(csv_path, mode='w', newline='') as f:
        csv.writer(f).writerow(header)

current_index = 0
cap = cv2.VideoCapture(0)
print("\n←/→: 라벨 이동 / [s]: 저장 / [q]: 종료")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    label = label_names[current_index]

    # 라벨 화면에 표시
    cv2.putText(frame, f"Label: [{label}]", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 상대좌표 계산
            coords = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            base_x, base_y, base_z = coords[0]  # 손목 기준

            relative = []
            for x, y, z in coords:
                relative.extend([x - base_x, y - base_y, z - base_z])

            row = [label] + relative

            with open(csv_path, mode='a', newline='') as f:
                csv.writer(f).writerow(row)
                print(f"✅ 저장: {label} ({len(relative)//3} points)")

    cv2.imshow("Hand Landmark Collector", frame)

    key = cv2.waitKey(10)

    # ← (또는 a)
    if key == 81 or key == ord('a'):
        current_index = (current_index - 1) % len(label_names)
        print(f"⬅ 라벨 이동: {label_names[current_index]}")

    # → (또는 d)
    elif key == 83 or key == ord('d'):
        current_index = (current_index + 1) % len(label_names)
        print(f"➡ 라벨 이동: {label_names[current_index]}")

    elif key == ord('q'):
        break
    
    elif key == ord('s'):
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                coords = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                base_x, base_y, base_z = coords[0]

                relative = []
                for x, y, z in coords:
                    relative.extend([x - base_x, y - base_y, z - base_z])

                row = [label] + relative

                with open(csv_path, mode='a', newline='') as f:
                    csv.writer(f).writerow(row)
                    print(f"✅ 저장됨: {label} ({len(relative)//3} points)")
        else:
            print("❌ 손이 감지되지 않았습니다.")
        # 저장은 위에서 자동 처리됨

cap.release()
cv2.destroyAllWindows()
