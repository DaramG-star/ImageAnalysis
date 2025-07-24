import cv2
import os

# 사용할 표정 라벨 리스트
expressions = ['surprise', 'serious', 'laugh', 'ugly', 'yawn']
current_idx = 0
img_counts = {}
for label in expressions:
    folder = f'face_data/{label}'
    os.makedirs(folder, exist_ok=True)
    img_counts[label] = len([f for f in os.listdir(folder) if f.endswith('.jpg')])

cap = cv2.VideoCapture(0)
print("🎮 'a' ← 이전 라벨, 'd' → 다음 라벨, 's' 저장, 'q' 종료")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    label = expressions[current_idx]
    count = img_counts[label]

    # 텍스트 표시
    cv2.putText(frame, f"Label: {label} | Saved: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, "Press [a] prev | [d] next | [s] save | [q] quit", (10, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    cv2.imshow("Collecting Expressions", frame)
    key = cv2.waitKey(1)

    if key == ord('a'):  # ← previous
        current_idx = (current_idx - 1) % len(expressions)
    elif key == ord('d'):  # → next
        current_idx = (current_idx + 1) % len(expressions)
    elif key == ord('s'):  # 저장
        save_path = f"face_data/{label}/{label}_000{count:03d}.jpg"
        cv2.imwrite(save_path, frame)
        img_counts[label] += 1
        print(f"✔ Saved: {save_path}")
    elif key == ord('q'):  # 종료
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 종료되었습니다.")
