import cv2
import numpy as np
from PIL import Image
import imagehash
import os

# 1️⃣ 이미지 해시로 유사도 계산 (pHash)
def get_hash(image_path):
    img = Image.open(image_path).convert("RGB")
    return imagehash.phash(img)  # pHash가 유사한 사진 구분에 강함

# 2️⃣ 흐림(블러) 판단 (Laplacian 분산)
def is_blurry(image_path, threshold=100):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap_var < threshold  # 값이 작으면 흐림

# 3️⃣ 폴더 내 모든 이미지 처리
folder = "images/"
files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg','.png','.jpeg'))]

hashes = {}
for f in files:
    path = os.path.join(folder, f)
    hashes[f] = get_hash(path)

# 4️⃣ 중복/유사 이미지 탐색
checked = set()
for f1 in files:
    for f2 in files:
        if f1 < f2 and (f1,f2) not in checked:
            diff = hashes[f1] - hashes[f2]
            if diff <= 5:  # 0=완전 동일, 1~5=매우 유사
                print(f"🔗 유사한 이미지: {f1} ↔ {f2} (diff={diff})")
            checked.add((f1,f2))

# 5️⃣ 흐림 이미지 탐색
for f in files:
    path = os.path.join(folder, f)
    if is_blurry(path):
        print(f"💤 흐림 이미지: {f}")