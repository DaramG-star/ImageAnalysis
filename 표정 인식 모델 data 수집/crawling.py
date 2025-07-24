import os
import re
import cv2
from icrawler.builtin import GoogleImageCrawler

# --------- 설정 ---------
keywords = ['엽사']
SAVE_ROOT = 'crawled'
MAX_PER_KEYWORD = 100

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
total_valid = 0

for keyword in keywords:
    # ✅ 한글 키워드는 검색용으로만 사용, 폴더명은 안전한 영문으로
    folder_name = 'ugly'  # ← 폴더명은 영문으로 고정!
    save_dir = os.path.join(SAVE_ROOT, folder_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"🔍 크롤링: '{keyword}' → {save_dir}")
    crawler = GoogleImageCrawler(storage={'root_dir': save_dir})
    crawler.crawl(keyword=keyword, max_num=MAX_PER_KEYWORD)

    # ✅ 얼굴 필터링
    valid_count = 0
    for fname in os.listdir(save_dir):
        fpath = os.path.join(save_dir, fname)

        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            os.remove(fpath)
            continue

        try:
            img = cv2.imread(fpath)
            if img is None or img.shape[0] < 50 or img.shape[1] < 50:
                os.remove(fpath)
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=2)

            if len(faces) == 0:
                os.remove(fpath)
            else:
                valid_count += 1
        except Exception as e:
            print(f"⚠️ {fname} 삭제됨: {e}")
            os.remove(fpath)

    print(f"✅ 얼굴 포함된 이미지: {valid_count}장\n")
    total_valid += valid_count

print(f"🎉 전체 얼굴 포함된 하품 이미지 수: {total_valid}장")
