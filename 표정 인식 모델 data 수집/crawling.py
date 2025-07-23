import os
import cv2
from icrawler.builtin import GoogleImageCrawler

# ---------- 설정 ----------
keywords = [
    'thinking face person',
    'serious face expression',
    'woman thinking hard',
    'deep in thought face',
    '고민하는 얼굴',
    '진지한 표정',
    '멍한 얼굴',
    '사람이 생각하는 얼굴'
]

SAVE_ROOT = 'crawled/thinking'
MAX_PER_KEYWORD = 1000  # 키워드당 최대 수

# 얼굴 검출기 로드 (OpenCV 내장 모델)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

total_valid = 0

for keyword in keywords:
    folder_name = keyword.replace(" ", "_")[:25]
    save_dir = os.path.join(SAVE_ROOT, folder_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"🔍 크롤링: {keyword} → {save_dir}")
    crawler = GoogleImageCrawler(storage={'root_dir': save_dir})
    crawler.crawl(keyword=keyword, max_num=MAX_PER_KEYWORD)

    # 얼굴 필터링
    valid_count = 0
    for fname in os.listdir(save_dir):
        fpath = os.path.join(save_dir, fname)
        img = cv2.imread(fpath)

        if img is None:
            os.remove(fpath)
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

        if len(faces) == 0:
            os.remove(fpath)
        else:
            valid_count += 1

    print(f"✅ 얼굴 포함된 이미지: {valid_count}장\n")
    total_valid += valid_count

print(f"🎉 전체 얼굴 포함된 고민 표정 이미지 수: {total_valid}장")
