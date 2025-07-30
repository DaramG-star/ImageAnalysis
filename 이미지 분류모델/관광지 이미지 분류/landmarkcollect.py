import os
from icrawler.builtin import BingImageCrawler

keywords = ['Cheomseongdae','Dabotap', '불국사 삼층석탑', '경복궁', '석굴암', '창덕궁', '수원 화성', '남한산성', '숭례문', 'DDP', '세종대왕상', '이순신동상', '롯데타워']  # 원하는 문화유산 리스트
SAVE_ROOT = 'heritage_images'
MAX_PER_KEYWORD = 1000  # 키워드당 이미지 수

for keyword in keywords:
    # 폴더명은 영어로 변환하거나, 그냥 keyword 그대로 써도 됨
    folder_name = keyword  
    save_dir = os.path.join(SAVE_ROOT, folder_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"🔍 크롤링: {keyword} → {save_dir}")
    crawler = BingImageCrawler(storage={'root_dir': save_dir})
    crawler.crawl(keyword=keyword, max_num=MAX_PER_KEYWORD)

print("✅ 모든 키워드 크롤링 완료!")
