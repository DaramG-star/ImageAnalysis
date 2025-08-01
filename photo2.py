import cv2
import numpy as np
from PIL import Image
import imagehash
import os
import csv
from tqdm import tqdm

# 이미지 해시로 유사도 계산 (pHash)
def get_image_hash(image_path):
    """지정된 경로의 이미지에 대한 pHash 값을 반환합니다."""
    try:
        img = Image.open(image_path).convert("RGB")
        return imagehash.phash(img)
    except (IOError, OSError) as e:
        print(f"오류: {image_path} 파일 처리 중 오류 발생 - {e}")
        return None

# 흐림(블러) 판단 (Laplacian 분산)
def is_blurry(image_path, threshold=100):
    """지정된 경로의 이미지가 흐린지 여부를 판단합니다."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return lap_var < threshold
    except cv2.error as e:
        print(f"오류: {image_path} 파일 처리 중 오류 발생 - {e}")
        return False

# 메인 실행 함수
def analyze_images(folder_path, hash_threshold=5, blur_threshold=100):
    """
    폴더 내 이미지들의 유사성 및 흐림 여부를 분석합니다.
    분석 결과는 'image_analysis_results.csv' 파일에 저장됩니다.
    """
    if not os.path.isdir(folder_path):
        print(f"오류: '{folder_path}' 폴더가 존재하지 않습니다.")
        return

    # 유효한 이미지 파일 목록 가져오기
    valid_extensions = ('.jpg', '.png', '.jpeg', '.webp', '.bmp')
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
    if not files:
        print(f"'{folder_path}' 폴더에 유효한 이미지 파일이 없습니다.")
        return

    print(f"'{folder_path}' 폴더 내 이미지 {len(files)}개 분석 시작...")

    # 이미지 해시 계산
    hashes = {}
    for f in tqdm(files, desc="이미지 해시 계산"):
        path = os.path.join(folder_path, f)
        hashes[f] = get_image_hash(path)

    results = []

    # 1. 유사 이미지 탐색
    print("\n유사 이미지 탐색 중...")
    checked = set()
    for i in tqdm(range(len(files)), desc="유사성 비교"):
        for j in range(i + 1, len(files)):
            f1, f2 = files[i], files[j]
            if (f1, f2) not in checked and hashes[f1] is not None and hashes[f2] is not None:
                diff = hashes[f1] - hashes[f2]
                if diff <= hash_threshold:
                    results.append({'Type': '유사', 'File1': f1, 'File2': f2, 'Difference': diff})
                checked.add((f1, f2))
    
    # 2. 흐림 이미지 탐색
    print("\n흐림 이미지 탐색 중...")
    for f in tqdm(files, desc="흐림 여부 판단"):
        path = os.path.join(folder_path, f)
        if is_blurry(path, blur_threshold):
            results.append({'Type': '흐림', 'File1': f, 'File2': 'N/A', 'Difference': blur_threshold})

    # CSV 파일로 결과 저장
    output_csv = "image_analysis_results.csv"
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Type', 'File1', 'File2', 'Difference']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n분석 완료! 결과가 '{output_csv}' 파일에 저장되었습니다.")

# --- 실행 ---
if __name__ == "__main__":
    image_folder = "images"  # 이미지 폴더 경로를 여기에 지정하세요
    analyze_images(image_folder)