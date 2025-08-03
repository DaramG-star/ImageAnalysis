# face_tagging_app.py
# Flask + DeepFace: 유사/흐림 분석 + 동일 인물 태깅 (ArcFace + RetinaFace)
# ---------------------------------------------------------------
import os
import shutil
import cv2
import numpy as np
from PIL import Image
import imagehash
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from deepface import DeepFace
from scipy.spatial import distance as dist

# ===== DeepFace 전역 설정 =====
MODEL_NAME       = "Facenet"
DETECTOR_BACKEND = "retinaface"
METRIC           = "cosine"
COS_TH           = 0.50

def get_embedding(img, enforce_detection=True):
    """
    img : 파일 경로(str) 또는 numpy array (DeepFace가 둘 다 허용)
    enforce_detection : 얼굴 미탐지 시 예외 발생 여부
    """
    rep = DeepFace.represent(
        img_path=img,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=enforce_detection,
        align=True,
    )
    return rep[0]["embedding"]

def cosine(a, b):
    return dist.cosine(a, b)

# ===== Flask & 폴더 =====
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER

os.makedirs(os.path.join(UPLOAD_FOLDER, 'sources'), exist_ok=True)
os.makedirs(os.path.join(UPLOAD_FOLDER, 'targets'), exist_ok=True)
os.makedirs(os.path.join(STATIC_FOLDER, 'results'), exist_ok=True)

# ===== 1) 유사/흐림 분석 (수정된 부분) =====
def get_image_hash(image_path):
    try:
        return imagehash.phash(Image.open(image_path).convert("RGB"))
    except (IOError, OSError):
        return None

def is_blurry(image_path, threshold=100):
    """
    사진에서 얼굴을 먼저 찾고, 얼굴 영역의 선명도를 측정하는 함수.
    얼굴이 없으면 전체 이미지의 선명도를 측정.
    """
    try:
        # DeepFace를 사용해 얼굴 부분만 추출
        face_objs = DeepFace.extract_faces(
            img_path=image_path,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True, # 얼굴이 없으면 예외 발생
            align=False
        )
        
        # 여러 얼굴 중 가장 선명한 얼굴을 기준으로 판단
        max_lap_var = 0
        for face_obj in face_objs:
            face = face_obj['face'] # 추출된 얼굴 이미지 (numpy array)
            # BGR -> GRAY 변환을 위해 채널 수 확인
            if face.shape[2] == 3:
                 gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            else: # 이미 Grayscale이거나 다른 형식일 경우
                 gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)

            lap = cv2.Laplacian(gray, cv2.CV_64F)
            lap_var = lap.var()
            if lap_var > max_lap_var:
                max_lap_var = lap_var
        
        # 가장 선명한 얼굴의 점수가 기준치보다 낮으면 '흐림'으로 판단
        return max_lap_var < threshold, max_lap_var

    except ValueError: # 얼굴을 찾지 못한 경우
        # 기존 방식대로 전체 이미지를 분석
        img = cv2.imread(image_path)
        if img is None:
            return False, 0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = lap.var()
        return lap_var < threshold, lap_var

def analyze_similarity_and_blur(folder, hash_th=5, blur_th=100):
    valid_exts = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
    files = [f for f in os.listdir(folder) if f.lower().endswith(valid_exts)]
    hashes = {f: get_image_hash(os.path.join(folder, f)) for f in files}
    results = []

    # 유사(해시)
    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            f1, f2 = files[i], files[j]
            if hashes[f1] is None or hashes[f2] is None:
                continue
            diff = hashes[f1] - hashes[f2]
            if diff <= hash_th:
                results.append({'Type': '유사', 'File1': f1, 'File2': f2, 'Value': f'해시차이: {diff}'})

    # 흐림 (수정된 is_blurry 함수 사용)
    for f in files:
        # 이제 blur_th는 얼굴 영역의 선명도 기준이 됩니다.
        # 인물이 흔들린 사진을 찾으려면 기준을 좀 더 높여도 좋습니다 (예: 100~150)
        blur, var = is_blurry(os.path.join(folder, f), blur_th)
        if blur:
            results.append({'Type': '흐림', 'File1': f, 'File2': '-', 'Value': f'얼굴 선명도: {var:.2f}'})
    return results

# ===== 2) 얼굴 태깅 =====
def _area_to_xywh(area, img_shape=None):
    if all(k in area for k in ("x", "y", "w", "h")):
        return area["x"], area["y"], area["w"], area["h"]
    if all(k in area for k in ("left", "top", "right", "bottom")):
        x1, y1, x2, y2 = area["left"], area["top"], area["right"], area["bottom"]
        return x1, y1, max(0, x2 - x1), max(0, y2 - y1)
    return area.get("x", 0), area.get("y", 0), area.get("w", 0), area.get("h", 0)

def tag_faces_in_images(target_dir, source_dir, result_dir):
    target_embeddings = {}
    for filename in os.listdir(target_dir):
        tpath = os.path.join(target_dir, filename)
        try:
            emb = get_embedding(tpath, enforce_detection=True)
            target_embeddings[filename] = emb
            print(f"[OK] 기준 인물 분석: {filename}")
        except Exception as e:
            print(f"[ERR] 기준 인물 실패: {filename} ({e})")

    if not target_embeddings:
        return []

    tagged_web_paths = []
    for filename in os.listdir(source_dir):
        spath = os.path.join(source_dir, filename)
        try:
            img = cv2.imread(spath)
            if img is None: continue

            faces = DeepFace.extract_faces(
                img_path=spath,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False,
                align=True
            )
            if not faces: continue

            modified = False
            for face in faces:
                try:
                    face_emb = get_embedding(face["face"], enforce_detection=False)
                except Exception as e:
                    print(f"[WARN] 임베딩 실패: {filename} ({e})")
                    continue
                
                best_name, best_dist = None, 1.0
                for tname, temb in target_embeddings.items():
                    d = cosine(temb, face_emb)
                    if d < best_dist:
                        best_dist, best_name = d, tname

                if best_dist < COS_TH:
                    area = face.get("facial_area", {})
                    x, y, w, h = _area_to_xywh(area, img.shape if img is not None else None)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    tag = f"{os.path.splitext(best_name)[0]} ({best_dist:.2f})"
                    cv2.putText(img, tag, (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    modified = True

            if modified:
                out_name = f"tagged_{filename}"
                out_path = os.path.join(result_dir, out_name)
                cv2.imwrite(out_path, img)
                tagged_web_paths.append(os.path.join('results', out_name).replace('\\', '/'))
        except Exception as e:
            print(f"[ERR] 소스 {filename}: {e}")

    return tagged_web_paths

# ===== 3) 유틸 =====
def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'삭제 실패: {file_path} - {e}')

# ===== 4) Flask 라우트 =====
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze_similarity', methods=['POST'])
def analyze_similarity_route():
    if 'images' not in request.files:
        return redirect(url_for('index'))

    src_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'sources')
    clear_folder(src_dir)

    for f in request.files.getlist('images'):
        if f and f.filename:
            f.save(os.path.join(src_dir, secure_filename(f.filename)))

    # 여기서 blur_th 값을 조절하여 '흔들린 사진'의 기준을 정할 수 있습니다.
    results = analyze_similarity_and_blur(src_dir, blur_th=120) 
    return render_template('results.html', results=results)

@app.route('/tag_faces', methods=['POST'])
def tag_faces_route():
    if 'target_images' not in request.files or 'source_images' not in request.files:
        return redirect(url_for('index'))

    tgt_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'targets')
    src_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'sources')
    res_dir = os.path.join(app.config['STATIC_FOLDER'], 'results')

    clear_folder(tgt_dir)
    clear_folder(src_dir)
    clear_folder(res_dir)

    for f in request.files.getlist('target_images'):
        if f and f.filename:
            f.save(os.path.join(tgt_dir, secure_filename(f.filename)))

    for f in request.files.getlist('source_images'):
        if f and f.filename:
            f.save(os.path.join(src_dir, secure_filename(f.filename)))

    tagged_images = tag_faces_in_images(tgt_dir, src_dir, res_dir)
    return render_template('tagging_results.html', tagged_images=tagged_images)

# ===== 5) 실행 =====
if __name__ == "__main__":
    app.run(debug=True)