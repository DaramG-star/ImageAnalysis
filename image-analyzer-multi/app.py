# app.py
import os
import shutil
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import models, transforms
from flask import Flask, request, render_template, send_from_directory
from insightface.app import FaceAnalysis
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 모델 준비 =====
face_app = FaceAnalysis(providers=['CUDAExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

cnn_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
cnn_model = torch.nn.Sequential(*list(cnn_model.children())[:-1])
cnn_model = cnn_model.to(device).eval()

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ===== 유틸 =====
def clear_folder(folder_path):
    for f in os.listdir(folder_path):
        fp = os.path.join(folder_path, f)
        if os.path.isfile(fp):
            os.remove(fp)

def read_img_robust(img_path):
    try:
        with open(img_path, "rb") as stream:
            bytes_data = bytearray(stream.read())
            numpy_array = np.asarray(bytes_data, dtype=np.uint8)
            return cv2.imdecode(numpy_array, cv2.IMREAD_UNCHANGED)
    except Exception as e:
        print(f"[오류] 이미지 읽기 실패 {img_path}: {e}")
        return None

def get_face_embeddings(img_path):
    img = read_img_robust(img_path)
    if img is None:
        return []
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return face_app.get(rgb)

def get_cnn_embedding(img_tensor):
    with torch.no_grad():
        return cnn_model(img_tensor.unsqueeze(0).to(device)).flatten()

def laplacian_var(image):
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return cv2.Laplacian(gray, cv2.CV_64F).var()

# ===== 분석 =====
def analyze_all(target_dir, source_dir, result_dir, blur_th=100, sim_th=0.9, tag_th=0.6):
    valid_exts = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
    files = [f for f in os.listdir(source_dir) if f.lower().endswith(valid_exts)]
    results = []
    embeddings = []

    for f in files:
        path = os.path.join(source_dir, f)
        img = read_img_robust(path)
        if img is None or img.size == 0:
            print(f"[경고] 이미지 로드 실패 또는 비어있음: {f}")
            continue

        faces = face_app.get(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        lap_var = 0
        if faces:
            max_area = 0
            for face in faces:
                x1, y1, x2, y2 = map(int, face["bbox"])
                crop = img[y1:y2, x1:x2]
                if crop is None or crop.size == 0:
                    continue
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    lap_var = laplacian_var(crop)
        else:
            lap_var = laplacian_var(img)

        blur = lap_var < blur_th
        results.append({"File": f, "Blur": blur, "LaplacianVar": lap_var})

        try:
            tensor = preprocess(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).to(device)
            emb = get_cnn_embedding(tensor)
            embeddings.append((f, emb))
        except Exception as e:
            print(f"[경고] 임베딩 실패: {f} / {e}")

    groups = []
    used = set()
    for i in range(len(embeddings)):
        if embeddings[i][0] in used:
            continue
        group = [embeddings[i][0]]
        for j in range(i + 1, len(embeddings)):
            if embeddings[j][0] not in used:
                sim = F.cosine_similarity(embeddings[i][1].unsqueeze(0), embeddings[j][1].unsqueeze(0)).item()
                if sim > sim_th:
                    group.append(embeddings[j][0])
                    used.add(embeddings[j][0])
        used.add(embeddings[i][0])
        if len(group) > 1:
            groups.append(group)

    targets = []
    for f in os.listdir(target_dir):
        tpath = os.path.join(target_dir, f)
        name = os.path.splitext(f)[0]
        faces = get_face_embeddings(tpath)
        if faces:
            for idx, face in enumerate(faces):
                face_name = f"{name}_{idx}" if len(faces) > 1 else name
                targets.append((face_name, torch.tensor(face["embedding"]).to(device)))
        else:
            print(f"[경고] 기준 인물 얼굴 인식 실패: {f}")

    # tagged_images를 딕셔너리로 변경
    tagged_images = defaultdict(list)
    for f in os.listdir(source_dir):
        spath = os.path.join(source_dir, f)
        img = read_img_robust(spath)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = face_app.get(img_rgb)
        
        tagged_in_image = False
        names_found = set()

        for face in faces:
            emb = torch.tensor(face["embedding"]).to(device)
            best_name, best_dist = None, 1.0
            for tname, temb in targets:
                dist = 1 - F.cosine_similarity(temb.unsqueeze(0), emb.unsqueeze(0)).item()
                if dist < best_dist:
                    best_dist, best_name = dist, tname
            if best_dist < tag_th:
                x1, y1, x2, y2 = map(int, face["bbox"])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{best_name} ({best_dist:.2f})", (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                tagged_in_image = True
                names_found.add(best_name)

        if tagged_in_image:
            out_name = f"tagged_{f}"
            ext = os.path.splitext(f)[1]
            result, buffer = cv2.imencode(ext, img)
            if result:
                with open(os.path.join(result_dir, out_name), "wb") as file:
                    file.write(buffer)
            # 찾은 모든 이름에 대해 이미지를 추가
            for name in names_found:
                tagged_images[name].append(out_name)

    return results, groups, tagged_images

# ===== Flask 설정 =====
BASE = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE, "uploads")
STATIC_FOLDER = os.path.join(BASE, "static")
TARGET_DIR = os.path.join(UPLOAD_FOLDER, "targets")
SOURCE_DIR = os.path.join(UPLOAD_FOLDER, "sources")
RESULT_DIR = os.path.join(STATIC_FOLDER, "results")

app = Flask(__name__, static_folder=STATIC_FOLDER, template_folder=os.path.join(BASE, "templates"))
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

for p in [TARGET_DIR, SOURCE_DIR, RESULT_DIR]:
    os.makedirs(p, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze_all", methods=["POST"])
def analyze_all_route():
    clear_folder(TARGET_DIR)
    clear_folder(SOURCE_DIR)
    clear_folder(RESULT_DIR)

    target_images = request.files.getlist("target_images")
    target_names = request.form.getlist("target_names")

    for img, name in zip(target_images, target_names):
        if name.strip() and img.filename:
            filename = f"{name.strip()}.jpg"
            img.save(os.path.join(TARGET_DIR, filename))

    for f in request.files.getlist("source_images"):
        if f.filename:
            f.save(os.path.join(SOURCE_DIR, f.filename))

    results, groups, tagged_images = analyze_all(TARGET_DIR, SOURCE_DIR, RESULT_DIR)
    return render_template("tagging_results.html", results=results, groups=groups, tagged_images=tagged_images)

@app.route('/uploads/sources/<path:filename>')
def uploaded_source_file(filename):
    return send_from_directory(SOURCE_DIR, filename, as_attachment=False)

if __name__ == "__main__":
    app.run(debug=True)