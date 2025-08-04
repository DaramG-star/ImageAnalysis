import os
import shutil
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import models, transforms
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from insightface.app import FaceAnalysis
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 모델 준비 =====
face_app = FaceAnalysis(providers=['CUDAExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

cnn_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
cnn_model = torch.nn.Sequential(*list(cnn_model.children())[:-1])  # FC 제거
cnn_model = cnn_model.to(device).eval()

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ===== 유틸 =====
def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

def get_face_embeddings(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"[경고] 이미지 로딩 실패: {img_path}")
        return []
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = face_app.get(img_rgb)
    return faces  # [{'embedding': ..., 'bbox': ...}, ...]

def get_cnn_embedding(img_tensor):
    with torch.no_grad():
        return cnn_model(img_tensor.unsqueeze(0).to(device)).flatten()

def laplacian_var_region(image):
    """Laplacian variance 계산"""
    if image is None:
        return 0  # 예외 처리
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return cv2.Laplacian(gray, cv2.CV_64F).var()

# ===== 흐림 + 유사 분석 =====
def analyze_images(folder, blur_th=100, sim_th=0.9):
    valid_exts = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
    files = [f for f in os.listdir(folder) if f.lower().endswith(valid_exts)]

    embeddings = []
    results = []

    for f in files:
        path = os.path.join(folder, f)
        img = cv2.imread(path)
        if img is None:
            print(f"[경고] 이미지 로딩 실패: {path}")
            continue

        faces = face_app.get(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        lap_var = 0

        if faces:
            max_area = 0
            for face in faces:
                x1, y1, x2, y2 = map(int, face["bbox"])
                face_crop = img[y1:y2, x1:x2]
                if face_crop is None or face_crop.size == 0:
                    continue
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    lap_var = laplacian_var_region(face_crop)
        else:
            lap_var = laplacian_var_region(img)

        blur = lap_var < blur_th
        results.append({"File": f, "Blur": blur, "LaplacianVar": lap_var})

        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = preprocess(img_rgb).to(device)
            emb_cnn = get_cnn_embedding(tensor)
            embeddings.append((f, emb_cnn))
        except Exception as e:
            print(f"[오류] CNN 임베딩 실패 - {f}: {e}")

    # 유사 사진 그룹화
    similar_groups = []
    used = set()

    for i in range(len(embeddings)):
        if embeddings[i][0] in used:
            continue
        group = [embeddings[i][0]]
        for j in range(i + 1, len(embeddings)):
            if embeddings[j][0] not in used:
                sim = F.cosine_similarity(
                    embeddings[i][1].unsqueeze(0), embeddings[j][1].unsqueeze(0)
                ).item()
                if sim > sim_th:
                    group.append(embeddings[j][0])
                    used.add(embeddings[j][0])
        used.add(embeddings[i][0])
        if len(group) > 1:
            similar_groups.append(group)

    return results, similar_groups

# ===== 얼굴 태깅 =====
def tag_faces(target_dir, source_dir, result_dir, cos_th=0.6):
    target_embeddings = []
    for filename in os.listdir(target_dir):
        tpath = os.path.join(target_dir, filename)
        faces = get_face_embeddings(tpath)
        if faces:
            target_embeddings.append(
                (filename, torch.tensor(faces[0]["embedding"]).to(device))
            )

    if not target_embeddings:
        return []

    tagged_paths = []
    for filename in os.listdir(source_dir):
        spath = os.path.join(source_dir, filename)
        img = cv2.imread(spath)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = face_app.get(img_rgb)

        modified = False
        for face in faces:
            face_emb = torch.tensor(face["embedding"]).to(device)
            best_name, best_dist = None, 1.0
            for tname, temb in target_embeddings:
                dist = 1 - F.cosine_similarity(
                    temb.unsqueeze(0), face_emb.unsqueeze(0)
                ).item()
                if dist < best_dist:
                    best_dist, best_name = dist, tname

            if best_dist < cos_th:
                x1, y1, x2, y2 = map(int, face["bbox"])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                tag = f"{os.path.splitext(best_name)[0]} ({best_dist:.2f})"
                cv2.putText(
                    img,
                    tag,
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                modified = True

        if modified:
            out_name = f"tagged_{filename}"
            out_path = os.path.join(result_dir, out_name)
            cv2.imwrite(out_path, img)
            tagged_paths.append(out_name)

    return tagged_paths

# ===== Flask 앱 설정 =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
STATIC_FOLDER = os.path.join(BASE_DIR, "static")

app = Flask(__name__, static_folder=STATIC_FOLDER, template_folder=os.path.join(BASE_DIR, "templates"))
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["STATIC_FOLDER"] = STATIC_FOLDER

os.makedirs(os.path.join(UPLOAD_FOLDER, "sources"), exist_ok=True)
os.makedirs(os.path.join(UPLOAD_FOLDER, "targets"), exist_ok=True)
os.makedirs(os.path.join(STATIC_FOLDER, "results"), exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze_similarity", methods=["POST"])
def analyze_similarity_route():
    src_dir = os.path.join(app.config["UPLOAD_FOLDER"], "sources")
    clear_folder(src_dir)
    for f in request.files.getlist("images"):
        if f and f.filename:
            f.save(os.path.join(src_dir, secure_filename(f.filename)))

    results, groups = analyze_images(src_dir, blur_th=120)
    return render_template("results.html", results=results, groups=groups)

@app.route("/tag_faces", methods=["POST"])
def tag_faces_route():
    tgt_dir = os.path.join(app.config["UPLOAD_FOLDER"], "targets")
    src_dir = os.path.join(app.config["UPLOAD_FOLDER"], "sources")
    res_dir = os.path.join(app.config["STATIC_FOLDER"], "results")

    clear_folder(tgt_dir)
    clear_folder(src_dir)
    clear_folder(res_dir)

    for f in request.files.getlist("target_images"):
        if f and f.filename:
            f.save(os.path.join(tgt_dir, secure_filename(f.filename)))

    for f in request.files.getlist("source_images"):
        if f and f.filename:
            f.save(os.path.join(src_dir, secure_filename(f.filename)))

    tagged_images = tag_faces(tgt_dir, src_dir, res_dir)
    return render_template("tagging_results.html", tagged_images=tagged_images)

if __name__ == "__main__":
    app.run(debug=True)
