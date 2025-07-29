from flask import Flask, request, render_template, redirect, url_for
from deepface import DeepFace
from scipy.spatial import distance
import numpy as np
import os
import cv2

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 등록된 유저 프로필 (유저 ID: 얼굴 임베딩 벡터)
user_profiles = {} 
# 얼굴 일치 여부를 판단하는 거리 임계값 (낮을수록 엄격)
DIST_THRESHOLD = 0.5 

# ✅ [개선 1] 메모리 내 이미지 배열로 임베딩 계산 (파일 I/O 제거)
def get_embedding_from_image_array(image_array):
    """Numpy 배열 형태의 이미지에서 직접 얼굴 임베딩을 추출합니다."""
    # DeepFace.represent는 파일 경로뿐만 아니라 numpy 배열도 직접 처리 가능
    embedding_obj = DeepFace.represent(
        img_path=image_array,
        model_name="ArcFace",
        enforce_detection=False # 이미 얼굴만 잘린 이미지를 받으므로 탐지 비활성화
    )
    return embedding_obj[0]["embedding"]

# ✅ [개선 2] 파일 저장 없이 메모리에서 바로 등록 처리
@app.route("/register", methods=["POST"])
def register_user():
    user_id = request.form["user_id"]
    files = request.files.getlist("profile_images")
    
    if not user_id or not files:
        return "User ID와 프로필 이미지를 모두 제출해야 합니다.", 400

    embeddings = []
    for f in files:
        # 파일을 읽어 Numpy 배열로 직접 디코딩
        filestr = f.read()
        np_img = np.frombuffer(filestr, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # 얼굴 탐지 후 첫 번째 얼굴만 사용 (프로필 사진이므로)
        try:
            face_obj = DeepFace.extract_faces(
                img_path=img, 
                detector_backend="retinaface"
            )
            # DeepFace가 반환하는 얼굴 이미지는 0-1 범위로 정규화되어 있으므로 255를 곱해줌
            face_array = (face_obj[0]['face'] * 255).astype(np.uint8)
            emb = get_embedding_from_image_array(face_array)
            embeddings.append(emb)
        except ValueError:
            # 이미지에서 얼굴을 찾지 못한 경우 건너뛰기
            print(f"경고: {f.filename} 파일에서 얼굴을 감지하지 못했습니다.")
            continue
    
    if not embeddings:
         return "제출된 어떤 이미지에서도 얼굴을 감지할 수 없습니다. 다른 사진을 사용해주세요.", 400

    # 여러 프로필 사진의 임베딩을 평균내어 대표 임베딩으로 사용
    user_profiles[user_id] = np.mean(embeddings, axis=0)
    print(f"✅ {user_id} 님이 성공적으로 등록되었습니다.")
    return redirect(url_for("index"))

# ✅ [개선 3] 효율적인 얼굴 처리 및 중복 태그 방지
@app.route("/upload", methods=["POST"])
def upload_photos():
    files = request.files.getlist("photos")
    results = {}

    for f in files:
        path = os.path.join(UPLOAD_FOLDER, f.filename)
        f.save(path) # 원본 사진은 결과 표시를 위해 저장

        # 이미지를 한번만 읽어서 처리
        img = cv2.imread(path)
        
        try:
            # extract_faces는 탐지된 얼굴 이미지(numpy 배열)까지 반환해 줌
            faces = DeepFace.extract_faces(img_path=img, detector_backend="retinaface")
        except ValueError:
            # 사진에서 얼굴을 전혀 찾지 못한 경우
            results[f.filename] = ["얼굴을 찾을 수 없음"]
            continue

        # ✨ 한 사진에서 동일 인물이 여러 번 태그되는 것을 막기 위해 set 사용
        found_users = set()

        for face in faces:
            # extract_faces가 이미 잘라준 얼굴 이미지를 사용 (재크롭 불필요)
            # DeepFace는 0-1로 정규화된 값을 반환하므로 다시 0-255 범위로 변환
            face_array = (face['face'] * 255).astype(np.uint8)
            
            # 개선된 함수를 사용해 임베딩 계산
            emb = get_embedding_from_image_array(face_array)
            
            min_dist = float('inf')
            matched_user = None

            # 등록된 모든 유저와 거리를 비교해 가장 가까운 유저를 찾음
            for user_id, user_emb in user_profiles.items():
                dist = distance.cosine(emb, user_emb)
                if dist < min_dist:
                    min_dist = dist
                    matched_user = user_id
            
            # 가장 가까운 유저가 임계값 이내일 경우 태그 추가
            if min_dist < DIST_THRESHOLD:
                found_users.add(matched_user)

        results[f.filename] = list(found_users) if found_users else ["알 수 없음"]

    return render_template("result.html", results=results)

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)