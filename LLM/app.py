from flask import Flask, render_template, request
import os
from caption import generate_caption
from summarize import generate_multiple_summaries
from transformers import pipeline # ✅ 번역 파이프라인 추가

# ✅ 영어->한국어 번역 파이프라인 설정
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-en-ko")

# ✅ 업로드 경로 설정 (OS 호환 슬래시)
UPLOAD_FOLDER = os.path.join("static", "uploads")

# ✅ 업로드 경로 없으면 자동 생성
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Flask 앱 설정
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ✅ 메인 페이지
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# ✅ 이미지 업로드 및 결과 처리
@app.route("/upload", methods=["POST"])
def upload():
    image_paths = []

    # 1. 이미지 3장 저장
    for i in range(1, 4):
        file = request.files.get(f"image{i}")
        if not file:
            continue
        filename = file.filename
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)
        image_paths.append(save_path)

    # 2. 이미지 설명 생성 (영어로 생성됨)
    english_captions = [generate_caption(p) for p in image_paths]
    
    # ✅ 2-1. 생성된 영어 설명을 한국어로 번역
    korean_captions = [translator(c)[0]['translation_text'] for c in english_captions]
    combined_text = " ".join(korean_captions)

    # 3. 제목 및 설명 3개씩 추천
    titles = generate_multiple_summaries("제목: " + combined_text)
    descriptions = generate_multiple_summaries("설명: " + combined_text)

    # 결과 렌더링
    return render_template("index.html", titles=titles, descriptions=descriptions, descriptions_ko=korean_captions)

# ✅ 서버 실행
if __name__ == "__main__":
    # app.run(debug=True) # debug 모드는 느릴 수 있으므로, 실제 구동 시에는 아래 코드로 실행
    app.run(host='0.0.0.0', port=5001)