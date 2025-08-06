import os
import uuid
import shutil
import requests
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ───── 설정 ─────

UPLOAD_DIR = "static/uploaded"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# imgbb 업로드 함수
def upload_to_imgbb(image_path):
    with open(image_path, "rb") as file:
        response = requests.post(
            "https://api.imgbb.com/1/upload",
            params={"key": IMGBB_API_KEY},
            files={"image": file}
        )
    data = response.json()
    return data["data"]["url"] if "data" in data else None


@app.post("/recommend", response_class=HTMLResponse)
async def recommend(request: Request, files: list[UploadFile] = File(...)):
    uploaded_images = []
    image_urls = []

    for file in files:
        # 파일 저장
        filename = f"{uuid.uuid4().hex}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIR, filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        uploaded_images.append(filename)

        # imgbb 업로드
        image_url = upload_to_imgbb(file_path)
        if image_url:
            image_urls.append(image_url)

    if not image_urls:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "uploaded_images": uploaded_images,
            "result_text": "[ERROR] 이미지 업로드 실패 😢"
        })

    # GPT 프롬프트
    messages = [
    {
        "role": "system",
        "content": (
            "너는 감성적인 앨범 편집자야. 지금부터 줄 사진들을 보고, "
            "사진 전체의 분위기와 스토리를 파악해서 앨범 제목과 설명을 추천해줘. "
            "이 앨범은 친구나 가족, 연인과의 추억을 담기 위한 것이니까, "
            "무조건 딱딱한 표현은 피하고, 진심 어린 말투, 말랑한 뉘앙스, 웃긴 드립도 살짝 섞어줘.\n\n"
            "제목은 아래 3가지 스타일로 각각 하나씩 추천해줘:\n"
            "- 드립 스타일: 짧고 센스 있게 웃긴 느낌. 유행어나 말장난 환영!\n"
            "- 귀여운 스타일: 말랑하고 아기자기한 느낌. 이모티콘 느낌 나는 말도 좋아.\n"
            "- 감성 스타일: 진짜 회고록처럼 마음이 따뜻해지는 문장. 문학적이거나 일기처럼 써줘.\n\n"
            "그리고 설명은 앨범 주인의 시선에서, 친구에게 편지 쓰듯 써줘. 회상 느낌 좋아. "
            "하루를 마무리하며 남기는 글 같아야 해. 문장은 여러 줄로 자연스럽게 써줘.\n\n"
            "형식은 아래처럼 맞춰줘:\n\n"
            "📸 앨범 제목 추천\n"
            "드립 스타일:\n《제목1》\n\n"
            "귀여운 스타일:\n《제목2》\n\n"
            "감성 스타일:\n《제목3》\n\n"
            "✍️ 설명\n설명 내용 여러 줄로 자연스럽게 써줘. 친구끼리 말하듯 따뜻하게."
        )
    },
    {
        "role": "user",
        "content": [
            { "type": "text", "text": "이 사진들로 앨범을 만들고 싶어. 제목 3개랑 설명 하나 추천해줘!" },
            *[ { "type": "image_url", "image_url": { "url": url } } for url in image_urls ]
        ]
    }
]

    payload = {
        "model": "gpt-4o-mini",
        "temperature": 1.2,
        "messages": messages
    }

    headers = {
        "Authorization": f"Bearer {GMS_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(GMS_API_URL, headers=headers, json=payload)
        result = response.json()
        print("📦 GMS 응답:", result)

        if "choices" in result:
            content = result["choices"][0]["message"]["content"]
        else:
            content = f"[ERROR] {result.get('error', result)}"
    except Exception as e:
        content = f"[ERROR] {str(e)}"

    return templates.TemplateResponse("index.html", {
        "request": request,
        "uploaded_images": uploaded_images,
        "result_text": content
    })


@app.get("/", response_class=HTMLResponse)
async def form_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
