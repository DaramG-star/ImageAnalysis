from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    use_safetensors=True 
)

def generate_caption(image_path):
    image = Image.open(image_path).convert('RGB')
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# if __name__ == "__main__":
#     test_image = "sample.jpg"  # 여기에 존재하는 이미지 경로 지정
#     caption = generate_caption(test_image)
#     print(f"📸 이미지 설명: {caption}")