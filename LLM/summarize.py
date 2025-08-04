from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Using device: {device}")

# ✅ 더 성능 좋은 요약 모델로 변경
model_name = "psyche/KoT5-summarization" 
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_safetensors=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def generate_multiple_summaries(text, num_return_sequences=3):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True).to(device)
    outputs = model.generate(
        **inputs,
        # ✅ num_beams 값을 줄여 속도 향상 (기존 5 -> 3)
        num_beams=3, 
        num_return_sequences=num_return_sequences,
        max_length=64,
        no_repeat_ngram_size=2,
        early_stopping=True,
    )
    return [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]