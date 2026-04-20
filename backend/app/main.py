from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from app.schemas import PredictRequest, PredictResponse
from app.utils import generate_symptom_summary
from app.model_service import load_models, unload_models, translate_tr_to_en, predict_top3

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield
    unload_models()

app = FastAPI(title="Triage Backend API", lifespan=lifespan)

@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(request: PredictRequest):
    text_tr = request.text_tr
    
    if not text_tr.strip():
        raise HTTPException(status_code=400, detail="Metin boş olamaz.")

    # 1. Kısa Semptom Özeti
    summary_tr = generate_symptom_summary(text_tr)

    # 2. Çeviri
    translated = translate_tr_to_en(text_tr)

    # 3. BERT ve RoBERTa Tahminleri
    bert_res = predict_top3(translated, "bert_model", "bert_tokenizer")
    roberta_res = predict_top3(translated, "roberta_model", "roberta_tokenizer")

    return PredictResponse(
        turkish_input=text_tr,
        short_symptom_summary=summary_tr,
        english_translation=translated,
        bert_top3=bert_res,
        roberta_top3=roberta_res
    )
