from pydantic import BaseModel

class PredictRequest(BaseModel):
    text_tr: str

class Top3Prediction(BaseModel):
    label: str
    label_tr: str
    score: float
    department: str
    urgency: str

class PredictResponse(BaseModel):
    turkish_input: str
    short_symptom_summary: str
    english_translation: str
    bert_top3: list[Top3Prediction]
    roberta_top3: list[Top3Prediction]
    final_predictions: list[Top3Prediction]
