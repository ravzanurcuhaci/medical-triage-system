from pydantic import BaseModel

class PredictRequest(BaseModel):
    text_tr: str

class Top3Prediction(BaseModel):
    label: str
    score: float

class PredictResponse(BaseModel):
    turkish_input: str
    short_symptom_summary: str
    english_translation: str
    bert_top3: list[Top3Prediction]
    roberta_top3: list[Top3Prediction]
