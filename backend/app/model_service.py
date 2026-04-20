import os
import torch
import torch.nn.functional as F
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM
)

BASE_PATH = os.getenv("MODELS_DIR", "models")
BERT_MODEL_PATH = f"{BASE_PATH}/bert_symptom_model"
ROBERTA_MODEL_PATH = f"{BASE_PATH}/roberta_model"
LABEL_MAP_PATH = f"{BASE_PATH}/label_map.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = {}

def load_models():
    print("Modeller yükleniyor, lütfen bekleyin...")
    
    # Label Map
    label_map_df = pd.read_csv(LABEL_MAP_PATH)
    models["id2label"] = dict(zip(label_map_df["label_id"], label_map_df["label"]))

    # BERT
    models["bert_tokenizer"] = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)
    bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
    bert_model.to(device)
    bert_model.eval()
    models["bert_model"] = bert_model

    # RoBERTa
    models["roberta_tokenizer"] = AutoTokenizer.from_pretrained(ROBERTA_MODEL_PATH)
    roberta_model = AutoModelForSequenceClassification.from_pretrained(ROBERTA_MODEL_PATH)
    roberta_model.to(device)
    roberta_model.eval()
    models["roberta_model"] = roberta_model

    # Çeviri
    tr_en_model_name = "Helsinki-NLP/opus-mt-tc-big-tr-en"
    models["tr_en_tokenizer"] = AutoTokenizer.from_pretrained(tr_en_model_name, clean_up_tokenization_spaces=True)
    models["tr_en_model"] = AutoModelForSeq2SeqLM.from_pretrained(tr_en_model_name)
    
    print("Tüm modeller başarıyla yüklendi!")

def unload_models():
    models.clear()

def translate_tr_to_en(text: str) -> str:
    from app.utils import clean_text
    text = clean_text(text)
    tokenizer = models["tr_en_tokenizer"]
    model = models["tr_en_model"]

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    outputs = model.generate(
        **inputs,
        max_length=256,
        num_beams=4,
        early_stopping=True
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def predict_top3(text: str, model_key: str, tokenizer_key: str) -> list[dict]:
    model = models[model_key]
    tokenizer = models[tokenizer_key]
    id2label = models["id2label"]

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)

    top_probs, top_indices = torch.topk(probs, k=3, dim=1)

    results = []
    for prob, idx in zip(top_probs[0], top_indices[0]):
        label = id2label[idx.item()]
        score = round(prob.item() * 100, 2)
        results.append({"label": label, "score": score})

    return results
