from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)

# -------------------------
# App initialization
# -------------------------
app = FastAPI(title="Sentiment Analysis API (BERT + SST)")

# -------------------------
# Device setup
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Load model & tokenizer
# -------------------------
MODEL_PATH = "models/bert_sst/final"

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

# -------------------------
# Label mapping (SST → 3 classes)
# -------------------------
LABEL_MAP = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

# -------------------------
# Request schema (JSON body)
# -------------------------
class TextRequest(BaseModel):
    text: str

# -------------------------
# Routes
# -------------------------
@app.get("/")
def home():
    return {
        "message": "Sentiment Analysis API (BERT + SST) is running"
    }


@app.post("/predict")
def predict_sentiment(request: TextRequest):
    # Tokenize input
    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    # Move tensors to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    # Prediction
    confidence, prediction = torch.max(probs, dim=1)
    sentiment = LABEL_MAP[prediction.item()]

    return {
    "sentiment": sentiment,
    "confidence": round(confidence.item(), 3),
    "probabilities": {
        "negative": round(probs[0][0].item(), 3),
        "neutral": round(probs[0][1].item(), 3),
        "positive": round(probs[0][2].item(), 3),
    }}
