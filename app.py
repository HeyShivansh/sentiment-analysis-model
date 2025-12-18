from fastapi import FastAPI
import joblib

from preprocessing.text_cleaning import clean_text

app = FastAPI(title="Sentiment Analysis API")

# Load model and vectorizer
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


@app.get("/")
def home():
    return {"message": "Sentiment Analysis API is running"}


@app.post("/predict")
def predict_sentiment(text: str):
    cleaned_text = clean_text(text)
    text_vec = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_vec)[0]
    probability = model.predict_proba(text_vec).max()

    sentiment = "Positive" if prediction == 1 else "Negative"

    return {
        "sentiment": sentiment,
        "confidence": round(float(probability), 3)
    }
