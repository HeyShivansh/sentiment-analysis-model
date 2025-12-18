import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

from preprocessing.text_cleaning import clean_text


def main():
    print("Loading dataset...")
    dataset = load_dataset("amazon_polarity")

    # Use a subset to keep training fast
    texts = dataset["train"]["content"][:100000]
    labels = dataset["train"]["label"][:100000]

    print("Cleaning text...")
    texts = [clean_text(text) for text in texts]

    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    print("Training model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    print("Evaluating model...")
    preds = model.predict(X_val_vec)
    acc = accuracy_score(y_val, preds)

    print(f"Validation Accuracy: {acc:.4f}")
    print(classification_report(y_val, preds))

    print("Saving model and vectorizer...")
    joblib.dump(model, "models/sentiment_model.pkl")
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

    print("Training complete!")


if __name__ == "__main__":
    main()
