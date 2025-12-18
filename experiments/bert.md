# BERT Sentiment Model (Final)

## Model Overview
This model uses **DistilBERT**, a transformer-based language model,
fine-tuned for binary sentiment classification on the Amazon Reviews
Polarity dataset.

It serves as the **final production model**, replacing classical
TF-IDF approaches.

---

## Dataset
- Amazon Reviews Polarity
- Task: Binary sentiment classification
- Training samples: 50,000
- Validation samples: 10,000

---

## Model Architecture
- Base model: distilbert-base-uncased
- Classification head: linear layer on [CLS] token
- Loss: Cross-entropy

---

## Training Configuration
- Epochs: 2
- Batch size: 16
- Learning rate: 2e-5
- Optimizer: AdamW
- Hardware: NVIDIA RTX 4050 (GPU)

---

## Evaluation Results

**Validation Metrics**
- Accuracy: **93.6%**
- F1-score: **93.64%**
- Precision: **93.81%**
- Recall: **93.47%**

**Evaluation Loss:** 0.223

---

## Key Improvements over TF-IDF
- Correct handling of negation (e.g., "not happy", "not bad at all")
- Context-aware word representations
- Improved robustness on long and complex reviews

---

## Limitations
- Higher inference cost than TF-IDF
- Requires GPU for efficient training
- Still limited on sarcasm and implicit sentiment

---

## Conclusion
Transformer-based models significantly outperform classical NLP
approaches for sentiment analysis, especially in understanding
context and negation.
