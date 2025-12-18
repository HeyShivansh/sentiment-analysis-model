# Baseline Sentiment Model — TF-IDF + Logistic Regression

## Model Overview
This is the first baseline model built for sentiment analysis on the  
**Amazon Reviews Polarity Dataset**.

The goal of this model was to establish a strong classical NLP baseline
before moving to advanced models like Transformers.

---

## Dataset
- Source: Amazon Reviews Polarity
- Task: Binary sentiment classification
- Labels:
  - 0 → Negative
  - 1 → Positive
- Training samples used: ~100,000
- Validation samples: ~20,000

---

## Preprocessing
- Lowercasing text
- Removing URLs
- Removing punctuation
- Normalizing whitespace

---

## Feature Extraction
- Method: TF-IDF
- Parameters:
  - `max_features = 5000`
  - `ngram_range = (1, 2)` (unigrams + bigrams)

---

## Model
- Algorithm: Logistic Regression
- `max_iter = 1000`
- No class weighting applied

---

## Evaluation Results

**Validation Accuracy:**  
**87.95%**

### Classification Report
| Class | Precision | Recall | F1-score | Support |
|------|----------|--------|---------|---------|
| Negative (0) | 0.87 | 0.88 | 0.87 | 9602 |
| Positive (1) | 0.89 | 0.88 | 0.88 | 10398 |
| **Overall Accuracy** |  |  | **0.88** | 20000 |

---

## Observed Issues
- The model struggles with **negation handling**
- Example:
  > "I am not happy with the product"  
  was incorrectly predicted as **Positive** with low confidence (~0.53)

This happens because:
- Positive words like *"happy"* dominate
- Important bigrams like *"not happy"* may be dropped due to feature limits
- TF-IDF lacks true contextual understanding

---

## Key Takeaways
- TF-IDF + Logistic Regression provides a strong and fast baseline
- Performance is good overall but fails on subtle linguistic patterns
- This model establishes a reference point for future improvements

---

## Next Steps
- Increase feature space (`max_features`)
- Use stronger n-gram ranges
- Apply class balancing
- Introduce neutral sentiment handling
- Compare with Transformer-based models (BERT)
