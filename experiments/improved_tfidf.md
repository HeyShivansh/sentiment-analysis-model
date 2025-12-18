# Improved TF-IDF Sentiment Model

## Objective
Improve the baseline TF-IDF sentiment classifier by addressing:
- Negation handling issues
- Positive word dominance
- Class imbalance effects

---

## Changes from Baseline

### Feature Extraction
- Increased `max_features` from **5000 → 20000**
- Expanded `ngram_range` from **(1,2) → (1,3)**
- Added `min_df = 2` to reduce noise

These changes allow the model to capture important sentiment phrases like:
- "not happy"
- "not worth buying"
- "not good at all"

---

### Model Improvements
- Logistic Regression with `class_weight = "balanced"`
- Reduced bias toward positive sentiment
- Improved recall for negative reviews

---

## Evaluation Results

**Validation Accuracy:**  
**89.25%**

### Classification Report
| Class | Precision | Recall | F1-score | Support |
|------|----------|--------|---------|---------|
| Negative (0) | 0.88 | 0.89 | 0.89 | 9602 |
| Positive (1) | 0.90 | 0.89 | 0.90 | 10398 |
| **Overall Accuracy** |  |  | **0.89** | 20000 |

---

## Behavioral Improvements

The improved model correctly classifies negation-heavy sentences:

> "I am not happy with the product"  
→ **Negative (confidence ≈ 0.93)**

This was previously misclassified by the baseline model.

---

## Key Takeaways
- Feature engineering significantly improves classical NLP models
- N-grams and class balancing reduce common sentiment errors
- TF-IDF models remain strong, fast, and interpretable baselines

---

## Limitations
- Still lacks deep contextual understanding
- Struggles with sarcasm and complex emotions
- Further improvements will have diminishing returns

---

## Next Steps
- Introduce a Neutral sentiment class
- Compare results with Transformer-based models (BERT)
