import torch
from datasets import load_dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# -----------------------
# Label mapping: SST-5 → 3 classes
# -----------------------
def map_label(label):
    if label in [0, 1]:
        return 0      # Negative
    elif label == 2:
        return 1      # Neutral
    else:
        return 2      # Positive


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def main():
    print("Loading SST dataset...")
    dataset = load_dataset("SetFit/sst5")

    # Apply label mapping
    dataset = dataset.map(
        lambda x: {"label": map_label(x["label"])}
    )

    tokenizer = DistilBertTokenizerFast.from_pretrained(
        "distilbert-base-uncased"
    )

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding=True,
            truncation=True,
            max_length=128
        )

    dataset = dataset.map(tokenize, batched=True)

    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"]
    )

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=3
    )

    training_args = TrainingArguments(
        output_dir="models/bert_sst",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    print("Training BERT on SST...")
    trainer.train()

    trainer.save_model("models/bert_sst/final")
    tokenizer.save_pretrained("models/bert_sst/final")

    print("SST BERT training complete!")


if __name__ == "__main__":
    main()
