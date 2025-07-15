import yaml
import json
import numpy as np
from pathlib import Path
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification, TrainerCallback
)
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
import sys
import transformers
print("Python:", sys.executable)
print("Transformers:", transformers.__version__)

BASE_DIR = Path(__file__).resolve().parent

# save metrics after each evaluation
class MetricsLoggerCallback(TrainerCallback):
    def __init__(self, output_path):
        self.output_path = output_path
        self.metrics = []

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        self.metrics.append(metrics)
        with open(self.output_path, "w") as f:
            json.dump(self.metrics, f, indent=2)

# Load YAML config
def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# Compute metrics using seqeval
def compute_metrics(p, id2label):
    preds = np.argmax(p.predictions, axis=2)
    true_preds, true_labels = [], []
    for pred, lab in zip(preds, p.label_ids):
        true_preds.append([id2label[p] for (p, l) in zip(pred, lab) if l != -100])
        true_labels.append([id2label[l] for (p, l) in zip(pred, lab) if l != -100])
    return {
        "precision": precision_score(true_labels, true_preds),
        "recall": recall_score(true_labels, true_preds),
        "f1": f1_score(true_labels, true_preds),
        "accuracy": accuracy_score(true_labels, true_preds),
    }

# Tokenize and align labels
def tokenize_and_align(batch, tokenizer, label_map):
    tokenized = tokenizer(
    batch["tokens"],
    is_split_into_words=True,
    truncation=True,
    max_length=512,           
    padding="max_length"      
)
    all_labels = []
    for i, word_ids in enumerate(tokenized.word_ids(batch_index=i) for i in range(len(batch["tokens"]))):
        labels = []
        prev = None
        for idx in word_ids:
            if idx is None:
                labels.append(-100)
            elif idx != prev:
                labels.append(label_map[batch["labels"][i][idx]])
            else:
                labels.append(label_map.get(batch["labels"][i][idx], 2))
            prev = idx
        all_labels.append(labels)
    tokenized["labels"] = all_labels
    return tokenized

# Main training logic
def train_model(config_path, pretrained_model_path, output_dir):
    cfg = load_config(config_path)
    label_map = {"O": 0, "B-METHOD": 1, "I-METHOD": 2}
    id2label = {v: k for k, v in label_map.items()}

    # Load dataset
    data_dir = BASE_DIR / "data"
    train_file = data_dir / "mydata_train.json"
    val_file = data_dir / "mydata_val.json"
    test_file = data_dir / "mydata_test.json"

    data = DatasetDict({
        "train": load_dataset("json", data_files=str(train_file), split="train"),
        "validation": load_dataset("json", data_files=str(val_file), split="train"),
        "test": load_dataset("json", data_files=str(test_file), split="train"),
    })


    # Load tokenizer and model from local path
    tokenizer = AutoTokenizer.from_pretrained(f"{pretrained_model_path}/tokenizer", add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(
        f"{pretrained_model_path}/model", num_labels=3, id2label=id2label, label2id=label_map
    )

    tokenized = data.map(lambda x: tokenize_and_align(x, tokenizer, label_map), batched=True)

    # Training args
    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=float(cfg["learning_rate"]),
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        num_train_epochs=cfg["epochs"],
        weight_decay=0.01,
        save_total_limit=1,
        report_to=[]
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=lambda p: compute_metrics(p, id2label),
        callbacks=[MetricsLoggerCallback(f"{output_dir}/metrics.json")]
    )

    trainer.train()
    model.save_pretrained(f"{output_dir}/model")
    tokenizer.save_pretrained(f"{output_dir}/tokenizer")
    print(f"Training complete. Metrics saved to {output_dir}/metrics.json.")

# Run all models
if __name__ == "__main__":
    model_configs = [
        {
            "name": "biobert",
            "config": BASE_DIR / "configs" / "finetune_biobert.yaml",
            "pretrained_path": BASE_DIR / "models" / "biobert",
            "output_dir": BASE_DIR / "results" / "biobert"
        },
        {
            "name": "scibert",
            "config": BASE_DIR / "configs" / "finetune_scibert.yaml",
            "pretrained_path": BASE_DIR / "models" / "scibert",
            "output_dir": BASE_DIR / "results" / "scibert"
        },
        {
            "name": "deberta",
            "config": BASE_DIR / "configs" / "finetune_deberta.yaml",
            "pretrained_path": BASE_DIR / "models" / "deberta",
            "output_dir": BASE_DIR / "results" / "deberta"
        }
    ]

    for cfg in model_configs:
        print(f"\n Training model: {cfg['name']}")
        Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
        train_model(cfg["config"], cfg["pretrained_path"], cfg["output_dir"])
