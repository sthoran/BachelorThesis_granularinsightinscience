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
from huggingface_hub import hf_hub_download

print("Python:", sys.executable)
print("Transformers:", transformers.__version__)

# Save metrics after each evaluation
class MetricsLoggerCallback(TrainerCallback):
    def __init__(self, output_path):
        self.output_path = output_path
        self.metrics = []

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        self.metrics.append(metrics)
        with open(self.output_path, "w") as f:
            json.dump(self.metrics, f, indent=2)

# Load YAML config from Hugging Face Hub
def load_config_from_hf(repo_id, filename):
    config_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# Compute seqeval metrics
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

# Tokenization + label alignment
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
def train_model(config_name, model_checkpoint, output_dir):
    cfg = load_config_from_hf("sthoran/ner-configs", config_name)

    label_map = {"O": 0, "B-METHOD": 1, "I-METHOD": 2}
    id2label = {v: k for k, v in label_map.items()}

    # Load datasets from Hugging Face
    dataset = DatasetDict({
        "train": load_dataset("sthoran/aidrugcorpus", data_files="mydata_train.json", split="train"),
        "validation": load_dataset("sthoran/aidrugcorpus", data_files="mydata_val.json", split="train"),
        "test": load_dataset("sthoran/aidrugcorpus", data_files="mydata_test.json", split="train")
    })

    # Load tokenizer and model from Hugging Face Hub
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint, num_labels=3, id2label=id2label, label2id=label_map
    )

    # Tokenize
    tokenized = dataset.map(lambda x: tokenize_and_align(x, tokenizer, label_map), batched=True)

    # Training args
    args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=float(cfg["learning_rate"]),
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        num_train_epochs=cfg["epochs"],
        weight_decay=0.01,
        save_total_limit=1,
        report_to=[]
    )

    # Trainer
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
            "config_name": "finetune_biobert.yaml",
            "model_checkpoint": "sthoran/aidrugcorpus_finetuned_biobert",
            "output_dir": "results/biobert"
        },
        {
            "name": "scibert",
            "config_name": "finetune_scibert.yaml",
            "model_checkpoint": "sthoran/aidrugcorpus_finetuned_scibert",
            "output_dir": "results/scibert"
        },
        {
            "name": "deberta",
            "config_name": "finetune_deberta.yaml",
            "model_checkpoint": "sthoran/aidrugcorpus_finetuned_deberta",
            "output_dir": "results/deberta"
        }
    ]

    for cfg in model_configs:
        print(f"\nTraining model: {cfg['name']}")
        Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
        train_model(cfg["config_name"], cfg["model_checkpoint"], cfg["output_dir"])
