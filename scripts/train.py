from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
import numpy as np
import json

# Load your JSON files
def load_ner_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return {"tokens": [d["tokens"] for d in data], "labels": [d["labels"] for d in data]}

# Paths to your files
train_data = load_ner_json("data/ner_train_data.json")
val_data = load_ner_json("data/ner_dev_data.json")
test_data = load_ner_json("data/ner_test_data.json")

# Convert to Hugging Face datasets
dataset = DatasetDict({
    "train": load_dataset("json", data_files="data/train.json")["train"],
    "validation": load_dataset("json", data_files="data/dev.json")["train"],
    "test": load_dataset("json", data_files="data/test.json")["train"]
})

# Load tokenizer and model
model_checkpoint = "allenai/scibert_scivocab_uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=3, id2label={0: "O", 1: "B-Method", 2: "I-Method"}, label2id={"O": 0, "B-Method": 1, "I-Method": 2})

# Tokenize and align labels
def tokenize_and_align_labels(batch):
    tokenized_inputs = tokenizer(
        batch["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding=True  
    )

    all_labels = []
    label_map = {"O": 0, "B-Method": 1, "I-Method": 2}

    for i, word_ids in enumerate(tokenized_inputs.word_ids(batch_index=i) for i in range(len(batch["tokens"]))):
        labels = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                labels.append(-100)
            elif word_idx != previous_word_idx:
                labels.append(label_map[batch["labels"][i][word_idx]])
            else:
                # continuation of the same word
                labels.append(label_map.get(batch["labels"][i][word_idx], 2))  # fallback to I-Method
            previous_word_idx = word_idx
        all_labels.append(labels)

    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# Training setup
args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    save_total_limit=1
)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# Compute metrics
import evaluate
seqeval = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [model.config.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [model.config.id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# Train
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
