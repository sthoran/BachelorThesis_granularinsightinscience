import json
import yaml
import numpy as np
import mlflow
from datasets import load_dataset, DatasetDict
from transformers import (AutoTokenizer, AutoModelForTokenClassification, TrainingArguments,
                          Trainer, DataCollatorForTokenClassification)
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import DebertaTokenizerFast

class NERTrainer:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.tokenizer = DebertaTokenizerFast.from_pretrained(
                            self.config["model_checkpoint"],
                            add_prefix_space=True
                            )
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.config["model_checkpoint"],
            num_labels=3,
            id2label={0: "O", 1: "B-Method", 2: "I-Method"},
            label2id={"O": 0, "B-Method": 1, "I-Method": 2}
        )
        self.data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
        self.dataset = self.load_datasets()
        self.tokenized_datasets = self.dataset.map(self.tokenize_and_align_labels, batched=True)
        self.training_args = self.set_training_args()
        
    def load_config(self, path):
        with open(path, "r") as file:
            return yaml.safe_load(file)

    def load_ner_json(self, path):
        with open(path, "r") as f:
            data = json.load(f)
        return {"tokens": [d["tokens"] for d in data], "labels": [d["labels"] for d in data]}

    def load_datasets(self):
        return DatasetDict({
            "train": load_dataset("json", data_files="../data/ner_train_data.json")["train"],
            "validation": load_dataset("json", data_files="../data/ner_dev_data.json")["train"],
            "test": load_dataset("json", data_files="../data/ner_test_data.json")["train"]
        })

    def tokenize_and_align_labels(self, batch):
        tokenized_inputs = self.tokenizer(
            batch["tokens"], truncation=True, is_split_into_words=True, padding=True
        )

        label_map = {"O": 0, "B-Method": 1, "I-Method": 2}
        all_labels = []

        for i, word_ids in enumerate(tokenized_inputs.word_ids(batch_index=i) for i in range(len(batch["tokens"]))):
            labels = []
            previous_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    labels.append(-100)
                elif word_idx != previous_word_idx:
                    labels.append(label_map[batch["labels"][i][word_idx]])
                else:
                    labels.append(label_map.get(batch["labels"][i][word_idx], 2))
                previous_word_idx = word_idx
            all_labels.append(labels)

        tokenized_inputs["labels"] = all_labels
        return tokenized_inputs

    def set_training_args(self):
        return TrainingArguments(
            output_dir=self.config["output_dir"],
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate= float(self.config.get("learning_rate")),
            per_device_train_batch_size=self.config["batch_size"],
            per_device_eval_batch_size=self.config["batch_size"],
            num_train_epochs=self.config["epochs"],
            weight_decay=0.01,
            save_total_limit=1,
            report_to=[]
        )
              
              
    def compute_metrics(self, p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [self.model.config.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.model.config.id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
            "accuracy": accuracy_score(true_labels, true_predictions),
        }


    def train_with_mlflow(self):
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("ner_deberta_experiment")

        with mlflow.start_run(run_name=f"{self.config['model_checkpoint'].split('/')[-1]}_NER"):
            mlflow.log_params({
                "model_checkpoint": self.config["model_checkpoint"],
                "epochs": self.config["epochs"],
                "batch_size": self.config["batch_size"],
                "learning_rate": float(self.config.get('learning_rate'))
            })

            trainer = Trainer(
                model=self.model,
                args=self.training_args,
                train_dataset=self.tokenized_datasets["train"],
                eval_dataset=self.tokenized_datasets["validation"],
                tokenizer=self.tokenizer,
                data_collator=self.data_collator,
                compute_metrics=self.compute_metrics
            )

            trainer.train()
            eval_results = trainer.evaluate()
            mlflow.log_metrics(eval_results)

            model_path = f"{self.config['output_dir']}/model"
            tokenizer_path = f"{self.config['output_dir']}/tokenizer"

            self.model.save_pretrained(model_path)
            self.tokenizer.save_pretrained(tokenizer_path)

            mlflow.log_artifacts(model_path, artifact_path="model")
            mlflow.log_artifacts(tokenizer_path, artifact_path="tokenizer")

            print("Training complete. Model logged to MLflow.")


if __name__ == "__main__":
    trainer = NERTrainer("../configs/deberta.yaml")
    trainer.train_with_mlflow()