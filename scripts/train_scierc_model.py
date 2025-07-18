import json
import yaml
import numpy as np
import mlflow
from datasets import load_dataset, DatasetDict
from transformers import (AutoTokenizer, AutoModelForTokenClassification, TrainingArguments,
                          Trainer, DataCollatorForTokenClassification)
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from huggingface_hub import hf_hub_download

class NERTrainer:
    def __init__(self, config_filename, config_repo="sthoran/ner-configs"):
        self.config_path = self.download_config(config_filename, config_repo)
        self.config = self.load_config(self.config_path)

        self.tokenizer = AutoTokenizer.from_pretrained(
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

        print("Raw dataset sizes:", {k: len(v) for k, v in self.dataset.items()})

        self.tokenized_datasets = self.dataset.map(self.tokenize_and_align_labels, batched=True)
        self.training_args = self.set_training_args()

    def download_config(self, filename, repo_id):
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset"
        )
        
    def load_config(self, path):
        with open(path, "r") as file:
            return yaml.safe_load(file)

    def load_datasets(self):
        return DatasetDict({
            "train": load_dataset(
                "sthoran/method_only_scierc", data_files="ner_train_data.json", split="train"
            ),
            "validation": load_dataset(
                "sthoran/method_only_scierc", data_files="ner_dev_data.json", split="train"
            ),
            "test": load_dataset(
                "sthoran/method_only_scierc", data_files="ner_test_data.json", split="train"
            )
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
            learning_rate=float(self.config.get("learning_rate")),
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
        # Use a local mlruns folder
        tracking_uri = self.config.get("tracking_uri", "./mlruns")
        mlflow.set_tracking_uri(f"file:{tracking_uri}")
        mlflow.set_experiment("ner_biobert_experiment")

        with mlflow.start_run(run_name=f"{self.config['model_checkpoint'].split('/')[-1]}_NER") as run:
            print("MLflow run started:", run.info.run_id)

            # Log parameters
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

            print("Starting training...")
            trainer.train()
            print("Training complete.")

            # Evaluate
            eval_results = trainer.evaluate()
            print("Eval results:", eval_results)

            # Safely log metrics
            safe_metrics = {
                k: float(v) for k, v in eval_results.items()
                if isinstance(v, (int, float, np.floating))
            }

            mlflow.log_metrics(safe_metrics)
            if "epoch" in eval_results:
                mlflow.log_metric("epoch", float(eval_results["epoch"]))

            # Save and log model + tokenizer
            model_path = f"{self.config['output_dir']}/model"
            tokenizer_path = f"{self.config['output_dir']}/tokenizer"

            self.model.save_pretrained(model_path)
            self.tokenizer.save_pretrained(tokenizer_path)

            mlflow.log_artifacts(model_path, artifact_path="model")
            mlflow.log_artifacts(tokenizer_path, artifact_path="tokenizer")

            print("Training complete. Model logged to MLflow.")


if __name__ == "__main__":
    trainer = NERTrainer("biobert.yaml")
    trainer.train_with_mlflow()
