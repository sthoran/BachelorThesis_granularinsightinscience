import pandas as pd
import json
from sklearn.model_selection import train_test_split
from pathlib import Path
from huggingface_hub import hf_hub_download

def group_by_sentence(df):
    grouped = df.groupby("sentence_id")
    examples = []
    for _, group in grouped:
        tokens = group["token"].tolist()
        labels = group["label"].tolist()
        examples.append({"tokens": tokens, "labels": labels})
    return examples

def save_jsonl(data, path):
    with open(path, "w") as f:
        for example in data:
            f.write(json.dumps(example) + "\n")
    print(f"Saved {len(data)} examples to {path}")

def main():
    # Download ner_bio_dataset.csv from Hugging Face
    csv_path = hf_hub_download(
        repo_id="sthoran/aidrugcorpus",
        filename="ner_bio_dataset.csv",
        repo_type="dataset"
    )

    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load CSV
    df = pd.read_csv(csv_path)

    # Group tokens by sentence_id into examples
    examples = group_by_sentence(df)

    # Split the data: 80% train, 10% val, 10% test
    train_data, temp_data = train_test_split(examples, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    # Save as JSONL
    save_jsonl(train_data, output_dir / "mydata_train.json")
    save_jsonl(val_data, output_dir / "mydata_val.json")
    save_jsonl(test_data, output_dir / "mydata_test.json")

if __name__ == "__main__":
    main()
