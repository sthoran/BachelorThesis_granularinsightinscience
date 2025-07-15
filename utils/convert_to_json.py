import pandas as pd
import json
from sklearn.model_selection import train_test_split
from pathlib import Path

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
    input_csv = "ner_bio_dataset.csv"
    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    # Group tokens by sentence_id into examples
    examples = group_by_sentence(df)

    # Split the data: 80% train, 10% val, 10% test
    train_data, temp_data = train_test_split(examples, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    # Save with custom names
    save_jsonl(train_data, output_dir / "mydata_train.json")
    save_jsonl(val_data, output_dir / "mydata_val.json")
    save_jsonl(test_data, output_dir / "mydata_test.json")

if __name__ == "__main__":
    main()
