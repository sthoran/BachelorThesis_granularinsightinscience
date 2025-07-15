import json
from pathlib import Path

def fix_json_file(file_path):
    fixed_lines = []
    file_path = Path(file_path)
    with file_path.open("r") as f:
        for line in f:
            try:
                entry = json.loads(line)
                entry["tokens"] = [str(tok) for tok in entry["tokens"]]
                fixed_lines.append(json.dumps(entry))
            except Exception as e:
                print(f"Skipping line due to error in {file_path.name}: {e}")
                fixed_lines.append(line.strip())

    # Backup original file
    file_path.rename(file_path.with_suffix(".json.bak"))

    # Write fixed file
    with file_path.open("w") as f:
        for line in fixed_lines:
            f.write(line + "\n")

    print(f"Fixed and backed up: {file_path.name} → {file_path.with_suffix('.json.bak').name}")

# Fix all three subsets
base_path = Path("/home/ec2-user/ner/data")
fix_json_file(base_path / "mydata_train.json")
fix_json_file(base_path / "mydata_val.json")
fix_json_file(base_path / "mydata_test.json")
