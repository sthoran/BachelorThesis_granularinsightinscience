#%%
import json
from pathlib import Path
import pandas as pd
from huggingface_hub import hf_hub_download


def load_scierc(path):
    with open(path, "r", encoding="utf-8") as f:
        scierc_data = [json.loads(line) for line in f]
        return scierc_data
    

def preprocess_data_for_ner(data):
    processed_data = []

    for doc in data:
        sentences = doc['sentences']
        flat_tokens = [token for sent in sentences for token in sent]
        sentence_lengths = [len(sent) for sent in sentences]

        # only 'Method' entity
        method_spans = [span for group in doc['ner'] for span in group if span[2] == 'Method']

        # start all labels as 'O'
        labels = ['O'] * len(flat_tokens)

        # apply BIO labelling
        for start, end, label in method_spans:
            if start < len(flat_tokens):
                labels[start] = 'B-Method'
                if start < end:
                    for i in range(start + 1, end + 1):
                        if i < len(flat_tokens):
                            labels[i] = 'I-Method'

        processed_data.append({
            'flat_tokens': flat_tokens,
            'sentence_lengths': sentence_lengths,
            'original_sentences': sentences,
            'method_ner': method_spans,
            'labels': labels
        })

    return processed_data

def export_to_json(processed_data, output_path):
    data = []
    
    for col in processed_data:
        data.append({
            'tokens': col['flat_tokens'],
            'labels': col['labels']
        })
        
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
        
    return data

if __name__ == '__main__':
    # Download from Hugging Face
    train_path = hf_hub_download("sthoran/scierc_processed_data", filename="train.json", repo_type="dataset")
    dev_path = hf_hub_download("sthoran/scierc_processed_data", filename="dev.json", repo_type="dataset")
    test_path = hf_hub_download("sthoran/scierc_processed_data", filename="test.json", repo_type="dataset")

    # Load
    train_data = load_scierc(train_path)
    dev_data = load_scierc(dev_path)
    test_data = load_scierc(test_path)

    # Preprocess
    processed_train_data = preprocess_data_for_ner(train_data)
    processed_dev_data = preprocess_data_for_ner(dev_data)
    processed_test_data = preprocess_data_for_ner(test_data)

    # Export
    export_to_json(processed_train_data, "ner_train_data.json")
    export_to_json(processed_dev_data, "ner_dev_data.json")
    export_to_json(processed_test_data, "ner_test_data.json")
