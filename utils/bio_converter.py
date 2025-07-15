import pandas as pd
import spacy
from huggingface_hub import hf_hub_download

def bio_label(text, ai_method):
    nlp = spacy.blank("en")  
    doc = nlp(str(text))
    tokens = [token.text for token in doc]
    labels = ['O'] * len(tokens)

    ai_method_tokens = [t.text for t in nlp(str(ai_method).lower())]
    token_texts_lower = [t.lower() for t in tokens]

    for i in range(len(tokens) - len(ai_method_tokens) + 1):
        if token_texts_lower[i:i+len(ai_method_tokens)] == ai_method_tokens:
            labels[i] = 'B-METHOD'
            for j in range(1, len(ai_method_tokens)):
                labels[i+j] = 'I-METHOD'
            break  

    return list(zip(tokens, labels))

def main():
    # Download raw_dataset.csv from Hugging Face
    csv_path = hf_hub_download(
        repo_id="sthoran/aidrugcorpus",
        filename="raw_dataset.csv",
        repo_type="dataset"
    )

    # Load dataset
    df = pd.read_csv(csv_path)
    nlp = spacy.blank("en")
    all_bio_rows = []

    for idx, row in df.iterrows():
        text = row['text']
        ai_method = row['AI_method_text']
        doc = nlp(str(text))
        tokens = [token.text for token in doc]
        labels = ['O'] * len(tokens)

        ai_method_tokens = [t.text for t in nlp(str(ai_method).lower())]
        token_texts_lower = [t.lower() for t in tokens]

        for i in range(len(tokens) - len(ai_method_tokens) + 1):
            if token_texts_lower[i:i+len(ai_method_tokens)] == ai_method_tokens:
                labels[i] = 'B-METHOD'
                for j in range(1, len(ai_method_tokens)):
                    labels[i+j] = 'I-METHOD'
                break

        for token, label in zip(tokens, labels):
            all_bio_rows.append({
                "sentence_id": idx,
                "token": token,
                "label": label
            })

    # Save BIO-labeled dataset
    bio_df = pd.DataFrame(all_bio_rows)
    bio_df.to_csv("ner_bio_dataset.csv", index=False)

if __name__ == "__main__":
    main()
