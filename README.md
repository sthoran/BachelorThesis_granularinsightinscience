#  Bachelor Thesis – Granular Insight into Scientific AI Methods Using NER

This project builds a full NLP pipeline to extract **AI and machine learning method mentions** from scientific literature using **Named Entity Recognition (NER)**. Developed as part of a Bachelor thesis, it aims to create **granular insight into the use of AI methods in drug discovery** publications.

---

##  Objective

To build a reproducible and domain-adapted NER system that:
- Detects `"Method"` entities (e.g., BERT, CNN, GNN)
- Trains and evaluates transformer-based models (BioBERT, SciBERT, DeBERTa)
- Combines real-world abstracts (from SpringerNature) with a curated benchmark dataset (SciERC)

---

## Hugging Face Integration

All datasets, config files, and trained models are hosted under the Hugging Face account [`sthoran`](https://huggingface.co/sthoran).

### Datasets
| Source | Repo |
|--------|------|
| Raw + BIO-labeled abstracts | [`sthoran/aidrugcorpus`](https://huggingface.co/datasets/sthoran/aidrugcorpus) |
| Modified SciERC (Method only) | [`sthoran/scierc_processed_data`](https://huggingface.co/datasets/sthoran/method_only_scierc) |
|  full SciERC preprocessed | [`sthoran/scierc_processed_data`](https://huggingface.co/datasets/sthoran/scierc_processed_data) |

### YAML Configs
- [`sthoran/ner-configs`](https://huggingface.co/datasets/sthoran/ner-configs)

### Trained Models

#### Phase 1: Trained on method-only SciERC

## initial Models Used to train method_only SciERC dataset

- [BioBERT](https://huggingface.co/dmis-lab/biobert-base-cased-v1.1)
- [SciBERT](https://huggingface.co/allenai/scibert_scivocab_cased)
- [DeBERTa](https://huggingface.co/microsoft/deberta-base)

These models were first fine-tuned **exclusively on `"Method"` entities** from the SciERC dataset:

| Model       | Hugging Face Repo                                      |
|-------------|--------------------------------------------------------|
| BioBERT     | [`sthoran/method_only_scierc_biobert`](https://huggingface.co/datasets/sthoran/method_only_scierc_biobert) → `/biobert`  
| SciBERT     | [`sthoran/method_only_scierc_scibert`](https://huggingface.co/datasets/sthoran/method_only_scierc_scibert) → `/scibert`  
| DeBERTa     | [`sthoran/method_only_scierc_deberta`](https://huggingface.co/datasets/sthoran/method_only_scierc_deberta) → `/deberta`  

Each folder contains:
- `model/` → model weights
- `tokenizer/` → tokenizer files
- `metrics.json` → performance on SciERC

#### Phase 2: Fine-tuned on real-world abstracts (aidrugcorpus)

The above models were **further fine-tuned** on the `aidrugcorpus` dataset (abstracts extracted from SpringerNature):

| Model       | Hugging Face Repo                                            |
|-------------|--------------------------------------------------------------|
| BioBERT     | [`sthoran/aidrugcorpus_finetuned_biobert`](https://huggingface.co/datasets/sthoran/aidrugcorpus_finetuned_biobert) → `/biobert`  
| SciBERT     | [`sthoran/aidrugcorpus_finetuned_scibert`](https://huggingface.co/datasets/sthoran/aidrugcorpus_finetuned_scibert) → `/scibert`  
| DeBERTa     | [`sthoran/aidrugcorpus_finetuned_deberta`](https://huggingface.co/datasets/sthoran/aidrugcorpus_finetuned_deberta) → `/deberta`  

---

## Pipeline Overview

### Step 1: Extract & Filter Abstracts
- `scripts/fetch_papers_to_db.py`: Query SpringerNature API and store results in SQLite
- `scripts/generate_filtered_ai_abstracts.py`: Filter for abstracts mentioning AI methods

### Step 2: Explore & Clean
- `notebooks/data_wrangling.ipynb`:  
  - Filter abstracts by AI abbreviation keywords  
  - Remove false positives  
  - Explore top publishers and trends over time

### Step 3: BIO Tagging
- `utils/bio_converter.py`: Tag filtered abstracts using BIO format
- `utils/convert_to_json.py`: Split into train/val/test for training
- `utils/scierc_bio_label.py`: Apply same BIO tagging to SciERC (Method only)

### Step 4: Model Training
- `scripts/train_scierc_model.py`: Train BioBERT, SciBERT, DeBERTa on SciERC (Method only)
- `scripts/train_mydata_with_model.py`: Fine-tune the above models on `aidrugcorpus`

---

##  Repository Structure


.
├── notebooks/
│ └── data_wrangling.ipynb # Exploratory filtering, abbreviation handling, statistics
├── scripts/
│ ├── fetch_papers_to_db.py # Extracts SpringerNature API data to SQLite
│ ├── generate_filtered_ai_abstracts.py # Filters abstracts based on AI method keywords
│ ├── train_mydata_with_model.py # Trains BERT models on aidrugcorpus + method-only SciERC
│ └── train_scierc_model.py # Trains BERT models on modified SciERC data only
├── utils/
│ ├── bio_converter.py # Converts raw_dataset.csv → BIO-labeled ner_bio_dataset.csv
│ ├── convert_to_json.py # Converts BIO-labeled CSV → train/val/test JSON files
│ └── scierc_bio_label.py # Converts processed SciERC → BIO format
├── requirements.txt
└── README.md



##  Evaluation Metrics

Evaluation is done using the [`seqeval`](https://github.com/chakki-works/seqeval) library with:
- **Precision**
- **Recall**
- **F1 Score**
- **Accuracy**

All metrics are saved as JSON files during training.

## Installation
git clone https://github.com/yourusername/BachelorThesis_granularinsightinscience.git
cd BachelorThesis_granularinsightinscience

python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

pip install -r requirements.txt




