# Bachelor Thesis – Granular Insight into Scientific AI Methods Using NER

This project builds a full pipeline for extracting **AI and machine learning method mentions** from scientific literature using **Named Entity Recognition (NER)**. It was developed as part of a Bachelor thesis with the goal of gaining **granular insight into the methods used in AI-related drug discovery research**.

The pipeline covers:
- Data extraction from scientific APIs (SpringerNature)
- Filtering and cleaning AI-related abstracts
- BIO tagging of method mentions
- Dataset creation (custom + SciERC)
- Fine-tuning transformer models for token classification
- Evaluation using standard NER metrics

---

## Objective

To create an automated and reproducible NER pipeline that identifies and extracts `"Method"` entities (e.g., `BERT`, `CNN`, `GNN`) from biomedical and scientific literature, and benchmark model performance using a modified version of the [SciERC](https://github.com/allenai/scierc) dataset.

---

## Hugging Face Integration

All data, configuration files, and trained model outputs are hosted on the Hugging Face Hub under the namespace **`sthoran`**.

| Type         | Repository                                                                 |
|--------------|----------------------------------------------------------------------------|
| Raw & Processed Data | [`sthoran/aidrugcorpus`](https://huggingface.co/datasets/sthoran/aidrugcorpus) |
| SciERC (modified)    | [`sthoran/scierc_processed_data`](https://huggingface.co/datasets/sthoran/scierc_processed_data) |
| YAML Configs         | [`sthoran/ner-configs`](https://huggingface.co/datasets/sthoran/ner-configs) |
| Fine-Tuned Models     | [`sthoran/aidrugcorpus_finetuned_biobert`](https://huggingface.co/datasets/sthoran/aidrugcorpus_finetuned_biobert) → `/biobert`  
|                          | [`sthoran/aidrugcorpus_finetuned_scibert`](https://huggingface.co/datasets/sthoran/aidrugcorpus_finetuned_scibert) → `/scibert`  
|                          | [`sthoran/aidrugcorpus_finetuned_deberta`](https://huggingface.co/datasets/sthoran/aidrugcorpus_finetuned_deberta) → `/deberta` |

---

## Project Structure

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


---

## Pipeline Overview

### 1. Data Extraction & Filtering
- `scripts/fetch_papers_to_db.py`: queries SpringerNature API and stores responses in a SQLite database.
- `scripts/generate_filtered_ai_abstracts.py`: filters abstracts for AI method keywords and saves them for further processing.

### 2. Data Cleaning & Statistics
- `notebooks/data_wrangling.ipynb`:  
  - Handles abbreviation cleaning  
  - Manual evaluation and removal  
  - Exploratory analysis: top publishers, papers per year

### 3. BIO Labeling
- `utils/bio_converter.py`: BIO-tags AI method mentions in abstracts.
- `utils/convert_to_json.py`: Converts tagged data into `mydata_train.json`, `mydata_val.json`, `mydata_test.json`.
- `utils/scierc_bio_label.py`: Applies BIO labeling to method-only SciERC.

### 4. Model Training
- `scripts/train_mydata_with_model.py`: Fine-tunes BioBERT, SciBERT, DeBERTa on custom aidrugcorpus + SciERC data.
- `scripts/train_scierc_model.py`: Trains same models on only modified SciERC dataset.

---

## Models Used

- [BioBERT](https://huggingface.co/dmis-lab/biobert-base-cased-v1.1)
- [SciBERT](https://huggingface.co/allenai/scibert_scivocab_cased)
- [DeBERTa](https://huggingface.co/microsoft/deberta-base)

Fine-tuned models and metrics are stored in:
- `sthoran/aidrugcorpus_finetuned_biobert/biobert`
- `sthoran/aidrugcorpus_finetuned_scibert/scibert`
- `sthoran/aidrugcorpus_finetuned_deberta/deberta`

Each folder contains:
- `model/` – fine-tuned weights
- `tokenizer/` – tokenizer files
- `metrics.json` – evaluation metrics per epoch

---

##  Evaluation Metrics

Evaluation is done using the [`seqeval`](https://github.com/chakki-works/seqeval) library with:
- **Precision**
- **Recall**
- **F1 Score**
- **Accuracy**

All metrics are saved as JSON files during training .



