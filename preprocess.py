#%%
import sqlite3
import scispacy
import spacy
import pandas as pd
import seaborn as sns
import re
#%%
# Connect to SQLite
con = sqlite3.connect("papers.db")

# Read only abstracts (and maybe titles too)
df = pd.read_sql_query("SELECT doi, title, abstract, language FROM paper", con)
#%%
# Combine title + abstract if useful
df["text"] = df["title"].fillna('') + ". " + df["abstract"].fillna('')
# %%
df['language'].unique()
# %%
df = df.drop_duplicates(subset="doi")
# %%
df.describe()
# %%
sns.histplot(df, x= 'language')
# %%
df = df[df['language'] != 'de']
df.describe()
# %%
def preprocess_for_biobert_ner(text):
    if not text:
        return ""
    
    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    
    # Clean up any stray characters
    text = text.strip()
    
    return text

# %%
df["cleaned_text"] = df["text"].fillna("").apply(preprocess_for_ner)

# %%
nlp = spacy.load("en_ner_bionlp13cg_md")
# %%
