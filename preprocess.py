#%%
import sqlite3

import pandas as pd
import seaborn as sns
import re
import re
import unicodedata
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')


# Connect to SQLite
class Data_cleanup:
    def __init__(self,data):
        self.df = data
    
    # def get_df_sqlite(self):
    #     con = sqlite3.connect('papers.db')
    #     self.df = pd.read_sql_query("SELECT doi, title, abstract, language FROM paper", con)
    #     con.close()
    #     return self.df

    def dataframe_cleanup(self):
        self.df = self.df.drop_duplicates(subset="doi")
        self.df = self.df[self.df['language'] != 'de']
        self.df["text"] = self.df["title"].fillna('') + ". " + self.df["abstract"].fillna('')
        return self.df
        
    def save_dataframe(self,filename):
        self.df.to_csv(filename, index = False)


class Preprocess:
    def __init__(self,df: pd.DataFrame, column: str):
        self.df = df.copy()
        self.column = column
        self.base_col = self.df[column]
    
    
    def preprocess_for_biobert_gene_ner(self, new_col_name = 'biobert_gene_text'):
        
        # Remove HTML tags
        self.df[new_col_name] = re.sub(r"<.*?>", "", self.base_col)
        
        # Remove URLs
        self.df[new_col_name] = re.sub(r"http\S+|www\S+", "", self.base_col)
        
        # Normalize whitespace
        self.df[new_col_name] = re.sub(r"\s+", " ", self.base_col)
        
        # Clean up any stray characters
        self.df[new_col_name] = self.base_col.strip()
        
        return self.df
    
    def preprocess_for_biobert_disease_or_chemical_ner(self, new_column = 'biobert_disease_chemical_text'):

        # Remove HTML tags
        self.df[new_column] = re.sub(r"<.*?>", "", self.base_col)

        # Remove URLs
        self.df[new_column] = re.sub(r"http\S+|www\S+", "", self.base_col)

        # Normalize Unicode (e.g., replace smart quotes, special symbols)
        self.df[new_column] = unicodedata.normalize("NFKD", self.base_col)

        # Normalize Greek letters manually if needed
        greek_replacements = {
            '\u03b1': 'alpha',  # α
            '\u03b2': 'beta',   # β
            '\u03b3': 'gamma',
            '\u03b4': 'delta',
        }
        for k, v in greek_replacements.items():
            self.df[new_column] = self.base_col.replace(k, v)

        # Normalize whitespace
        self.df[new_column] = re.sub(r"\s+", " ", self.base_col)

        # Trim
        self.df[new_column] = self.base_col.strip()

        # Sentence splitting (important for context-sensitive NER)
        self.df[new_column] = self.df[new_column].apply(sent_tokenize)

        return self.df  # Return list of clean sentences
    
    
    def save_dataframe(self,filename):
        self.df.to_csv(filename, index = False)
    
        

print('test')

