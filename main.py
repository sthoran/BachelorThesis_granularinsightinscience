import preprocess as pre
import sqlite3
import pandas as pd


def load_data():
    print('bla')
    con = sqlite3.connect('papers.db')
    df = pd.read_sql_query("SELECT doi, title, publisher, abstract, language, publicationDate FROM paper", con)
    return df
    

if __name__ == '__main__':
    df = load_data()
    clean_obj = pre.Data_cleanup(df)
    df_cleaned = clean_obj.dataframe_cleanup()
    preprossed_obj = pre.Preprocess(df_cleaned, 'text')
    df_preprossed = preprossed_obj.preprocess_for_biobert_gene_ner()
    df_preprossed = preprossed_obj.preprocess_for_biobert_disease_or_chemical_ner()