import preprocess as pre
import sqlite3




def load_data():
    print('bla')
    con = sqlite3.connect('papers.db')
    df = pd.read_sql_query("SELECT doi, title, subjects , disciplines ,publisher, abstract, language, publicationDate FROM paper", con)
    return df
    

if __name__ == '__main__':
   pass