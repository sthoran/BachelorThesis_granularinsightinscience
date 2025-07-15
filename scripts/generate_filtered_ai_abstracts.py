import sqlite3
import pandas as pd
from huggingface_hub import hf_hub_download

#  Data Cleanup Class 
class DataCleanup:
    def __init__(self):
        self.df = None

    def get_df_sqlite(self, db_repo="sthoran/aidrugcorpus", filename="papers.db"):
        db_path = hf_hub_download(
            repo_id=db_repo,
            filename=filename,
            repo_type="dataset"
        )
        con = sqlite3.connect(db_path)
        self.df = pd.read_sql_query("""
            SELECT 
                doi, title, subjects, disciplines, publisher,
                abstract, language, publicationDate
            FROM paper
        """, con)
        con.close()
        return self.df

    def dataframe_cleanup(self):
        self.df = self.df.drop_duplicates(subset="doi")
        self.df = self.df[self.df['language'] != 'de']
        self.df["text"] = self.df["title"].fillna('') + ". " + self.df["abstract"].fillna('')
        self.df["text"] = self.df["text"].str.lower()
        return self.df


class Preprocess:
    def __init__(self, df: pd.DataFrame, column: str):
        self.df = df.copy()
        self.column = column
        self.base_col = self.df[self.column]

    def filtered_AI_dataframe(self, new_col_name='AI_method_text'):
        ai_methods = [
            # Neural Networks
            "neural network", "deep neural network", "artificial neural network",
            "multi-layer perceptron", " mlp ", "feedforward neural network", "fnn",

            # Convolutional Models
            "convolutional neural network", " cnn ", "1d cnn", "2d cnn", "3d cnn",
            "residual neural network", "resnet", "u-net", "inceptionnet",
            "efficientnet", "densenet",

            # Recurrent Models
            "recurrent neural network", " rnn ", "long short-term memory", "lstm",
            "gated recurrent unit", " gru ", "bidirectional lstm", "bilstm",

            # Transformers & Attention
            "transformer", " bert ", " gpt ", "roberta", " t5 ", "xlnet", "electra",
            "deberta", "biobert", "scibert", "pubmedbert", "vision transformer",
            " vit ", "performer", "linformer", "longformer", "albert", "distilbert",

            # Graph-Based Methods
            "graph neural network", " gnn ", "graph convolutional network", " gcn ",
            "graph attention network", " gat ", "message passing neural network", " mpnn ",
            "graph isomorphism network", " gin ", "graph transformer",

            # Autoencoders
            "autoencoder", " ae ", "variational autoencoder", " vae ",
            "sparse autoencoder", "denoising autoencoder",

            # Generative Models
            "generative adversarial network", " gan ", "conditional gan", " cgan ",
            "cyclegan", "stylegan", "diffusion model",
            "denoising diffusion probabilistic model", " ddpm ",

            # Tree-Based Models
            "decision tree", "random forest", "gradient boosting", "xgboost",
            "lightgbm", "catboost", "adaboost",

            # Linear Models
            "logistic regression", "linear regression", "ridge regression",
            "lasso regression", "elasticnet",

            # Kernel Methods
            "support vector machine", "svm", "kernel ridge regression",
            "gaussian process", "gaussian processes", " gp ",

            # Clustering & Dim Reduction
            "k-means", "dbscan", "agglomerative clustering",
            "hierarchical clustering", "t-sne", " pca ", " umap ",

            # Reinforcement Learning
            "q-learning", "deep q-network", " dqn ", "proximal policy optimization",
            " ppo ", " a3c ", " a2c ", "reinforce", "actor-critic",
            "monte carlo tree search", " mcts ",

            # Other Methods
            "self-supervised learning", "few-shot learning", "zero-shot learning",
            "contrastive learning", "transfer learning", "meta-learning",
            "active learning", "multi-task learning", "federated learning"
        ]

        self.df[new_col_name] = self.base_col.apply(
            lambda x: next((kw for kw in ai_methods if kw in x), None)
        )
        return self.df[self.df[new_col_name].notna()].reset_index(drop=True)

if __name__ == '__main__':
    # load and clean the data
    cleaner = DataCleanup()
    df = cleaner.get_df_sqlite()
    df = cleaner.dataframe_cleanup()

    # filter to AI-relevant abstracts
    prep = Preprocess(df, column='text')
    df_ai = prep.filtered_AI_dataframe()

    # save
    df_ai.to_csv('filtered_ai_abstracts.csv', index=False)
