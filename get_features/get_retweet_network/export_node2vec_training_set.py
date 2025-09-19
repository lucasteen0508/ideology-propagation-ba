import numpy as np
import pandas as pd

# Node-2-Vec-Embeddings einlesen
df = pd.read_csv("/Users/lucasteen/Desktop/ideology-classifier/ideology_classifier/get_features/data/node2vec_embeddings.csv", dtype=str)

# Bert-Embedding-Training einlesen
training = pd.read_csv(
    '/Users/lucasteen/Desktop/ideology-classifier/ideology_classifier/build_label_propagation_classifier/data/baseline_training_data.csv',
    dtype=str
)

# Zusammenfassen der Embeddings zu Numpy-Array (Restrukturierung: Jedes Embedding lag als eigener Index in Liste vor)
embedding_cols = [str(i) for i in range(64)]
df['node2vec_embedding'] = df[embedding_cols].apply(lambda row: np.array(row, dtype=float), axis=1)

# Alte Spalten entfernen
df = df.drop(columns=embedding_cols)

# Merge auf User-ID: Verbinden von Bert-Embeddings und Node-2-Vec-Embeddings
merged_df = training.merge(df, on="user_id", how="left")

# 64-dim Nullvektor initalisieren
null_vector = np.zeros(64)
# Funktion, um komplett fehlende Embeddings zu ersetzen
def fix_embedding_with_zero(x):
    if x is None or np.isnan(x).all():  # komplett fehlend
        return null_vector
    else:
        # optionale Variante: teilweise NaNs durch 0 ersetzen
        return np.where(np.isnan(x), 0, x)

# Setze Nullvektor für isolierte User
merged_df['node2vec_embedding'] = merged_df['node2vec_embedding'].apply(fix_embedding_with_zero)

# Anzahl ist vor Durchführung des Merges höher, weil User außerhalb des Datensatzes hinzukommen -> logisch konsistent
unique_users = merged_df['user_id'].nunique()
print(merged_df)

# Check auf Vollständigkeit
print(f"Anzahl einzigartiger user_ids: {unique_users}")
print(f"Shape nach Merge: {merged_df.shape}")
print(merged_df[['user_id','node2vec_embedding']].head())

# Prüfen, welche Embeddings Nullvektoren sind
is_null = np.all(np.stack(merged_df['node2vec_embedding'].values) == 0, axis=1)
print(is_null)

# Exportiere Trainingset für Node2Vec+Bert-Modell-Architektur
# merged_df.to_csv('bert_node2vec_training_data.csv')


