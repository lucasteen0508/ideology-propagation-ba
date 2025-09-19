import networkx as nx
from node2vec import Node2Vec
import numpy as np
import pandas as pd
import umap.umap_ as umap
import matplotlib.pyplot as plt
from tqdm import tqdm

# Lese Kantenliste ein
df = pd.read_csv("/Users/lucasteen/Desktop/ideology-classifier/ideology_classifier/get_features/data/retweet_edges.csv", dtype=str)

# Checke User -> 2051 User, die nicht isoliert sind
unique_users = df['user_id'].nunique()
print(unique_users)
print(len(df))

# Gerichteten Graph definieren
G = nx.DiGraph()

# Graph auf Bais der Kantenliste erzeugen mit Fortschrittsanzeige
for _, row in tqdm(df.iterrows(), total=len(df), desc="Adding edges"):
    G.add_edge(str(row["user_id"]), str(row["retweeted_user_id"]), weight=float(row.get("weight", 1)))

# Node2Vec-Konstruktor, Hyperparameter auf Lokalität stimmen
node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4, p=1, q=0.5)

# Embeddings trainieren
model = node2vec.fit(window=10, min_count=1, batch_words=4)

# Embeddings extrahieren
node_embeddings = {str(node): model.wv[str(node)] for node in tqdm(G.nodes(), desc="Extracting embeddings")}

# Matcht Anzahl der Knoten User-Anzahl?
print(len(node_embeddings))

# In DataFrame umwandeln
node2vec_df = pd.DataFrame.from_dict(node_embeddings, orient="index")
node2vec_df.index.name = "user_id"
node2vec_df.reset_index(inplace=True)

# UMAP-Reduktion (Projektion der Embeddings in zwei-dimensionalen Raum)
reducer = umap.UMAP(n_components=2, random_state=42)
emb_2d = reducer.fit_transform(node2vec_df.drop(columns=["user_id"]).values)

# Plot erstellen
plt.figure(figsize=(6, 6))
plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c="blue", alpha=0.7)
plt.title("Node2Vec Embeddings (UMAP)")
plt.savefig("node2vec_umap.png", dpi=300, bbox_inches="tight")

# Embeddings exportieren
node2vec_df.to_csv("node2vec_embeddings.csv", index=False)
