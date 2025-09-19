import requests
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
from preprocessing import load_and_preprocess_tweets

# TQDM für Pandas aktivieren
tqdm.pandas()

# Hydration laden
df = pd.read_csv("/Volumes/Seagate Portabl/Users/luca/Arbeit/BA-Arbeit/SVM/hydration_output", dtype= str)

# Prozessierte Tweets laden
df['tweets_processed'] = load_and_preprocess_tweets("/Volumes/Seagate Portabl/Users/luca/Arbeit/BA-Arbeit/SVM/hydration_output.csv")
print(df[["text",'tweets_processed']].head())
print(df[["text", "tweets_processed"]].head(1).T)

# Aggregiere Tweets pro User über Apply-Funktion
df["tweets_processed_aggregated"] = df.groupby("user_id")["tweets_processed"].transform(
    lambda x: " ".join(filter(None, x))
)

# Entferne leere Spalten -> Knapp 2000 Reduktion (nicht über API abrufbar)
df = df.dropna(how="all")
print(len(df))

# Alle Zeilen entfernen, wo Profil-Text NaN ist
df = df.dropna(subset=["profile_text"])
print(len(df))

# Überprüfen, ob Zeilen existieren, in denen Profiltext leerer String ist -> Perfekt, für jeden User existiert ein Profil
empty_profile_text = df[df["profile_text"] == ""]
print(empty_profile_text)

# Duplikate entfernen, die im Prozess entstanden sind
df = df.drop_duplicates(subset = 'user_id')

# Anzahl User checken
print(len(df))

# Pre-Trained S-Bert-Modell initialisieren
model = SentenceTransformer("distiluse-base-multilingual-cased-v2")

def get_profile_and_tweets_embedding(row, alpha=0.7):

    # Erzeuge Embeddings für Profiltexte
    profile = str(row["profile_text"]) if pd.notna(row["profile_text"]) else ""
    if profile.strip():
        profile_emb = model.encode(profile)
    else:
        profile_emb = np.zeros(model.get_sentence_embedding_dimension())

    # Erzeuge Embeddings für aggreguierte Tweets
    tweet_cols = [col for col in df.columns if col.startswith("tweets_processed_aggregated")]
    tweet_texts = [str(row[col]) for col in tweet_cols if pd.notna(row[col]) and str(row[col]).strip()]

    if tweet_texts:
        tweet_embs = model.encode(tweet_texts)
        tweets_emb = np.mean(tweet_embs, axis=0)  # Mittelung über alle Tweets
    else:
        tweets_emb = np.zeros(model.get_sentence_embedding_dimension())

    # Kombination: gewichtetes Mittel -> Fokus auf Profiltexte, Neugeweichtung innerhalb des Vektors
    combined_emb = alpha * profile_emb + (1 - alpha) * tweets_emb

    return combined_emb

# Funktion ausführen, Embeddings in neuer Spalte speichern
tqdm.pandas()
df["bert_embeddings"] = df.progress_apply(get_profile_and_tweets_embedding, axis=1)

# Exportiere S-Bert-Baseline-Training-Set
df.to_csv('user_bert_embeddings.csv')


