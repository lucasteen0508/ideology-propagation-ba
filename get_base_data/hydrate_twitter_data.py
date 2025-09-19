import numpy as np
import requests
import pandas as pd
import time
import os
from tqdm import tqdm

# Vollständigen Original-Korpus laden
df = pd.read_csv('/Users/lucasteen/Desktop/ideology-classifier/ideology_classifier/get_base_data/data/full_stance_22_corpus.csv', dtype = str)

# Tweets als Liste speichern
tweet_ids = df['tweet_id'].astype(str).to_list()

# Chunking-Funktion
def chunk_list(lst, n=100):
    return [lst[i:i+n] for i in range(0, len(lst), n)]

# Nested List mit 100 IDs pro Chunk
chunked_ids = chunk_list(tweet_ids, 100)

# Liste, über die die Zeilen des Dataframes während der Hydration aktualisiert werden
all_rows = []

# Output-File bestimmen
output_file = "data/hydration_output.csv"

# If-Verzweigung, falls File bereits existiert (für den Fall eines vorzeitigen Abbruchs)
if os.path.exists(output_file):
    df_existing = pd.read_csv(output_file)
    processed_ids = set(df_existing['tweet_id'].astype(str).to_list())
    all_rows.extend(df_existing.to_dict('records'))
else:
    processed_ids = set()

# Iteration über alle Chunks mit Progress-Bar
for i, chunk in enumerate(tqdm(chunked_ids, desc="Processing Chunks")):
    if all(tid in processed_ids for tid in chunk):
        continue

    query = ','.join(chunk)
    success = False
    retries = 3 # Erlaube maximal 3 Retries (sonst wahrscheinlich Server-Probleme oder Ähnliches)

    # Erzeuge Query über für jeden Chunk, stelle Anfrage an die Api, hydriere Daten, konstruiere Dataframe
    while not success and retries > 0:
        try:
            url = "https://api.twitterapi.io/twitter/tweets"
            querystring = {"tweet_ids": query}
            headers = {"X-API-Key": "0c92ef71e2b843e293f423777e085adf"}
            response = requests.get(url, headers=headers, params=querystring, timeout=60)

            if response.ok:
                data = response.json()

                # Extrahiere Daten aus den von der API übergebenen JSON-Files (für jeden Tweet)
                for tweet in data.get('tweets', []):
                    tweet_id = tweet.get('id', np.nan)
                    text = tweet.get('text', np.nan)
                    author = tweet.get('author', {}).get('userName', np.nan)
                    author_id = tweet.get('author', {}).get('id', np.nan)
                    profile_description = tweet.get('author', {}).get('description', np.nan)
                    hashtags = tweet.get('entities', {}).get('hashtags', np.nan)

                    # Speichere Metadaten für jeden Tweet in Row ab
                    all_rows.append({
                        "tweet_id": tweet_id,
                        "author": author,
                        "text": text,
                        "user_id": author_id,
                        "profile_text": profile_description,
                        "hashtags": hashtags,
                    })

                success = True
            else:
                retries -= 1
                time.sleep(5) # Ratelimit abfangen

        except (requests.exceptions.RequestException, ValueError) as e:
            retries -= 1
            time.sleep(5) # Ratelimit abfangen

    pd.DataFrame(all_rows).to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"Fertig! Alle gesammelten Tweets gespeichert in {output_file}")
