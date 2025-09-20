import requests
import pandas as pd
import networkx as nx
import numpy as np

# Hydration laden
df = pd.read_csv('/Volumes/Seagate Portabl/Users/luca/Arbeit/BA-Arbeit/SVM/hydration_output.csv', dtype=str)

# Einzigartige User-IDs speichern
df["user_id"] = df["user_id"].astype(str)
unique_user_ids = df["user_id"].unique()

# Erzeugt Liste, die alle Retweeter eines Users enthält (ansonsten analoges Hydration-Vorgehen)
def get_retweeters(tweet_ids):

    retweeter_list = []

    for tweet in tweet_ids: # Test

        print(tweet)

        url = "https://api.twitterapi.io/twitter/tweet/retweeters"

        querystring = {"tweetId": tweet}

        headers = {"X-API-Key": "ausgegraut"}

        response = requests.get(url, headers=headers, params=querystring)

        if response.ok:

            try:
                data = response.json()

                for retweeter in  data.get('users', []):
                    retweet_id = retweeter.get('id', np.nan)
                    retweeter_list.append(retweet_id)

            except ValueError as e:
                print("Fehler beim Parsen von JSON:", e)
        else:
            print("Request fehlgeschlagen:", response.status_code)


    return list(set(retweeter_list))

# Dictionary initialisieren
results = {}

# Iteriere über alle einzigartigen User und wende get_retweeters() an
for user in unique_user_ids:

    # Subset der Zeilen, die zu diesem User gehören
    user_subset = df.loc[df["user_id"] == user]

    # Extrahiere alle Tweet-IDs des Users
    tweet_ids = user_subset["tweet_id"].unique()

    # Mappe Retweet-Liste auf User
    results[user] = get_retweeters(tweet_ids)

# Networkgraph definieren
G = nx.DiGraph()

# Gewichtete Edge-Liste konstruieren (Gerichtet, tatsächliche Retweets; also nicht A->B impliziert B->A)
for user, retweeted_list in results.items():
    for retweeted_user in retweeted_list:
        if G.has_edge(user, retweeted_user):
            G[user][retweeted_user]["weight"] += 1
        else:
            G.add_edge(user, retweeted_user, weight=1)

# Alle Kanten inklusive Gewicht extrahieren
edge_list = []
for u, v, d in G.edges(data=True):
    edge_list.append({
        "user_id": u,
        "retweeted_user_id": v,
        "weight": d.get("weight", 1)  # Standardgewicht=1
    })

# Edge-Liste in DataFrame umwandeln
df_edges = pd.DataFrame(edge_list)

# CSV speichern
df_edges.to_csv("retweet_edges.csv", index=False)

