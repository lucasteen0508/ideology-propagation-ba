import pandas as pd
import re

# Logik des Pre-Processing: Hauptsächlich Bewahrung von syntaktischer Struktur;
# alles, was keine semantische/syntaktische Info liefert, komplett entfernen

# Pre-Processing-Funktion definieren
def load_and_preprocess_tweets(csv_path: str) -> pd.DataFrame:

    # CSV einlesen
    df = pd.read_csv(csv_path, dtype= str)

    # Pfad anpassen, falls nötig
    df.rename(columns={"text": "tweet"}, inplace=True)  # Spalte sicherheitshalber umbenennen

    # Spalten als String casten
    df['tweet'] = df['tweet'].astype(str)
    df['tweet_id'] = df['tweet_id'].astype(str)

    # Hilfsfunktionen definieren
    def remove_hashtags(text: str) -> str:
        if not isinstance(text, str):
            return ""
        # 1. Entfernt führende Hashtags am Anfang des Tweets (z. B. "#afd #noafd Hallo" → "Hallo")
        text = re.sub(r'^(#\w+\s*)+', '', text).strip()

        # 2. Entfernt das "#" bei allen verbleibenden Hashtags (z. B. "Guten Morgen #hendrickwuest" → "Guten Morgen hendrickwuest")
        # -> Bewahrung von Syntax
        text = re.sub(r'#(\w+)', r'\1', text)

        return text

    # Entferne Links
    def remove_tco_links(text):
        return re.sub(r"https?://t\.co/\S+", "", text)

    # Analog zur Hashtag-Logik
    def remove_mentions(text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = re.sub(r'^(@\w+\s*)+', '', text).strip()
        text = re.sub(r'@(\w+)', r'\1', text)
        return text

    # Entferne Emojies
    def remove_emojis(text: str) -> str:
        if not isinstance(text, str):
            return ""

        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"
            u"\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F6FF"
            u"\U0001F1E0-\U0001F1FF"
            u"\U00002500-\U00002BEF"
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001f926-\U0001f937"
            u"\U00010000-\U0010ffff"
            u"\u2640-\u2642"
            u"\u2600-\u2B55"
            u"\u200d"
            u"\u23cf"
            u"\u23e9"
            u"\u231a"
            u"\ufe0f"
            u"\u3030"
            "]+", flags=re.UNICODE
        )
        return emoji_pattern.sub(r'', text)

    # Doch behalten, da sie zu Tokens gehören oder als Tokens bewertet werden sollten
    #def remove_numbers(text):
        #return re.sub(r'\d+', '', text)

    # Satzstrukturen bleiben erhalten
    def remove_punctuation_keep_sentence(text):
        return re.sub(r"[^A-Za-z0-9äöüÄÖÜß\s,.!?]", "", text)

    # Skelett-Funktion für Pre-Processing
    def preprocess_tweet(text):

        text_clean = text.lower() # lowercase, keine Hilfsfunktion
        text_clean = remove_hashtags(text_clean)
        text_clean = remove_tco_links(text_clean)
        text_clean= remove_mentions(text_clean)
        text_clean = remove_emojis(text_clean)
        #text_clean = remove_numbers(text_clean)
        text_clean = remove_punctuation_keep_sentence(text_clean)

        return text_clean

    # Anwenden auf den DataFrame
    df["tweets_processed"] = df["tweet"].apply(preprocess_tweet)

    return df["tweets_processed"]
