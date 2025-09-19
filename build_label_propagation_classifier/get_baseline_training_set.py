import pandas as pd

# Embeddings einlesen
user_embeddings = pd.read_csv('/Users/lucasteen/Desktop/ideology-classifier/ideology_classifier/build_label_propagation_classifier/data/user_bert_embeddings.csv', dtype=str)

# Fehlerhafte Spalte droppen
if "Unnamed: 0" in user_embeddings.columns:
    user_embeddings = user_embeddings.drop(columns=["Unnamed: 0"])


# Annotierte User einlesen (Keyword-Basiert)
df_annotated = pd.read_csv(
    "/Users/lucasteen/Desktop/ideology-classifier/ideology_classifier/build_label_propagation_classifier/data/keywords_sampled_combined_anott.csv",
    dtype=str
)

# Annotierte User einlesen (Hashtag-Basiert)
left_users = pd.read_csv(
    '/Users/lucasteen/Desktop/ideology-classifier/ideology_classifier/build_label_propagation_classifier/data/progressive_hashtags_anott.csv',
    dtype=str
)
right_users = pd.read_csv(
    '/Users/lucasteen/Desktop/ideology-classifier/ideology_classifier/build_label_propagation_classifier/data/conservative_hashtags_annot.csv',
    dtype=str
)

# Bereinigen
for df in [left_users, right_users]:
    if "Unnamed: 2" in df.columns:
        df.drop(columns=["Unnamed: 2"], inplace=True)

# Progressive/Konservative Seeds zusammenführen
hashtags_merged = left_users.merge(right_users[['user_id', 'label']], on="user_id", how="outer")
hashtags_merged['label'] = hashtags_merged['label_y'].combine_first(hashtags_merged['label_x'])
hashtags_merged = hashtags_merged.drop(columns=['label_x', 'label_y'], errors='ignore')


# Annotierte Keyword-User mit Trainingsdaten mergen
keywords_merged = user_embeddings.merge(
    df_annotated[["author", "label"]],
    on="author",
    how="left")

# Duplikate entfernen, falls notwendig
keywords_merged = keywords_merged.drop_duplicates('author')

# Annotierte Hashtag-User mit Trainingsdaten mergen -> Reduktion auf
#0    207
#1    158
#2     27
# durch Intersektion
training_merged = keywords_merged.merge(
    hashtags_merged[["user_id", "label"]],
    on="user_id",
    how="left")

# Labelspalten zusammenführen
training_merged['label'] = training_merged['label_y'].combine_first(training_merged['label_x'])
training_merged = training_merged.drop(columns=['label_x', 'label_y'], errors='ignore')

# Reduktion auf 213 annotierte User durch Intersektion von Keyword/Hashtag-Usern
training_merged = training_merged.drop_duplicates(subset='user_id', keep='first')  # behält den ersten Eintrag, reduziert von 216/214

# Duplikate entfernen, falls vorhanden
training_merged = training_merged.drop_duplicates()

# Export Baseline-Trainings-Daten
training_merged.to_csv('baseline_training_data.csv')




