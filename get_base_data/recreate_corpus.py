import json
import glob
import os
import pandas as pd

# Mache die Unterteilung von Train-/Test-Data rückgängig, um Zugriff auf den gesamten Datensatz zu erhalten
def merge_json_files(input_folder, output_file):
    merged_data = []

    # Alle JSON-Dateien im Ordner einlesen
    for file in glob.glob(os.path.join(input_folder, "*.json")):
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Falls es eine Liste ist → extend, falls es ein Dict ist → append
            if isinstance(data, list):
                merged_data.extend(data)
            else:
                merged_data.append(data)

    # In DataFrame umwandeln
    df = pd.DataFrame(merged_data)
    print(len(df))

    # Als CSV speichern
    df.to_csv(output_file, index=False, encoding="utf-8")

    print(f"✅ {len(merged_data)} Einträge zusammengeführt in {output_file}")

# Führe aus
merge_json_files("/Users/lucasteen/Desktop/ideology-classifier/ideology_classifier/get_base_data/data/German-Elections-NRW22-Stance-Dataset-main/data",
                 "data/full_stance_22_corpus.csv")
