import pandas as pd
from sklearn.semi_supervised import LabelPropagation
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import ast

# Lade Dataset
dataset = pd.read_csv(
    '/Users/lucasteen/Desktop/ideology-classifier/ideology_classifier/build_label_propagation_classifier/data/bert_node2vec_training_data.csv',
    dtype=str
)

# Duplikate entfernen, falls vorhanden
dataset = dataset.drop_duplicates()
# Bereinigen
dataset=dataset.drop(dataset.columns[0], axis=1)
print(f"Dataset Länge nach Duplikaten entfernen: {len(dataset)}")

# Labels als Int64 initialisieren (erlaubt NaN)
dataset["label"] = dataset["label"].astype("Int64")

# Unsichere Labels (2) direkt auf -1 setzen
dataset.loc[dataset['label'] == 2, 'label'] = -1

# Embeddings-Spalte umwandeln (String -> NumPy-Array)
def str_to_array_whitespace(s):
    if pd.isna(s):
        return np.nan
    s = s.strip('[]')
    numbers = [float(x) for x in s.split() if x]
    return np.array(numbers, dtype=float)

# Funktion ausführen
dataset['bert_embeddings'] = dataset['bert_embeddings'].apply(str_to_array_whitespace)

# Node2Vec-Spalte umwaneln (String -> NumPy-Array)
dataset['node2vec_embedding'] = dataset['node2vec_embedding'].apply(
    lambda x: np.fromstring(x.strip("[]"), sep=" ") if isinstance(x, str) else x
)

# S-BERT-Embeddings und Node2Vec-Embdeddings zusammenführen
X_combined = np.stack([
    np.concatenate([b, n])
    for b, n in zip(dataset['bert_embeddings'], dataset['node2vec_embedding'])]
)

# Standardisierung für RBF-Kernel und um Vergleich mit PCA zu garantieren
X_scaled = StandardScaler().fit_transform(X_combined)
print("Feature-Matrix Shape:", X_combined.shape)

# Labels vorbereiten (NaN für Label-Propagation bereits auf -1 setzen)
y = dataset['label'].values
y = y.fillna(-1)

# Exakt 150 Seeds pro Klasse mit Random-Seed
rng = np.random.default_rng(42)
seed_indices_per_class = {}
for cls in [0, 1]:
    cls_indices = np.where(y == cls)[0]
    chosen = rng.choice(cls_indices, size=150, replace=False)
    seed_indices_per_class[cls] = chosen

# Seed-Indizes speichern
all_seed_indices = np.concatenate(list(seed_indices_per_class.values()))

# Holdout pro Klasse (20%) über die Seed-Indizes
holdout_indices = []
for cls, indices in seed_indices_per_class.items():
    holdout_size = int(0.2 * len(indices))  # 20%
    holdout = rng.choice(indices, size=holdout_size, replace=False)
    holdout_indices.extend(holdout)
holdout_indices = np.array(holdout_indices)

# Input für Evaluation
y_holdout_true = y[holdout_indices].copy()

# Trainings-Seeds ohne Holdout
train_seed_indices = np.setdiff1d(all_seed_indices, holdout_indices)

# Parameter-Grid für LabelPropagation
param_grid = {'kernel': ['rbf','knn'],
              'gamma': [0.1, 0.5, 1.0, 5.0, 10, 20, 30,50,100],
              'n_neighbors': [4,5,6,7,8]}

best_acc = 0
best_params = None
best_model = None

# Manuelle Grid Search
for kernel in param_grid['kernel']:
    for gamma in param_grid['gamma']:
        for n_neighbor in param_grid['n_neighbors']:

            # LabelPropagation-Labels vorbereiten
            y_lp = -1 * np.ones_like(y)
            y_lp[train_seed_indices] = y[train_seed_indices]

            # Modell initialisieren
            lp_model = LabelPropagation(kernel=kernel, gamma=gamma, n_neighbors = n_neighbor, max_iter=100000, tol=1e-4)
            lp_model.fit(X_scaled, y_lp)

            # Holdout Vorhersagen
            y_holdout_pred = lp_model.transduction_[holdout_indices]

            # Accuracy auf Holdout
            acc = (y_holdout_pred == y_holdout_true).mean()
            print(f"kernel={kernel}, gamma={gamma}, holdout acc={acc:.3f}")

            # Bestes Modell merken
            if acc > best_acc:
                best_acc = acc
                best_params = {'kernel': kernel, 'gamma': gamma}
                best_model = lp_model

# -----------------------------
# Ergebnisse für bestes Modell
# -----------------------------
print("\n=== Bestes Modell ===")
print("Beste Params:", best_params)
print("Beste Holdout Accuracy:", best_acc)
y_holdout_pred = best_model.transduction_[holdout_indices]
print(classification_report(y_holdout_true, y_holdout_pred))

# Evaluation Visualisierung
def plot_evaluation_labels(y_true, y_pred, title_suffix=""):
    label_mapping = {0: "Progressiv", 1: "Konservativ"}
    y_true_mapped = [label_mapping[y] for y in y_true]
    y_pred_mapped = [label_mapping[y] for y in y_pred]
    labels = ["Progressiv", "Konservativ"]

    # Confusion Matrix
    cm = confusion_matrix(y_true_mapped, y_pred_mapped, labels=labels)
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, ax=ax1)
    ax1.set_title(f"Confusion Matrix {title_suffix}")

    # Precision/Recall/F1
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_mapped, y_pred_mapped, labels=labels
    )
    x = np.arange(len(labels))
    width = 0.2

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    bars1 = ax2.bar(x - width, precision, width, label='Precision')
    bars2 = ax2.bar(x, recall, width, label='Recall')
    bars3 = ax2.bar(x + width, f1, width, label='F1-Score')

    # Werte auf den Balken anzeigen
    ax2.bar_label(bars1, fmt='%.2f', padding=3)
    ax2.bar_label(bars2, fmt='%.2f', padding=3)
    ax2.bar_label(bars3, fmt='%.2f', padding=3)

    # Achsen und Titel
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Score")
    ax2.set_title(f"Precision / Recall / F1 {title_suffix}")

    # Legende oben rechts
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

# Plot-Funktion ausführen
plot_evaluation_labels(y_holdout_true, y_holdout_pred, title_suffix="Eval. Holdout Seed Users (S-Bert/Node2Vec)")

# Predictions für alle User
y_pred_all = best_model.transduction_
# Wahrscheinlichkeiten für Klassenzugehörigkeiten der Knoten speichern
y_proba_all = best_model.predict_proba(X_scaled)
# Gib Wahrscheinlichkeit für die bestimmte Klasse aus ("Confidence") ->
# Sinnvoll für eventuelles Filtern nach extrem sicheren Klassifikationen, aber sonst nicht verwendet
confidence_all = np.max(y_proba_all, axis=1)

dataset['predicted_label'] = y_pred_all
dataset['confidence'] = confidence_all


# Export der Klssifikationen
export_df = dataset.copy()
export_df = export_df.drop(columns=['bert_embeddings'], errors='ignore')  # Optional: Embeddings weglassen
export_df.to_csv('predictions_bert_node2vec.csv', index=False)

print(f"CSV exportiert: {export_df.shape[0]} User, Spalten: {export_df.shape[1]}")
