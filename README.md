# Ideology Propagation – Political Stance Classification on German Social Media

## Overview

This project implements a semi-supervised machine learning pipeline to classify German Twitter users into political camps (*progressive* vs. *conservative*) based on their language usage and social network structure.

The approach combines:

* **S-BERT embeddings** (semantic text features)
* **Node2Vec embeddings** (network structure)
* **Label Propagation** (semi-supervised learning)

The goal is to analyze whether political ideology can be inferred from linguistic patterns, social structure, or a combination of both.

---

## Research Context

This project is based on my Bachelor's thesis:

**“Wortwahl als Weltanschauung: Politische Lagerklassifikation deutscher Social-Media-Kommentare mittels maschineller Lernverfahren”** 

The work focuses on low-resource scenarios with limited labeled data and explores how classical ML methods can be combined with modern NLP representations.

---

## Methodology

### 1. Data Processing

* Dataset: NRW22 Twitter dataset (German political discourse)
* Reconstruction via **hydration of tweet IDs**
* Aggregation on **user level**
* Text preprocessing:

  * removal of noise (URLs, emojis)
  * handling of mentions & hashtags
  * normalization (lowercasing)

→ Final dataset: **5647 users**

---

### 2. Label Generation (Semi-Supervised Setup)

* Manual annotation of **seed users**
* Heuristic filtering using:

  * hashtags
  * keywords
* Final labeled subset:

  * ~207 progressive users
  * ~158 conservative users

---

### 3. Feature Engineering

#### Semantic Features

* Model: `distiluse-base-multilingual-cased-v2`
* Input:

  * aggregated tweets
  * profile descriptions
* Weighted combination (α = 0.7 for profile text)

#### Network Features

* Retweet graph construction
* Embedding via **Node2Vec**
* Encodes user position & homophily in network

---

### 4. Models

Three model variants were evaluated:

1. **S-BERT Baseline**
2. **S-BERT + Node2Vec**
3. **S-BERT + Node2Vec + PCA**

All models use:

* **Label Propagation**
* RBF kernel (optimized via Grid Search)

---

## Results

| Model                   | Accuracy | Key Insight                                   |
| ----------------------- | -------- | --------------------------------------------- |
| S-BERT                  | 0.73     | Language alone is moderately predictive       |
| S-BERT + Node2Vec       | 0.83     | Strong performance boost via network features |
| S-BERT + Node2Vec + PCA | 0.85     | Slight improvement + better generalization    |

### Key Finding

> Network structure has a stronger influence on political classification than language alone.

This suggests that **social embedding (who interacts with whom)** is more informative than purely linguistic patterns.

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run pipeline

```bash
python main.py
```

### 3. Optional steps

* Run hydration script to reconstruct dataset
* Generate embeddings
* Train model & evaluate

---

## Project Structure

```
├── data/               # raw & processed datasets
├── preprocessing/      # data cleaning & aggregation
├── feature_engineering/
│   ├── sbert.py
│   ├── node2vec.py
├── models/
│   ├── label_propagation.py
├── evaluation/        # metrics & plots
├── main.py
└── README.md
```

---

## Tech Stack

* Python
* scikit-learn
* sentence-transformers (S-BERT)
* NetworkX
* Node2Vec
* pandas / numpy

---

## Limitations

* Dependence on manually annotated seed users
* Class imbalance and annotation bias
* Limited interpretability of embeddings (black-box issue)
* Partial dataset reconstruction (~66% hydration success)

---

## Future Work

* Replace Label Propagation with supervised models (e.g. SVM)
* Feature interpretability (e.g. SHAP, feature importance)
* Larger datasets or multilingual transfer
* Integration of LLM-based classification

---

## Author

Luca Steen

---
