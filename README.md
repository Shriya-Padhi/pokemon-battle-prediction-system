# Pokémon Battle Prediction System

> A machine learning system that predicts the winner of a Pokémon battle from base stats and type matchups, achieving **96.2% accuracy** with a Random Forest classifier, alongside exploratory K-Means clustering and full statistical analysis of the Pokémon dataset.

[![Language](https://img.shields.io/badge/Python-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![ML](https://img.shields.io/badge/scikit--learn_[RandomForest%20%7C%20KMeans]-orange?style=flat-square)](https://scikit-learn.org/stable/user_guide.html)

---

## Overview

Given two Pokémon, can a model predict which one wins a battle without simulating the actual combat? This project builds a supervised classifier that learns from a dataset of 800 Pokémon and 50,000 historical battle outcomes, using stat differentials and type-effectiveness matchups as features.

Beyond the prediction task, the project includes a full exploratory data analysis of the Pokémon stat distribution (strongest/weakest by generation, legendary vs. non-legendary breakdown, type frequency) and an unsupervised K-Means clustering analysis to identify natural groupings of Pokémon by combat stats.

---

## Dataset

| File | Description |
| :--- | :--- |
| `pokemon.csv` | 800 Pokémon with stats (HP, Attack, Defense, Sp. Atk, Sp. Def, Speed), type(s), generation and legendary status |
| `combats.csv` | ~50,000 historical 1v1 battle outcomes (winner ID) used as training labels |
| `tests.csv` | Held-out matchups for final prediction |

Source: [Pokémon with Stats - Kaggle](https://www.kaggle.com/abcsds/pokemon)

---

## Methodology

### 1. Data Preprocessing

- Filled missing `Type 2` values with `'None'` (most Pokémon are single-typed)
- Normalized inconsistent type labeling (`Fighting` → `Fight`)
- Converted `Legendary` boolean to binary (0/1) for modeling
- Encoded battle winners as 0 (first Pokémon) / 1 (second Pokémon)

### 2. Feature Engineering

The core modeling decision was framing each battle as a **stat differential problem** rather than feeding in raw stats for both Pokémon independently:

- **Stat differences**: `HP_diff`, `Attack_diff`, `Defense_diff`, `Sp.Atk_diff`, `Sp.Def_diff`, `Speed_diff`, `Legendary_diff` computed as (Pokémon 1 stats, Pokémon 2 stats)
- **Type effectiveness**: four engineered columns (`P1_type1`, `P1_type2`, `P2_type1`, `P2_type2`) scoring each Pokémon's type advantage against the opponent, using the standard type chart (2× for super-effective, 0.5× for not-very-effective, 0× for no-effect matchups)

This compresses the raw 12+ stat columns per Pokémon pair into a feature set that directly encodes "who has the advantage," which proved highly predictive.

### 3. Modeling

Two classifiers were trained and compared on the engineered feature set:

| Model | Accuracy |
| :--- | :---: |
| Decision Tree | 94.2% |
| **Random Forest** | **96.2%** |

The Random Forest model (100 estimators) was selected as the final predictor and used to generate predictions on the held-out test set.

### 4. Exploratory Clustering (K-Means)

A separate unsupervised analysis applied K-Means clustering on HP vs. Attack stats to identify natural Pokémon archetypes:

- Used the **elbow method** (WCSS across k = 1 to 12) to select the optimal cluster count
- Settled on **k = 5** clusters, visualized as distinct groupings in HP-Attack space

This was exploratory rather than predictive, aimed at understanding whether Pokémon naturally group into archetypes (e.g., glass cannons, tanks, balanced fighters) independent of the battle outcome data.

### 5. Exploratory Data Analysis

Additional analysis covered:

| Finding | Detail |
| :--- | :--- |
| Strongest/weakest by total stats | Mega Rayquaza and Mega Mewtwo X/Y top the list; Sunkern and Azurill are weakest |
| Legendary distribution | Only 9.89% of Pokémon are legendary |
| Generation strength | Generation 1 has the highest average total stats; Generation 6 the lowest |
| Type frequency | Flying is the most common secondary type; type distribution is highly imbalanced |

---

## Repository Structure

```
Pokemon-battle-prediction-system/
│
├── Final code.ipynb              # Main pipeline: preprocessing, feature engineering, RF/DT models, prediction
├── POKEMON CLUSTER.ipynb         # K-Means clustering analysis (elbow method, HP vs Attack)
├── Project_Report.pdf            # Slide-style project report with methodology and visualizations
│
├── pokemon.csv                    # Pokémon stats and metadata
├── combats.csv                    # Historical battle outcomes (training labels)
└── tests.csv                      # Held-out battle matchups for prediction
```

---

## Installation

```bash
git clone https://github.com/<your-username>/pokemon-battle-prediction-system.git
cd pokemon-battle-prediction-system
pip install pandas numpy scikit-learn seaborn matplotlib
jupyter notebook "Final code.ipynb"
```

---

## Team

Originally developed as a 4-person group project for **CS322: Data Warehousing and Data Mining Course, MITWPU** .

| Name |
| :--- |
| Siddhant Deshpande |
| Shriya Padhi |
| Arnav Sinha |
| Atharv Ganla |

---

## References

- [Pokémon with Stats Dataset - Kaggle](https://www.kaggle.com/abcsds/pokemon)
