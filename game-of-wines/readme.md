# Game of Wines 🍷 — Wine Quality Classification with ML

This project uses supervised machine learning to **predict wine quality** from **physicochemical lab measurements** (acidity, sugar, sulphates, alcohol, etc.). The original target is a **0–10 quality score**, but in this project it’s **converted into a 3-class classification problem**:

- **Very good**: quality **≥ 7**
- **Average**: quality **5–6**
- **Insipid**: quality **< 5**

The main work is implemented in the notebook (`completed-code_game-of-wines.ipynb`), and the repo also contains HTML exports for easy viewing (`game-of-wines.html`, `game-of-wines-report.html`).

## Dataset

This repo includes the well-known **Wine Quality** dataset (red + white variants of Portuguese *Vinho Verde* wines). Each row is a wine sample with:

- **11 input features** from physicochemical tests (fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, sulfur dioxide measures, density, pH, sulphates, alcohol)
- **1 output label**: expert-rated wine quality score (0–10) :contentReference[oaicite:1]{index=1}

Dataset sizes:

- **Red wine**: 1599 samples
- **White wine**: 4898 samples

> Note: Classes are **ordered and imbalanced**, with many more “normal/average” wines than extremes.

## Approach

### Data preparation

- Load the dataset (red wine is the default in the notebook; white is included as an extension idea).
- Convert the regression target (0–10) into the 3 quality buckets above. :contentReference[oaicite:4]{index=4}

### Train & evaluate multiple models

The notebook compares multiple supervised learning models and visualizes:

- training time vs prediction time
- accuracy vs F-score performance across different training set sizes :contentReference[oaicite:5]{index=5}

Evaluation uses **Accuracy** and an **F-score** variant (the notebook uses an F-beta setup with `beta = 0.5` and `average="micro"`). :contentReference[oaicite:6]{index=6}

### Optimization + feature importance

A **RandomForestClassifier** is tuned and used as the final model, and feature importance plots are supported via `visuals.py`. :contentReference[oaicite:7]{index=7}

## Results (from the included report)

The report shows an example final tuned model:

- **Unoptimized**: accuracy **0.8906**, F-score **0.8906**
- **Optimized RandomForestClassifier** (e.g., `n_estimators=30`, `max_features=3`): final accuracy **0.8969**, final F-score **0.8969** :contentReference[oaicite:8]{index=8}

## Repository structure

```bash:
game-of-wines/
├─ completed-code_game-of-wines.ipynb
├─ floyd_requirements.txt
├─ game-of-wines-report.html
├─ game-of-wines.html
├─ visuals.py
└─ data/
├─ winequality-red.csv
├─ winequality-white.csv
└─ winequality.names.txt
```

---

## Setup

### Clone the repository

```bash
git clone https://github.com/AbhiChanGit/ml-cv-lab.git
cd credit_fraud_detector
```

### Create and activate a virtual environment (recommended)

```bash:
python -m venv .venv
. .venv\Scripts\Activate.ps1
```

**macOS/Linux:**

```bash:
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash:
pip install -r floyd_requirements.txt
```

### Option A — Run the notebook

Open: ```completed-code_game-of-wines.ipynb```

### Option B — View the HTML exports

Open either in your browser:

- ```game-of-wines.html```
- ```game-of-wines-report.html```

## Making predictions

The notebook includes a “try it yourself” section where you provide inputs in this order:

```fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol```

It then outputs a predicted quality class (e.g., 1/2/3 depending on the bin encoding used in the notebook)
