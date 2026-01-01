# Credit Fraud Detector (ML + Data Science)

A machine learning project that detects **fraudulent credit card transactions** using a supervised classifier trained on historical transaction data. The workflow is implemented in a Jupyter notebook and includes basic exploratory analysis, train/test splitting, model training, and evaluation with multiple metrics + a confusion matrix visualization.

---

## What this project does

This project trains a **Random Forest classifier** to predict whether a transaction is:

- `0` → **Normal**
- `1` → **Fraud**

It also highlights a key real-world challenge in fraud detection: **extreme class imbalance** (fraud cases are rare), so metrics like **precision/recall/F1** matter more than accuracy.

---

## Dataset

File: `creditcard.csv`

- Rows: **284,807** transactions
- Fraud cases: **492** (~**0.173%**)
- Columns: **31**
  - `Time` (seconds elapsed between each transaction and the first transaction)
  - `V1` … `V28` (anonymized features, commonly PCA-transformed)
  - `Amount` (transaction amount)
  - `Class` (label: 0 normal, 1 fraud)

> Note: Because the dataset is highly imbalanced, a model can achieve high accuracy by predicting “normal” most of the time. That’s why this project reports multiple metrics.

---

## Approach / Pipeline

Implemented in `fraud_detector.ipynb`:

1. **Load data**
   - Reads `creditcard.csv` into a pandas DataFrame.
2. **Explore imbalance**
   - Splits the dataset into fraud vs. valid transactions and prints their counts.
3. **Basic EDA**
   - Compares `Amount` statistics for fraud vs. valid transactions.
   - Plots a **correlation heatmap**.
4. **Train/Test split**
   - Uses an 80/20 split (`random_state=42`).
5. **Model training**
   - Trains a `RandomForestClassifier()` from scikit-learn (default settings).
6. **Evaluation**
   - Accuracy
   - Precision
   - Recall
   - F1 score
   - Matthews Correlation Coefficient (MCC)
7. **Visualization**
   - Confusion matrix heatmap (Normal vs Fraud)

---

## Repository structure

```bash
credit_fraud_detector/
├─ fraud_detector.ipynb
├─ creditcard.csv
├─ README.md
└─ requirements.txt
```

## How to run

### Clone the repository

```bash
git clone https://github.com/AbhiChanGit/ml-cv-lab.git
cd credit_fraud_detector
```

### Create and activate a conda environment (recommended)

```bash
conda create -n 'env name' python=3.11
conda activate 'env name'
```

### Install dependencies

```bash:
pip install -r requirements.txt
```

### Launch Jupyter Notebook

## Implementation notes (important)

- **Imbalanced data**: Fraud is extremely rare. In practice, you should prioritize:
  - **Recall** (catch as many fraud cases as possible)
  - **Precision** (avoid too many false alarms)
  - **F1 / MCC** for balanced evaluation

- **Baseline model**: The Random Forest here is a solid baseline, but it uses default parameters. For better performance, consider:
  - `class_weight="balanced"` or cost-sensitive learning
  - resampling methods (undersampling, oversampling, SMOTE)
  - threshold tuning using predicted probabilities
  - evaluating ROC-AUC and especially PR-AUC (often more informative for imbalanced tasks)
