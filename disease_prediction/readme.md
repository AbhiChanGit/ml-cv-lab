# Disease Predictor (Machine Learning)

A supervised machine learning project that predicts the **most likely disease** based on a patient’s reported **symptoms**. The model is trained on labeled medical symptom data and demonstrates a complete ML workflow from data loading to model evaluation.

---

## What this project does

This project builds a **multi-class disease classification system** where:

- **Input:** A set of symptoms (binary features indicating presence/absence)
- **Output:** A predicted disease label

The notebook compares multiple machine learning models and evaluates their performance, illustrating how different classifiers behave on the same medical dataset.

---

## Dataset

Files:

- `Training.csv`
- `Testing.csv`

### Dataset characteristics

- Each row represents a **patient case**
- Columns represent **symptoms** (binary: `0` = absent, `1` = present)
- The target column is **`prognosis`**, which contains the disease label

Example:

| Fever | Headache | Nausea | ... | prognosis |
|-------|----------|--------|-----|-----------|
| 1     | 0        | 1      | ... | Dengue    |

---

## Approach / Pipeline

This is a **multi-class classification problem** with many possible disease outcomes.

Implemented in `disease_prediction.ipynb`:

1. **Load datasets**
   - Reads `Training.csv` and `Testing.csv` into pandas DataFrames.
2. **Feature / label separation**
   - `X`: symptom columns
   - `y`: disease (`prognosis`)
3. **Model training**
   - Trains multiple classifiers, including:
     - Decision Tree
     - Random Forest
     - Naive Bayes
     - K-Nearest Neighbors
     - Support Vector Machine
4. **Evaluation**
   - Compares model accuracy on the test dataset
   - Identifies the best-performing model
5. **Prediction**
   - Uses the trained model to predict disease outcomes based on symptoms

---

## Models Used

The notebook demonstrates and compares several ML algorithms:

- **Decision Tree** – interpretable and fast baseline
- **Random Forest** – ensemble model for better generalization
- **Naive Bayes** – probabilistic approach for binary features
- **KNN** – instance-based learning
- **SVM** – margin-based classifier

This comparison highlights trade-offs between accuracy, interpretability, and computational cost.

---

## Repository structure (suggested)

```bash:
disease_predictor/
├─ disease_prediction.ipynb
├─ Training.csv
├─ Testing.csv
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

## Implementation notes

- **Binary symptom encoding**: Each symptom is represented as a binary feature, which simplifies preprocessing.
- **Multi-class classification**: The model predicts one disease out of many possible conditions.
- **Accuracy-focused evaluation**: This notebook primarily reports accuracy; in real medical applications, additional metrics (precision, recall per class) are crucial.
- **Educational focus**: This project is intended for learning and experimentation—not for real medical diagnosis.
