# ml-cv-lab

![Python](https://img.shields.io/badge/Python-ML%20%7C%20Data%20Science-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Models%20%26%20Pipelines-f7931e)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-TensorFlow%20%2F%20Keras-ff6f00)

A personal portfolio of **data science + machine learning projects** (with some CV/Deep Learning work mixed in), built to practice end-to-end workflows: **data prep → modeling → evaluation → visualization/reporting**.

Each project lives in its own folder with its own notebook/script, dependencies, and dataset(s).

## Projects

| Project | Type | What it does | Key tools |
|---|---|---|---|
| `credit_fraud_detector/` | Classification | Detects fraudulent credit card transactions (imbalanced ML) | pandas, scikit-learn, seaborn |
| `disease_prediction/` | Multi-class classification | Predicts disease from symptom inputs | pandas, scikit-learn |
| `game-of-wines/` | Classification + Reporting | Predicts wine quality class + exports reports (HTML) | pandas, scikit-learn, matplotlib |
| `stock_predictor/` | Time series (Deep Learning) | Predicts stock closing prices using LSTM | yfinance, TensorFlow/Keras |

## Repo layout

```text
ml-cv-lab/
├─ credit_fraud_detector/
├─ disease_prediction/
├─ game-of-wines/
└─ stock_predictor/
```

Each project folder typically includes:

- ```readme.md``` — project overview + run instructions
- ```requirements.txt``` (or similar) — dependencies
- ```data/``` or ```dataset/``` — datasets (when included)
- ```*.ipynb``` / ```*.py``` — the main notebook or script

Quick start

- **Choose a project folder** (example: credit_fraud_detector/)
- Open that project’s ```readme.md```
- Install its dependencies: ```pip install -r requirements.txt```  
- Run the notebook/script as instructed in that project.

> Some projects include datasets in the repo for convenience. For larger datasets, a common best practice is keeping them out of Git history and documenting how to download/place them.
