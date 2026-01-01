# Stock Price Predictor (LSTM + Time Series)

This project trains a deep learning model (LSTM) to **predict a stock’s closing price** using the previous **60 days of closing prices** as input. It downloads historical data via `yfinance`, trains an LSTM model, evaluates predictions on a recent time range, and plots **Actual vs Predicted** prices.

> ⚠️ Educational project only — not financial advice.

## What this project does

- Downloads historical stock price data using Yahoo Finance (`yfinance`) :contentReference[oaicite:1]{index=1}  
- Scales closing prices using `MinMaxScaler` :contentReference[oaicite:2]{index=2}  
- Builds training sequences using a **60-day rolling window** :contentReference[oaicite:3]{index=3}  
- Trains a **3-layer LSTM network** with dropout regularization :contentReference[oaicite:4]{index=4}  
- Predicts closing prices on a “test” time period and plots results :contentReference[oaicite:5]{index=5}  

## How it works (pipeline)

1. **Choose a stock ticker**
   - Default: `META` :contentReference[oaicite:6]{index=6}  

2. **Train data**
   - Trains on: `2015-01-01 → 2023-01-01` :contentReference[oaicite:7]{index=7}  

3. **Create sequences**
   - `prediction_days = 60` (use the last 60 closes to predict the next close) :contentReference[oaicite:8]{index=8}  

4. **Model architecture**
   - LSTM(50) → Dropout(0.2)
   - LSTM(50) → Dropout(0.2)
   - LSTM(50) → Dropout(0.2)
   - Dense(1) :contentReference[oaicite:9]{index=9}  

5. **Test data**
   - Tests on: `2023-01-01 → today` :contentReference[oaicite:10]{index=10}  

6. **Visualization**
   - Plots actual vs predicted closing price :contentReference[oaicite:11]{index=11}  

## Repo structure (suggested)

```bash:
stock_price_predictor/
├─ stock_predictor.py
├─ README.md
└─ requirements.txt
```

## Setup

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
pip install -r floyd_requirements.txt
```

### Run

```bash:
python stock_predictor.py
```

A plot window will appear showing:

- **Actual closing prices** (black)
- **Predicted closing prices** (green)

## Customize (change ticker / time ranges)

In ```stock_predictor.py```:

- Change the ticker:
  - ```company = 'META'```
- Adjust training dates:
  - ```start = dt.datetime(2015, 1, 1)```
  - ```end = dt.datetime(2023, 1, 1)```
- Adjust the lookback window:
  - ```prediction_days = 60```
- Training parameters:
  - ```epochs=25```, ```batch_size=32```
