# 💰 Bitcoin Price Forecasting AI

An LSTM-based deep learning model that predicts short-term Bitcoin market movements using the last 89 days of historical data.

> Data includes:  
> - 📈 Prices  
> - 💸 Total Volumes  
> - 🏦 Market Capitalizations  
>  
> All data is sourced from [CoinGecko](https://www.coingecko.com/) 🦎

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/AZWOLD/BitCoin_PRE_AI
```
### 2. Navigate to the folder:
Find the Bitcoin_PRE_AI folder and open a command prompt inside it

### 3. Install dependencies:
Ensure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

---

## 🤖 Project Overview
This project includes two main Python scripts:

### 🔮 BitCoinAI_Predict.py — Prediction Script
- Uses the last 72 hours of Bitcoin data 
- Predicts the market behavior for the next 12 hours
- Output is saved in: `Predictions.csv`

**Run with:**
```bash
python BitCoinAI_Predict.py
```
The Predictions are Saved into `Predictions.csv`

### 🏋️ BitCoinAI_Train.py — Training Script
- Retrains the LSTM model on the most recent 89 days of data
- Improves accuracy with updated market conditions
- Results are saved to: `Model Training Results.csv`
```bash
python BitCoinAI_Train.py
```
> 💡 Tip: It's recommended to retrain the model every 10–15 days for best results.

---

## 🙏 Credit:
Created with 💻 by [AZWOLD](https://github.com/AZWOLD)

Bitcoin data powered by [CoinGecko](https://www.coingecko.com/)🦎
