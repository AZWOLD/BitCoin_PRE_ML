#======================Libraries=========================
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras import backend as K
from keras.models import Sequential,load_model
from keras.layers import LSTM, Dense,Dropout
#==================loading LSTM Model====================
ML_model = load_model('BitcoinAI_12HF')
#===================Variables/Configurators======================
days = 89
window_size=72
Future_Steps = 12
url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
params = {
  "vs_currency" : "usd",
  "days" : days
}
Time_index = []
Scaler_Prices = StandardScaler()
Scaler_MV = StandardScaler()
Scaler_MC = StandardScaler()
def Custom_Loss(mean, scale):
  mean = K.constant(mean.reshape((1,)))
  scale = K.constant(scale.reshape((1,)))

  def Loss(y_True,y_pred):
    y_True_Real = y_True * scale + mean
    y_pred_Real = y_pred * scale + mean
    return K.mean(K.abs(y_True_Real - y_pred_Real))
  return Loss
#=============Fetching/Requesting DATA + Formating===============
response = requests.get(
  url,
  params=params
)
data = response.json()
#======================Parsing Prices/Market_caps/Market_volumes============================
prices = np.array([i[1] for i in data["prices"]]).reshape(-1,1)
market_volumes = np.array([i[1] for i in data["total_volumes"]]).reshape(-1,1)
market_caps = np.array([i[1] for i in data["market_caps"]]).reshape(-1,1)
split = int(len(prices) * 0.8)
Scaled_Prices = Scaler_Prices.fit_transform(prices)
Scaled_MV = Scaler_MV.fit_transform(market_volumes)
Scaled_MC = Scaler_MC.fit_transform(market_caps)
last_window = np.array([
    [p, mv, mc] for p, mv, mc in zip(
        Scaled_Prices[-window_size:],
        Scaled_MV[-window_size:],
        Scaled_MC[-window_size:]
    )
]).reshape(1, window_size, 3)
Pred = ML_model.predict(last_window)
Real_Pred = Scaler_Prices.inverse_transform(Pred)

Results = pd.DataFrame({
    f"Hour_{i+1}": [Real_Pred[0, i]] for i in range(Future_Steps)
})
Results.to_csv("Preditcions.csv",index=False)
