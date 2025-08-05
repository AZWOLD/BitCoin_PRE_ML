#======================Libraries=========================
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow.keras.backend as K
from keras.models import Sequential
from keras.layers import LSTM, Dense,Dropout
#==================loading LSTM Model====================
ML_model = Sequential()
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
def Creat_Seq(Prices,MV,MC,window_size):
  x,y = [],[]
  for i in range(len(Prices) - (window_size + Future_Steps)):
    x.append([[i,j,k] for i,j,k in zip(Prices[i:i+window_size],MV[i:i+window_size],MC[i:i+window_size])])
    y.append(Prices[i+window_size:i+window_size+Future_Steps].flatten())
  return np.array(x),np.array(y)
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
prices_TR = Scaler_Prices.fit_transform(prices[:split])
prices_TS = Scaler_Prices.transform(prices[split:])
MV_TR = Scaler_MV.fit_transform(market_volumes[:split])
MV_TS = Scaler_MV.transform(market_volumes[split:])
MC_TR = Scaler_MC.fit_transform(market_caps[:split])
MC_TS = Scaler_MC.transform(market_caps[split:])


X_train,y_train = Creat_Seq(prices_TR,MV_TR,MC_TR,window_size)
X_test,y_test = Creat_Seq(prices_TS,MV_TS,MC_TS,window_size)

Time_index = [i[0] for i in data["prices"]]
# --- model ---
ML_model = Sequential()
ML_model.add(LSTM(145, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
ML_model.add(Dropout(0.1))
ML_model.add(Dense(Future_Steps))  # Predict N steps ahead
ML_model.compile(optimizer='adam', loss=Custom_Loss(Scaler_Prices.mean_, Scaler_Prices.scale_))
ML_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.23)

# --- prediction ---
Prices_Pred = ML_model.predict(X_test)  # shape: (samples, Future_Steps)

# --- inverse transform predictions ---
Prices_Pred_real = Scaler_Prices.inverse_transform(Prices_Pred)
y_test_real = Scaler_Prices.inverse_transform(y_test)

# --- save predictions ---
columns = {}
for i in range(Future_Steps):
    columns[f"Real_{i+1}"] = y_test_real[:, i]
    columns[f"Pred_{i+1}"] = Prices_Pred_real[:, i]

Result = pd.DataFrame(columns)
Result.to_csv("Model Training Results.csv", index=False)


ML_model.save("BitCoinAI_12HF")