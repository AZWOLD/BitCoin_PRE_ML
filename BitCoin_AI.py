#======================Libraries=========================
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
#==================loading LSTM Model====================
ML_model = Sequential()
#===================Variables/Configurators======================
days = 89
window_size=72
url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
params = {
  "vs_currency" : "usd",
  "days" : days
}
Time_index = []
Scaler = StandardScaler()
def Creat_Seq(data,window_size):
  x,y = [],[]
  for i in range(len(data) - window_size):
    x.append(data[i:i+window_size])
    y.append(data[i+window_size])
  return np.array(x),np.array(y)

#=============Fetching/Requesting DATA + Formating===============
response = requests.get(
  url,
  params=params
)
data = response.json()
DF = pd.DataFrame(data)
#======================Parsing Prices============================
prices = np.array([i[1] for i in data["prices"]]).reshape(-1,1)
split = int(len(prices) * 0.8)
prices_TR = Scaler.fit_transform(prices[:split])
prices_TS = Scaler.transform(prices[split:])
X_train,y_train = Creat_Seq(prices_TR,window_size)
X_test,y_test = Creat_Seq(prices_TS,window_size)

Time_index = [i[0] for i in data["prices"]]
DF.index = Time_index
ML_model.add(LSTM(55,return_sequences=False,input_shape=(X_train.shape[1],X_train.shape[2])))
ML_model.add(Dense(1))
ML_model.compile(optimizer='adam',loss='mean_squared_error')
ML_model.fit(X_train,y_train,epochs=30,batch_size=32,validation_split=0.1)
np.set_printoptions(suppress=True, precision=4)
Prices_Pred = Scaler.inverse_transform(ML_model.predict(X_test))
y_test = Scaler.inverse_transform(y_test)
