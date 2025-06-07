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
window_size=60
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
prices = Scaler.fit_transform(np.array([i[1] for i in data["prices"]]).reshape(-1,1))
X,y = Creat_Seq(prices,window_size)
split = int(len(X) * 0.8)
X_train,X_test = X[:split],X[split:]
y_train,y_test = y[:split],y[split:]

Time_index = [i[0] for i in data["prices"]]
DF.index = Time_index
ML_model.add(LSTM(50,return_sequences=False,input_shape=(X_train.shape[1],X_train.shape[2])))
ML_model.add(Dense(1))
ML_model.compile(optimizer='adam',loss='mean_squared_error')
ML_model.fit(X_train,y_train,epochs=30,batch_size=32,validation_split=0.1)

Prices_Pred = ML_model.predict(X_test)
print(Prices_Pred)