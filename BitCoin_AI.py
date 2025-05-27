import requests
import pandas as pd
days = 89
url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
params = {
  "vs_currency" : "usd",
  "days" : days
}

response = requests.get(url,params=params)
data = response.json()
DF = pd.DataFrame(data)
print(DF)