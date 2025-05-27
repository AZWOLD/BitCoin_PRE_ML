#Libraries:
import requests
import pandas as pd
#=============Variable/Configurators=================
days = 89
url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
params = {
  "vs_currency" : "usd",
  "days" : days
}
#=============Fetching/Requesting DATA + Formating===============
response = requests.get(url,params=params)
data = response.json()
DF = pd.DataFrame(data)
print(DF)