import requests
import pandas as pd

API_KEY = "HE4GW6VF8BJM3MB0"
URL = f'https://www.alphavantage.co/query'

# EARNINGS
params = {
    'function': 'EARNINGS',
    'symbol': 'IBM',
    'apikey': f'{API_KEY}'
}

r = requests.get(URL, params=params)
df_quarterly_earnings = pd.DataFrame(data=r.json()['quarterlyEarnings'])

# OVERVIEW
params = {
    'function': 'OVERVIEW',
    'symbol': 'IBM',
    'apikey': f'{API_KEY}'
}

r = requests.get(URL, params=params)

df_overview = pd.Series(r.json())

# SENTIMENT

params = {
    'function': 'NEWS_SENTIMENT',
    'symbol': 'AAPL',
    'apikey': f'{API_KEY}'
}

r = requests.get(URL, params=params)

