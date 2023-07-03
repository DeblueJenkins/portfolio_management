import requests

API_KEY = 'OjE2MmZhNGQ5ZTNjYWZjYTU0OGViNjllYmEwMjE1OTZk'

ticker = 'AAPL'

URL = f'https://api-v2.intrinio.com/companies'


params = {
    'identifier': ticker,
    'fundamentals': {
        ''
    }
    'function': 'EARNINGS',
    'symbol': 'IBM',
    'apikey': f'{API_KEY}'
}

r = requests.get(URL, params=params)