from models.data.source import Eikon, load_fed_rates_from_excel
from portfolios.equity import EquityPortfolio
import os
from pathlib import Path

PATH_API_KEYS = r'C:\Users\serge\OneDrive\reuters\apikeys.csv'
PATH_DATA = fr'{Path(__file__).parents[2]}\models\data'


portfolio = EquityPortfolio(config_path='config_example.yaml')

fields_of_interest = {
    ''
}

params = {
    'rics': portfolio.asset_universe,
    # 'field': ['TR.PriceClose', 'Price Close'],
    'field': ['TR.TotalAssetsReported', 'Total Assets, Reported'],
    'date_field': ['TR.TotalAssetsReported.calcdate', 'Calc Date'],
    # 'load_path': os.path.join(PATH_DATA, r'csv\total_assets')
    'save_config': {'save': True, 'path': r'C:\Users\serge\IdeaProjects\portfolio_management\models\data\csv\total_assets' },
    'params': {
        'SDate':'2005-12-31',
        'EDate': '2024-08-29',
    }
}

api = Eikon(PATH_API_KEYS)
# data = api.load_timeseries(**params)
data = api.download_timeseries(**params, out=True)

# api.download_timeseries(**params, set_as_data_data_attribute=False)