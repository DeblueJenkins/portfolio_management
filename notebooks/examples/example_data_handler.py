from models.data.source import Eikon
from models.data.handler import DataHandler

data_path = r'C:\Users\serge\IdeaProjects\portfolio_manager\data\fama-french-factors'
path_apikeys = r'C:\Users\serge\OneDrive\Documents\apikeys.csv'
rics_list = [
    'LVMH.PA', 'ASML.AS', 'NESN.S', 'LIN',
    'NOVOb.CO', 'AAPL.O', 'ROG.S', 'UNH',
    'SAPG.DE', 'MSFT.O'
]

params = {
    'rics': rics_list,
    'field': ['TR.PriceClose', 'Price Close'],
    'date_field': ['TR.PriceClose.calcdate', 'Calc Date'],
    'params': {
        'SDate':'1999-12-31',
        'EDate': '2021-06-04',
        'Curn':'Native',
    }
}

download = False

eikon_api = Eikon(path_apikeys)

params = {
    'rics': rics_list,
    'load_path': r'C:\Users\serge\IdeaProjects\portfolio_manager\portfolio_management\models\data\csv',
    'field': ['TR.PriceClose', 'Price Close'],
    'date_field': ['TR.PriceClose.calcdate', 'Calc Date'],
}

data = eikon_api.load_timeseries(**params)
preprocessor = DataHandler(data=data, date_col=params['date_field'][1])
rets = preprocessor.get_returns(period=1)
print(rets)