from models.data.source import Eikon

data_path = r'C:\Users\serge\IdeaProjects\portfolio_manager\data\fama-french-factors'
path_apikeys = r'C:\Users\serge\OneDrive\Documents\apikeys.csv'
# rics_list = [
#     'LVMH.PA', 'ASML.AS', 'NESN.S', 'LIN',
#     'NOVOb.CO', 'AAPL.O', 'ROG.S', 'UNH',
#     'SAPG.DE', 'MSFT.O'
# ]

rics_list = [
    'LIN',
    'NOVOb.CO'
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

if download:
    eikon_api.download_timeseries(**params, save_config={'save': True, 'path': r'C:\Users\serge\IdeaProjects\portfolio_manager\portfolio_management\models\data\csv'})
    print(eikon_api.data)
    print(eikon_api.data.columns)
else:

    params = {
        'rics': rics_list,
        'load_path': r'C:\Users\serge\IdeaProjects\portfolio_manager\portfolio_management\models\data\csv',
        'field': ['TR.PriceClose', 'Price Close'],
        'date_field': ['TR.PriceClose.calcdate', 'Calc Date'],
    }

    eikon_api.load_timeseries(**params)
    print(eikon_api.data)
    print(eikon_api.data.columns)