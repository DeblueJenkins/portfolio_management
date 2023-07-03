from models.rolling import RollingModel
from models.data.source import Eikon
import numpy as np
from models.data.handler import DataHandler
import pickle

data_path = r'C:\Users\serge\IdeaProjects\portfolio_manager\data\fama-french-factors'
path_apikeys = r'C:\Users\serge\OneDrive\Documents\apikeys.csv'

index = '.SPX'
rics_list = pickle.load(open(fr'{data_path}\rics_of_{index.replace(".","")}.pkl', 'rb'))
print('Unpickled.')

rics_list = rics_list[:20]

# rics_list = [
#     'LVMH.PA', 'ASML.AS', 'NESN.S', 'LIN',
#     'NOVOb.CO', 'AAPL.O', 'ROG.S', 'UNH',
#     'SAPG.DE', 'MSFT.O'
# ]

params = {
    'rics': rics_list,
    'field': ['TR.PriceClose', 'Price Close'],
    'date_field': ['TR.PriceClose.calcdate', 'Calc Date'],
    'load_path': r'C:\Users\serge\IdeaProjects\portfolio_manager\portfolio_management\models\data\csv'
}

config = {
    'PCA': True,
    'n_components': 7,
    'OLS': True
}
eikon_api = Eikon(path_apikeys)
data = eikon_api.load_timeseries(**params)

preprocessor = DataHandler(data=data, date_col=params['date_field'][1])

X = preprocessor.get_returns(period=15)
X.dropna(axis=1, inplace=True)

rolling = RollingModel(rolling_window=200, data=X, demean=True)
rolling.estimate(config=config, RIC=rics_list[2])
rolling.plot_resids()