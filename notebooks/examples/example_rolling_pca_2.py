
from models.unsupervised.pca import R2Pca
from models.data.source import Eikon
from models.data.handler import DataHandler
import pickle
import os
from models.stat_models.linearregression import MultiOutputLinearRegressionModel
import numpy as np

data_path = r'C:\Users\HP\Documents\GitHub\portfolio_management\models\data'
path_apikeys = r'C:\Users\HP\IdeaProjects\FinIgor\data\apikeys.csv'
eikon_api = Eikon(path_apikeys)

index = '.SPX'

get_constituents = False
download = False

univariate = True

if get_constituents:
    rics_list = eikon_api.get_index_constituents()
    with open(fr'{data_path}\rics_of_{index.replace(".","")}.pkl', 'wb') as f:
        pickle.dump(rics_list, f)
        print('Pickled.')

else:
    rics_list = pickle.load(open(fr'{data_path}\rics_of_{index.replace(".","")}.pkl', 'rb'))
    print('Unpickled.')


# rics_list = [
#     'LVMH.PA', 'ASML.AS', 'NESN.S', 'LIN',
#     'NOVOb.CO', 'AAPL.O', 'ROG.S', 'UNH',
#     'SAPG.DE', 'MSFT.O'
# ]

params = {
    'rics': rics_list,
    'field': ['TR.PriceClose', 'Price Close'],
    'date_field': ['TR.PriceClose.calcdate', 'Calc Date'],
    'load_path': os.path.join(data_path, 'csv')
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
np.array(X)

rolling = R2Pca(data=X, rolling_window=150)
a = rolling.components(n=3)


line_model = MultiOutputLinearRegressionModel(x=rolling.components, y=rolling.raw_data[(150-1):, :],
                                              tickers=list(rolling.assets))

line_model.plot_ticker(list(rolling.assets)[20])

