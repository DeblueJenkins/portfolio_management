from portfolios.equity import EquityPortfolio
from models.data.source import Eikon
from models.data.handler import DataHandler
from pathlib import Path
import os
import pickle
from models.linear_programming import LinearFactorModel
from optimization.optimizer import Optimizer

# data_path = r'C:\Users\HP\Documents\GitHub\portfolio_management\models\data'
path_data = fr'{Path(__file__).parents[2]}\models\data'
# path_apikeys = r'C:\Users\HP\IdeaProjects\FinIgor\data\apikeys.csv'
path_apikeys = r'C:\Users\serge\OneDrive\reuters\apikeys.csv'


index = '.SPX'

get_constituents = False
download = False

eikon_api = Eikon(path_apikeys)


if get_constituents:

    rics_list = eikon_api.get_index_constituents()
    with open(fr'{path_data}\rics_of_{index.replace(".","")}.pkl', 'wb') as f:
        pickle.dump(rics_list, f)
        print('Pickled.')

else:
    rics_list = pickle.load(open(fr'{path_data}\rics_of_{index.replace(".","")}.pkl', 'rb'))
    print('Unpickled.')

params = {
    'rics': rics_list,
    'field': ['TR.PriceClose', 'Price Close'],
    'date_field': ['TR.PriceClose.calcdate', 'Calc Date'],
    'load_path': os.path.join(path_data, 'csv')
}

config = {
    'PCA': True,
    'n_components': 7,
    'OLS': True
}



eikon_api = Eikon(path_apikeys)
data = eikon_api.load_timeseries(**params)



## ALL OF THE FOLLOWING SHOULD BE WRAPPED IN ONE CLASS CALLED EXECUTOR

portfolio = EquityPortfolio(config_path='config_example.yaml')

preprocessor = DataHandler(data=data, date_col=params['date_field'][1])
returns = preprocessor.get_returns(period=15)

# for now just drop, otherwise we should interpolate or impute
returns.dropna(inplace=True, axis=1)


linear_model = LinearFactorModel(config_path='config_example.yaml', portfolio=portfolio, data=returns)

factors, factor_loadings = linear_model.fit()

optimizer = Optimizer(config_path='config_example.yaml', factors=factors, factor_loadings=factor_loadings, portfolio=portfolio)
optimizer.get_portfolio_return()