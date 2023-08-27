from portfolios.equity import EquityPortfolio
from models.data.source import Eikon
from models.data.handler import DataHandler
from pathlib import Path
import os
import pickle
from models.linear_programming import LinearFactorModel
from models.unsupervised.pca import PcaHandler, R2Pca
from optimization.optimizer import Optimizer
import numpy as np

## ALL OF THE FOLLOWING SHOULD BE WRAPPED IN ONE CLASS CALLED EXECUTOR

# data_path = r'C:\Users\HP\Documents\GitHub\portfolio_management\models\data'
path_data = fr'{Path(__file__).parents[2]}\models\data'
# path_apikeys = r'C:\Users\HP\IdeaProjects\FinIgor\data\apikeys.csv'
path_apikeys = r'C:\Users\serge\OneDrive\reuters\apikeys.csv'

eikon_api = Eikon(path_apikeys)
portfolio = EquityPortfolio(config_path='config_example.yaml')

params = {
    'rics': portfolio.assets,
    'field': ['TR.PriceClose', 'Price Close'],
    'date_field': ['TR.PriceClose.calcdate', 'Calc Date'],
    'load_path': os.path.join(path_data, 'csv')
}


eikon_api = Eikon(path_apikeys)
data = eikon_api.load_timeseries(**params)

preprocessor = DataHandler(data=data,
                           date_col=params['date_field'][1])

returns = preprocessor.get_returns(period=15)

# for now just drop, otherwise we should interpolate or impute or deal with it somehow
returns.dropna(inplace=True, axis=1)

# n_comps should come from config
factors = preprocessor.get_pca_factors(n_components=5,
                                       data=returns,
                                       method='ordinary')

linear_model = LinearFactorModel(config_path='config_example.yaml',
                                 portfolio=portfolio)

# this should be a ModelSelector class
# trials = linear_model.tune_hyperparams()

factors_t, factor_loadings_t = linear_model.fit(X=factors, y=returns)

optimizer = Optimizer(config_path='config_example.yaml',
                      factors=factors_t,
                      factor_loadings=factor_loadings_t,
                      portfolio=portfolio)
weights = optimizer.find_optimal_weights()
print(weights)