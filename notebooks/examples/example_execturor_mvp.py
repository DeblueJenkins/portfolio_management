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
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

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

portfolio.set_returns(returns)

# for now just drop, otherwise we should interpolate or impute or deal with it somehow
returns.dropna(inplace=True, axis=1)

# n_comps should come from config
# n_comps should be a hyperparam
factors, eigen_values = preprocessor.get_pca_factors(n_components=15,
                                       data=returns,
                                       method='ordinary')


linear_model = LinearFactorModel(config_path='config_example.yaml',
                                 portfolio=portfolio)

linear_model.fit(X=factors, y=returns)
linear_model.set_factor_var_covar(eigen_values)

# this should be a ModelSelector class
# trials = linear_model.tune_hyperparams()
# sns.displot(linear_model.multi_regressor.residuals.iloc[:,5])
# plt.show()


equal_portfolio_returns = returns.to_numpy().dot(portfolio.weights)
equal_mean_return = np.mean(equal_portfolio_returns)
equal_total_variance = np.var(equal_portfolio_returns)

sample_mean = lambda weights: np.mean(returns.to_numpy().dot(weights))
sample_variance = lambda weights: np.var(returns.to_numpy().dot(weights))

linear_model.get_portfolio_total_variance(portfolio.weights)
linear_model.get_portfolio_factor_return(portfolio.weights)

mse = linear_model.multi_regressor.mse
r2 = linear_model.multi_regressor.r2
r2_adj = linear_model.multi_regressor.r2_adj


r2.plot.bar(figsize=(11,8))
r2_adj.plot.bar(figsize=(11,8))
plt.show()

optimizer = Optimizer(config_path='config_example.yaml',
                      model=linear_model,
                      portfolio=portfolio,
                      mean_function=linear_model.get_portfolio_factor_return,
                      var_function=linear_model.get_portfolio_total_variance)

weights = optimizer.find_optimal_weights()
print(weights)