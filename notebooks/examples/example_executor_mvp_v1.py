from portfolios.equity import EquityPortfolio
from models.data.source import Eikon
from models.data.handler import DataHandler
from backtesting import PerformanceAssesser
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

START_DATE = '1999-12-31'
END_DATE = '2024-04-21'

## ALL OF THE FOLLOWING SHOULD BE WRAPPED IN ONE CLASS CALLED EXECUTOR

# data_path = r'C:\Users\HP\Documents\GitHub\portfolio_management\models\data'
path_data = fr'{Path(__file__).parents[2]}\models\data'
# path_apikeys = r'C:\Users\HP\IdeaProjects\FinIgor\data\apikeys.csv'
path_apikeys = r'C:\Users\serge\OneDrive\reuters\apikeys.csv'

eikon_api = Eikon(path_apikeys)
portfolio = EquityPortfolio(config_path='config_example_2.yaml')

# params = {
#     'rics': portfolio.assets,
#     'field': ['TR.PriceClose', 'Price Close'],
#     'date_field': ['TR.PriceClose.calcdate', 'Calc Date'],
#     'load_path': os.path.join(path_data, 'csv')
# }

already_downloaded = [x[:-4] for x in os.listdir(os.path.join(path_data, 'csv'))]
rics = [x for x in portfolio.assets if x not in already_downloaded]

params = {
    'rics': rics,
    'field': ['TR.PriceClose', 'Price Close'],
    'date_field': ['TR.PriceClose.calcdate', 'Calc Date'],
    'params': {
        'SDate': START_DATE,
        'EDate': END_DATE,
        'Curn':'Native',
    },
    'save_config': {'save': True, 'path': os.path.join(path_data, 'csv')}
}

eikon_api = Eikon(path_apikeys)
# data = eikon_api.load_timeseries(**params)
data = eikon_api.download_timeseries(**params)

preprocessor = DataHandler(data=data,
                           date_col=params['date_field'][1])

returns = preprocessor.get_returns(period=14)

portfolio.set_returns(returns)

# for now just drop, otherwise we should interpolate or impute or deal with it somehow
before_drop = returns.columns
returns.dropna(inplace=True, axis=1)
after_drop = returns.columns
remove_from_portfolio = list(set(before_drop).difference(set(after_drop)))



# n_comps should be a hyperparam
n_comps = portfolio.config['MODEL']['main_factors']['PCA']
factors, eigen_values = preprocessor.get_pca_factors(n_components=n_comps,
                                       data=returns,
                                       method='ordinary')


# since this takes portfolio, it can have the returns and factors in self
linear_model = LinearFactorModel(config_path='config_example.yaml',
                                 portfolio=portfolio)

linear_model.fit(X=factors, y=returns)
linear_model.set_factor_var_cov(eigen_values)

# this should be a ModelSelector class
# trials = linear_model.tune_hyperparams()
# sns.displot(linear_model.multi_regressor.residuals.iloc[:,5])
# plt.show()


# equal_portfolio_returns = returns.to_numpy().dot(portfolio.weights)
# equal_mean_return = np.mean(equal_portfolio_returns)
# equal_total_variance = np.var(equal_portfolio_returns)
#
# sample_mean = lambda weights: np.mean(returns.to_numpy().dot(weights))
# sample_variance = lambda weights: np.var(returns.to_numpy().dot(weights))

linear_model.get_portfolio_total_variance(portfolio.weights)
linear_model.get_portfolio_factor_return(portfolio.weights)

mse = linear_model.estimator.mse
r2 = linear_model.estimator.r2
r2_adj = linear_model.estimator.r2_adj


# r2.plot.bar(figsize=(11,8))
# r2_adj.plot.bar(figsize=(11,8))
# plt.show()

# if model goes inside, the functions are not needed to be input
# also, model.factors and model.factor_loadings are not needed, only size is needed
# which can be extracted from portfolio
optimizer = Optimizer(config_path='config_example.yaml',
                      model=linear_model,
                      portfolio=portfolio,
                      mean_function=linear_model.get_portfolio_factor_return,
                      var_function=linear_model.get_portfolio_total_variance)

weights = optimizer.find_optimal_weights()
print(weights)

# tester = PerformanceAssesser(START_DATE, END_DATE, weights)
# tester.get_historical_returns(returns)
# tester.get_benchmark_index_returns(eikon_api, '.SP500', 'TR.PriceClose', 'TR.PriceClose.calcdate')