from portfolios.equity import EquityPortfolio
from models.data.source import Eikon
from models.data.handler import DataHandler
from pathlib import Path
import os
from models.linear_programming import LinearFactorModel
from optimization.optimizer import Optimizer
import pandas as pd
import matplotlib.pyplot as plt

class Executor:

    def __init__(self, start_date: str, end_date: str, path_data, path_api_keys):

        self.path_api_keys = path_api_keys
        self.start_date = start_date
        self.end_date = end_date
        self.path_data = path_data

        self.portfolio = EquityPortfolio(config_path='config_example.yaml')

        self.params = {
            'rics': self.portfolio.assets,
            'field': ['TR.PriceClose', 'Price Close'],
            'date_field': ['TR.PriceClose.calcdate', 'Calc Date'],
            'load_path': os.path.join(self.path_data, 'csv')
        }

        self.api = Eikon(self.path_api_keys)

    def _load_data(self):


        self.data = self.api.load_timeseries(**self.params)
        self.data = self.data[(self.data['Calc Date'] > self.start_date) & (self.data['Calc Date'] < self.end_date)]

    def _preprocess(self):

        preprocessor = DataHandler(data=self.data,
                                   date_col=self.params['date_field'][1])

        self.returns = preprocessor.get_returns(period=self.portfolio.horizon)
        self.returns.dropna(inplace=True, axis=1)

        self.n_pca_components = self.portfolio.config['MODEL']['main_factors']['PCA']
        self.pca_method = self.portfolio.config['MODEL']['pca_method']

        self.factors, self.eigen_values = preprocessor.get_pca_factors(n_components=self.n_pca_components,
                                                                       data=self.returns,
                                                                       method=self.pca_method)

        self.portfolio.set_returns(self.returns)
        self.portfolio.set_factors(self.factors)

    def _fit_factor_model(self):

        # since this takes portfolio, it can have the returns and factors in self
        self.linear_model = LinearFactorModel(config_path='config_example.yaml',
                                         portfolio=self.portfolio)

        self.linear_model.fit()
        self.linear_model.set_factor_var_covar(self.eigen_values)

    def run_optimization(self):

        self._load_data()
        self._preprocess()
        self._fit_factor_model()

        optimizer = Optimizer(config_path='config_example.yaml',
                              model=self.linear_model,
                              portfolio=self.portfolio)

        weights = optimizer.find_optimal_weights()

        return pd.Series(index=self.portfolio.assets, data=weights)


if __name__ == '__main__':

    START_DATE = '1999-12-31'
    END_DATE = '2023-06-04'
    PATH_DATA = fr'{Path(__file__).parents[2]}\models\data'
    PATH_API_KEYS = r'C:\Users\serge\OneDrive\reuters\apikeys.csv'

    executor = Executor(START_DATE, END_DATE, PATH_DATA, PATH_API_KEYS)

    weights = executor.run_optimization()

    weights.plot.hist()
    plt.show()

    print(f"Max weight: {max(weights)}")
    print(f"Min weight: {min(weights)}")

    print(f"Top 5 weights: {weights[-5]}")
