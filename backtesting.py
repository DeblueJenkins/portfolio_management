import numpy as np
import matplotlib.pyplot as plt
from models.data.source import Eikon
from portfolios.equity import EquityPortfolio

import os
import pandas as pd


class PerformanceAssesser:

    def __init__(self, start_date: str, end_date: str, path_data, path_api_keys: str):

        self.start_date = start_date
        self.end_date = end_date
        self.path_data = path_data

        self.portfolio = EquityPortfolio(config_path='config_example.yaml')

        self.params_load = {
            'rics': self.portfolio.assets,
            'field': ['TR.PriceClose', 'Price Close'],
            'date_field': ['TR.PriceClose.calcdate', 'Calc Date'],
            'load_path': os.path.join(self.path_data, 'csv')
        }

        self.api = Eikon(path_api_keys)
        self._load_data()

    def set_portfolio(self, weights):

        self.weights = weights
        self.n = len(weights)

        equal_weights = pd.Series(index=self.portfolio.assets,
                                  data=np.repeat(1/self.portfolio.n_assets, self.portfolio.n_assets))

        res = {}

        self.value_portfolio = self.weights.T.dot(self.data_test.loc[:, self.portfolio.assets].T)
        self.value_portfolio_eq = equal_weights.T.dot(self.data_test.loc[:, self.portfolio.assets].T)

        self.value_portfolio_hist = self.weights.T.dot(self.data_all.loc[:, self.portfolio.assets].T)
        self.value_portfolio_eq_hist = equal_weights.T.dot(self.data_all.loc[:, self.portfolio.assets].T)

        self.optimal_returns = np.log(self.value_portfolio / self.value_portfolio.shift(1))
        self.equal_returns = np.log(self.value_portfolio_eq / self.value_portfolio_eq.shift(1))

        self.optimal_returns_hist = np.log(self.value_portfolio_hist / self.value_portfolio_hist.shift(1))
        self.equal_returns_hist = np.log(self.value_portfolio_eq_hist / self.value_portfolio_eq_hist.shift(1))

        # res['value_portfolio'] = self.value_portfolio
        # res['value_portfolio_eq'] = self.value_portfolio_eq
        # res['optimal_returns'] = self.optimal_returns
        # res['equal_returns'] = self.equal_returns

        res['total_return_equally_weighted'] = self.equal_returns.sum().round(4)
        res['total_return_optimally_weighted'] = self.optimal_returns.sum().round(4)
        res['volatility_equally_weighted'] = np.std(self.equal_returns_hist).round(4)
        res['volatility_optimally_weighted'] = np.std(self.optimal_returns_hist).round(4)

        return res


    def _load_data(self):

        self.data = self.api.load_timeseries(**self.params_load)
        self.data.set_index('Calc Date', inplace=True)
        self.data_test = self.data[(self.data.index > self.start_date) & (self.data.index < self.end_date)]
        self.data_all = self.data[(self.data.index < self.end_date)]



    def plot(self, path_save: str = None):

        plt.figure(figsize=(11,8))
        plt.plot(self.equal_returns, label='equally weighted', marker='o')
        plt.plot(self.optimal_returns, label='optimally weighted', marker='x')
        plt.grid(True)
        plt.legend()
        if path_save is not None:
            plt.savefig(path_save)
        # plt.show()





