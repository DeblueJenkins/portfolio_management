import copy

from portfolios.equity import EquityPortfolio
from models.data.source import Eikon, load_fed_rates_from_excel, load_risk_free_from_ff_data
from models.data.handler import DataHandler
from pathlib import Path
import os
from models.linear_programming import LinearFactorModel
from optimization.optimizer import Optimizer
import pandas as pd
import matplotlib.pyplot as plt
from backtesting import PerformanceAssesser
import datetime as dt

class Executor:

    def __init__(self, start_date: str, end_date: str, path_data, path_api_keys):

        self.path_api_keys = path_api_keys
        self.start_date = start_date
        self.end_date = end_date
        self.path_data = path_data

        self.portfolio = EquityPortfolio(config_path='config_example.yaml')

        self.params = {
            'rics': self.portfolio.asset_universe,
            'field': ['TR.PriceClose', 'Price Close'],
            'date_field': ['TR.PriceClose.calcdate', 'Calc Date'],
            'load_path': os.path.join(self.path_data, 'csv')
        }

        self.api = Eikon(self.path_api_keys)

    def _load_data(self):


        self.data = self.api.load_timeseries(**self.params)
        self.data = self.data[(self.data['Calc Date'] > self.start_date) & (self.data['Calc Date'] < self.end_date)]

        self.rf = load_fed_rates_from_excel(fr"{PATH_DATA}\fed_rates\FEDRates21042024.csv")

    def _preprocess(self):

        preprocessor = DataHandler(data=self.data,
                                   date_col=self.params['date_field'][1])

        preprocessor.set_risk_free_rates(
            data_rates=self.rf,
            horizon=self.portfolio.horizon)

        self.returns = preprocessor.get_excess_returns(period=self.portfolio.horizon)

        self.returns.dropna(inplace=True, how='all', axis=0)
        self.returns.dropna(inplace=True, how='all', axis=1)
        self.returns.dropna(inplace=True, how='any', axis=1)
        # self.returns = self.returns.T

        self.n_pca_components = self.portfolio.config['MODEL']['main_factors']['PCA']
        self.pca_parameters = self.portfolio.config['MODEL']['pca_parameters']


        self.factors, self.eigen_values = preprocessor.get_pca_factors(n_components=self.n_pca_components,
                                                                       data=self.returns,
                                                                       **self.pca_parameters)

        preprocessor._pca_model.benchmark_test()
        preprocessor._pca_model.plot(self.n_pca_components)

        plt.savefig(fr"{self.portfolio.config['PATHS']['save_path_diagnostics']}\explained_variance_pca.png")
        plt.close()

        self.portfolio.set_returns(self.returns)
        self.portfolio.set_factors(self.factors)

    def _fit_factor_model(self):

        # since this takes portfolio, it can have the returns and factors in self
        self.linear_model = LinearFactorModel(config_path='config_example.yaml',
                                         portfolio=self.portfolio)

        self.linear_model.fit()

        self.linear_model.set_factor_var_cov(self.eigen_values)
        self.linear_model.get_factor_mean()
        self.linear_model.get_residual_var_cov()


    def run_optimization(self):

        self._load_data()
        self._preprocess()
        self._fit_factor_model()

        optimizer = Optimizer(config_path='config_example.yaml',
                              model=self.linear_model,
                              portfolio=self.portfolio)

        weights = optimizer.find_optimal_weights()

        return weights



if __name__ == '__main__':

    START_DATE = '2005-12-31'
    # this is included for data, but there is no leakage
    END_DATE = START_DATE_TEST = '2023-06-04'
    END_DATE_TEST = '2023-07-20'

    PATH_DATA = fr'{Path(__file__).parents[2]}\models\data'
    PATH_API_KEYS = r'C:\Users\serge\OneDrive\reuters\apikeys.csv'
    PATH_SAVE_PLOTS = os.path.join(r'results\figures')


    store_results = {}

    while END_DATE < '2024-04-21':

        print(END_DATE)

        store_ = {}

        executor = Executor(START_DATE, END_DATE, PATH_DATA, PATH_API_KEYS)

        horizon = executor.portfolio.horizon

        weights, res = executor.run_optimization()



        tester = PerformanceAssesser(START_DATE_TEST, END_DATE_TEST, PATH_DATA, PATH_API_KEYS)
        portfolio_summary = tester.set_portfolio(weights)
        tester.plot(path_save=fr"{PATH_SAVE_PLOTS}\{END_DATE}")

        store_['portfolio_summary'] = portfolio_summary
        store_['weights'] = weights

        store_results[END_DATE] = store_

        END_DATE = str((dt.datetime.strptime(END_DATE, '%Y-%m-%d') + dt.timedelta(days=horizon)).date())
        END_DATE_TEST = str((dt.datetime.strptime(END_DATE_TEST, '%Y-%m-%d') + dt.timedelta(days=horizon)).date())

    table_metrics = pd.DataFrame({k: v['portfolio_summary'] for k,v in store_results.items()})
    table_weights = pd.DataFrame({k: v['weights'] for k,v in store_results.items()})

    table_metrics.to_csv(fr"{os.path.join('results')}\metrics.csv")
    table_weights.to_csv(fr"{os.path.join('results')}\weights.csv")