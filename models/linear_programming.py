import yaml
from abc import ABC, abstractmethod
from models.data.source import Eikon
from models.data.handler import DataHandler
import pandas as pd
from models.unsupervised.pca import PcaHandler
from models.stat_models.linearregression import MultiOutputLinearRegressionModel
from portfolios.equity import EquityPortfolio
import numpy as np
import time
from sklearn.covariance import LedoitWolf

class AbstractModel(ABC):

    def __init__(self, config_path: str):
        with open(config_path, 'rb') as f:
            self.config = yaml.safe_load(f)


    @abstractmethod
    def fit(self):
        pass

class LinearFactorModel(AbstractModel):
    '# this does one-step estimation'
    def __init__(self, portfolio: EquityPortfolio, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.portfolio = portfolio
        self.assets = self.portfolio.assets



    def fit(self, y, X, out: bool = True):

        self.multi_regressor = MultiOutputLinearRegressionModel(x=X,
                                                                y=y,
                                                                method=self.config['MODEL']['regression_method'])
        self.multi_regressor.fit()
        self.multi_regressor.get_errors()
        self.multi_regressor.diagnostics()

        # if this is not diagonal, it's an approximate factor model

        self.factors = self.multi_regressor.x[0,:].copy() # this is the last factor in the time-series
        self.factor_loadings = self.multi_regressor.betas.copy()


        if out:
            return self.factors.copy(), self.factor_loadings.copy()

    def get_portfolio_factor_return(self, weights):
        return self.factor_loadings.dot(weights).T.dot(self.factors)

    def get_residual_var_covar(self, weights):
        # estimator = LedoitWolf()
        # return estimator.fit(self.multi_regressor.residuals).covariance
        sigma_e = np.cov(self.multi_regressor.residuals, rowvar=False)
        w = weights.reshape(len(weights), 1)
        return w.T.dot(sigma_e).dot(w)[0][0]

    def set_factor_var_covar(self, eigen_values):
        self.eig_vals = eigen_values
        self.factor_var_covar = np.diag(eigen_values)

    def get_portfolio_factor_variance(self, weights: np.ndarray):
        w = weights.reshape(len(weights), 1)
        # ignore the loading of the intercept
        b = self.factor_loadings[1:].dot(w)
        return b.T.dot(self.factor_var_covar).dot(b)[0][0]

    def get_portfolio_total_variance(self, weights: np.ndarray):

        # track this in optimization
        sigma_e = self.get_residual_var_covar(weights)
        sigma_f = self.get_portfolio_factor_variance(weights)
        return sigma_f + sigma_e
    def _get_portfolio_simple_return(self, y, weights):
        portfolio_returns = np.multiply(weights, y)
        self.mean_sample = np.mean(portfolio_returns)

class HybridFactorModel(LinearFactorModel):
    pass
    '# this will do two-step estimation'



class Optimizer:

    def __init__(self):
        pass