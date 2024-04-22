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

    @abstractmethod
    def get_portfolio_factor_return(self, weights):
        pass

    @abstractmethod
    def get_portfolio_factor_variance(self, weights):
        pass

class LinearFactorModel(AbstractModel):
    '# this does one-step estimation'
    def __init__(self, portfolio: EquityPortfolio, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.portfolio = portfolio
        self.assets = self.portfolio.assets



    def fit(self, y = None, X = None, out: bool = True):

        if X is None:
            X = self.portfolio.factors
        if y is None:
            y = self.portfolio.returns

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

    def get_semi_variance(self, weights):
        """
            X = a + bF + e
            E[X] = a + bF

            a+bF+e - a + bF

            E(2bF + e)**2
            E[2bF**2 + 4bFe + e**2]

            2bF**2 + 0 + 1

            ((2bF + 1)* 1[if X < a+bF]) ** 0.5
        """
        mean = self.get_portfolio_factor_return(weights)
        indicators = (self.get_factor_mean(True) < mean).astype(int)
        semi_variance = (2 * self.factor_loadings.T.dot(self.factors) + np.ones(len(indicators))) * indicators
        return semi_variance

    def get_factor_mean(self, out=False):

        self.factor_mean = self.factor_loadings.T.dot(self.factors)
        if out:
            return self.factor_mean

    def get_residual_var_cov(self, out=False):

        self.residual_var_cov = np.cov(self.multi_regressor.residuals, rowvar=False)
        if out:
            return self.residual_var_cov

    def set_factor_var_cov(self, eigen_values):

        self.eig_vals = eigen_values
        # factors are independent in this case and the var-cov matrix is diagonal
        # with eigenvalues
        self.factor_var_cov = np.diag(eigen_values)

    def get_factor_var_cov(self):
        if hasattr(self, self.factor_var_cov):
            return self.factor_var_cov
        else:
            raise Exception('Factor covariance matrix not set')

    def get_portfolio_semi_variance(self, weights):

        ind = (self.factor_mean < self.get_portfolio_factor_return(weights)).astype(int)
        weights = weights * ind
        w = weights.reshape(len(weights), 1)

        b = self.factor_loadings[1:].dot(w)
        return b.T.dot(self.factor_var_cov).dot(b)[0][0]


    def get_portfolio_factor_return(self, weights):

        return self.factor_loadings.dot(weights).T.dot(self.factors)

    def get_portfolio_residual_covariance(self, weights):
        # estimator = LedoitWolf()
        # return estimator.fit(self.multi_regressor.residuals).covariance
        w = weights.reshape(len(weights), 1)
        return w.T.dot(self.residual_var_cov).dot(w)[0][0]


    def get_portfolio_factor_variance(self, weights: np.ndarray):

        w = weights.reshape(len(weights), 1)
        # ignore the loading of the intercept
        b = self.factor_loadings[1:].dot(w)
        return b.T.dot(self.factor_var_cov).dot(b)[0][0]

    def get_portfolio_total_variance(self, weights: np.ndarray):

        # track this in optimization
        sigma_e = self.get_portfolio_residual_covariance(weights)
        sigma_f = self.get_portfolio_factor_variance(weights)

        var = (sigma_e + sigma_f)
        print(f"Proportion of volatility coming from factor estimate: {sigma_f / var}")
        print(f"Proportion of volatility coming from residual estimate: {sigma_e / var}")

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