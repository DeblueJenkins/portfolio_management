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
        self.factors = self.multi_regressor.x[0,:].copy() # this is the last factor in the time-series
        self.factor_loadings = self.multi_regressor.beta.copy()
        if out:
            return self.factors.copy(), self.factor_loadings.copy()

class HybridFactorModel(LinearFactorModel):
    pass
    '# this will do two-step estimation'



class Optimizer:

    def __init__(self):
        pass