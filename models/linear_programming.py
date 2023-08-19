import yaml
from abc import ABC
from models.data.source import Eikon
from models.data.handler import DataHandler
import pandas as pd
from models.unsupervised.pca import PcaHandler
from models.stat_models.linearregression import MultiOutputLinearRegressionModel
from portfolios.equity import EquityPortfolio

class AbstractModel(ABC):

    def __init__(self, config_path: str, data: pd.DataFrame):
        with open(config_path, 'rb') as f:
            self.config = yaml.safe_load(f)
        self.data = data


class LinearFactorModel(AbstractModel):
    '# this does one-step estimation'
    def __init__(self, portfolio: EquityPortfolio, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.portfolio = portfolio
        self.assets = self.portfolio.assets
        self.data = self.data.loc[:, self.assets]
        self.pca_model = PcaHandler(self.data, demean=True)

    def get_factors(self):
        return self.pca_model.components(self.config['MODEL']['main_factors']['PCA']).copy()

    def fit_hyperparams(self):
        # there need to be sample weights here for OLS
        # lambda for L1 or L2
        pass

    def fit(self, out: bool = True):
        self.multi_regressor = MultiOutputLinearRegressionModel(x=self.get_factors(), y=y)
        self.factors = self.multi_regressor.x[0,:].copy() # this is the last factor in the time-series
        self.factor_loadings = self.multi_regressor.betas[1:, :].copy()
        if out:
            return self.factors.copy(), self.factor_loadings.copy()

class HybridFactorModel(LinearFactorModel):
    pass
    '# this will do two-step estimation'



class Optimizer:

    def __init__(self):
        pass