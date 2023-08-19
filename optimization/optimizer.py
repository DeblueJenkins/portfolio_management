import yaml
from portfolios.equity import EquityPortfolio
import numpy as np


class Optimizer:

    def __init__(self,  config_path: str, factors: np.ndarray, factor_loadings: np.ndarray, portfolio: EquityPortfolio):
        with open(config_path, 'rb') as f:
            self.config = yaml.safe_load(f)
            self.portfolio = portfolio
            self.factors = factors
            self.factor_loadings = factor_loadings

    def get_portfolio_return(self):
        pass

    def run(self):
        pass
