from typing import Dict
import yaml
import numpy as np
import pandas as pd


class EquityPortfolio:

    def __init__(self, config_path: str):
        """
        :param config: this is a config that initiates the portfolios, e.g.:
            config = {
                'MSFT.OQ': {'constr': 0.035},
                'AAPL.OQ': {'constr': 0.035},
                'AMZN.OQ': {'constr': 0.035},
                'GOOGL.OQ': {'constr': 0.035},
                'FB.OQ': {'constr': 0.035},
                'TSLA.OQ': {'constr': 0.035},
            }
        """

        with open(config_path, 'rb') as f:
            self.config = yaml.safe_load(f)

        self.assets = list(self.config['ASSETS'].keys())
        self.n_assets = len(self.assets)
        self.weights_constraints = np.array([list(_.values())[0] for _ in self.config['ASSETS'].values()])
        self.weights = np.repeat(1/self.n_assets, self.n_assets)
        self.allow_short_selling = self.config['CONSTRAINTS']['allow_short_selling']
        self.allow_leverage = self.config['CONSTRAINTS']['allow_leverage']
        self.returns = pd.DataFrame({})

    def set_returns(self, returns: pd.DataFrame):
        self.returns = returns






