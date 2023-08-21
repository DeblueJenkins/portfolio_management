import yaml
from portfolios.equity import EquityPortfolio
import numpy as np
import scipy.optimize as sco

class Optimizer:

    def __init__(self,  config_path: str, factors: np.ndarray, factor_loadings: np.ndarray, portfolio: EquityPortfolio):
        with open(config_path, 'rb') as f:
            self.config = yaml.safe_load(f)
            self.portfolio = portfolio
            self.factors = factors
            self.factor_loadings = factor_loadings
            self.weights = self.portfolio.weights
            self.allow_short_selling = self.portfolio.allow_short_selling
            self.allow_leverage = self.portfolio.allow_leverage


            self.n = len(factors)
            self.m = len(self.weights)

            assert self.m == self.factor_loadings.shape[1]
            assert self.n == self.factor_loadings.shape[0]


    def get_portfolio_return(self, weights):
        return self.factor_loadings.dot(weights).T.dot(self.factors)

    def _get_negative_portfolio_return(self, l):
        w = np.linalg.pinv(self.factor_loadings).dot(l)
        return -1 * self.get_portfolio_return(w)

    def get_bounds(self, allow_short_selling: bool, allow_leverage: bool = True):
        """
        TO DO: do it per asset
        returns bounds for short selling and not short selling for the scipy optimizer

        :param allow_short_selling: boolean variance false or true
        :return: a set of tuples consistent with the scipy optimizer
        """
        if allow_short_selling and allow_leverage:
            return tuple((None, None) for x in range(self.m))
        elif not allow_short_selling and allow_leverage:
            return tuple((0, None) for x in range(self.m))
        else:
            return tuple((0, 1) for x in range(self.m))

    def _get_one_constraint(self, l, i):
        w = np.linalg.pinv(self.factor_loadings).dot(l)
        return -1 * (w[i] - self.portfolio.weights_constraints[i])

    def _get_sum_constraint(self, l):
        w = np.linalg.pinv(self.factor_loadings).dot(l)
        return np.sum(w) - 1

    def get_all_constraints(self):
        cons = []
        # these are all individual constraints
        for i in range(self.m):
            cons.append({'type': 'ineq', 'fun': lambda l: self._get_one_constraint(l, i)})
        # this is sum to 1 constraint
        cons.append({'type': 'eq', 'fun': self._get_sum_constraint})
        return cons

    def find_optimal_weights(self):
        x0 = np.ones(self.m)/self.m
        x0 = x0.dot(self.factor_loadings.T)
        # TO DO: add bounds
        res = sco.minimize(self._get_negative_portfolio_return, x0=x0, method='SLSQP',
                           constraints=self.get_all_constraints(), tol=1e-15)
        res = res.x[:, np.newaxis]
        weights = np.linalg.pinv(self.factor_loadings).dot(res)
        return weights
