import yaml
from portfolios.equity import EquityPortfolio
import numpy as np
import scipy.optimize as sco

from models.linear_programming import AbstractModel
from typing import Callable

class Optimizer:

    def __init__(self,  config_path: str, model: AbstractModel, portfolio: EquityPortfolio,
                 target='sharpe_ratio'):

        self.model = model
        self.get_mean = self.model.get_portfolio_factor_return
        self.get_variance = self.model.get_portfolio_factor_variance

        self.factors = model.factors
        self.factor_loadings = model.factor_loadings

        with open(config_path, 'rb') as f:

            self.config = yaml.safe_load(f)

        self.portfolio = portfolio
        self.target = target
        self.weights = self.portfolio.weights
        self.allow_short_selling = self.portfolio.allow_short_selling
        self.allow_leverage = self.portfolio.allow_leverage



        self.equal_weight_mean = self.get_mean(self.weights)
        self.equal_weight_var = self.get_variance(self.weights)
        self.equal_sharpe = self._get_sharpe(self.weights)

        self.n = len(self.factors)
        self.m = len(self.weights)

        assert self.m == self.factor_loadings.shape[1]
        assert self.n == self.factor_loadings.shape[0]


    # def _get_negative_portfolio_return(self, l):
    #     w = np.linalg.pinv(self.factor_loadings).dot(l)
    #     return -1 * self.get_portfolio_return(w)



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

    def _get_sharpe(self, weights):
        return self.get_mean(weights) / np.sqrt(self.get_variance(weights))

    def find_optimal_weights(self):

        if self.target == 'sharpe_ratio':
            f_ = lambda x: -1 * self.model.get_portfolio_factor_return(x) / self.model.get_portfolio_factor_variance(x)
        else:
            raise Exception('Target not defined')

        # individual constraints
        constraint_matrix = np.eye(self.m)
        constraints_list = [sco.LinearConstraint(constraint_matrix, lb=0, ub=self.portfolio.weights_constraints)]

        # sum to one constraint
        constraint_matrix = np.ones((1, self.m))  # Coefficients for all elements of x
        constraint_bound = 1                      # Bound for the sum of x
        constraint_sum_to_one = sco.LinearConstraint(constraint_matrix, lb=constraint_bound, ub=constraint_bound)
        constraints_list.append(constraint_sum_to_one)

        res = sco.minimize(f_, x0=self.weights, method='trust-constr', tol=1e-7, constraints=constraints_list)

        return res.x
