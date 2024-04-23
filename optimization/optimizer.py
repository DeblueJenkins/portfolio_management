import warnings

import pandas as pd
import yaml
from portfolios.equity import EquityPortfolio
import numpy as np
import scipy.optimize as sco
import scipy as sp
import pandas as pd
from scipy import stats

from models.linear_programming import LinearFactorModel
from typing import Callable

class Optimizer:

    def __init__(self,  config_path: str, model: LinearFactorModel, portfolio: EquityPortfolio):

        self.model = model
        self.get_mean = self.model.get_portfolio_factor_return
        self.get_variance = self.model.get_portfolio_factor_variance

        self.factors = model.factors
        self.factor_loadings = model.factor_loadings

        with open(config_path, 'rb') as f:

            self.config = yaml.safe_load(f)

        self.portfolio = portfolio
        self.target = self.config['MODEL']['opt_function']
        if self.target == 'expected_shortfall_univariate':
            if self.config['MODEL']['quantile'] is None:
                warnings.warn('ES Quantile set to 1% default, please specify otherwise')
                self.conf = 0.99
            else:
                self.conf = self.config['MODEL']['quantile']

        self.weights = self.portfolio.weights
        self.allow_short_selling = self.portfolio.allow_short_selling
        self.allow_leverage = self.portfolio.allow_leverage

        self.reg_lambda = self.portfolio.config['MODEL']['reg_lambda']


        self.equal_weight_mean = self.get_mean(self.weights)
        self.equal_weight_var = self.get_variance(self.weights)
        self.equal_sharpe = self._get_sharpe(self.weights)

        self.n = len(self.factors)
        self.m = len(self.weights)

        assert self.m == self.factor_loadings.shape[1]
        assert self.n == self.factor_loadings.shape[0]


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

    def _get_sharpe(self, x):
        return self.model.get_portfolio_factor_return(x) / self.model.get_portfolio_factor_variance(x) ** 0.5

    def _get_sortino(self, x):

        return self.model.get_portfolio_factor_return(x) / self.model.get_portfolio_semi_variance(x) ** 0.5

    def _add_penalty(self, x, l):

        return l * np.linalg.norm(x ** 2)

    def _get_expected_shortfall(self, x, conf: float):
        alpha = 1 - conf
        mu = self.model.get_portfolio_factor_return(x)
        vol = self.model.get_portfolio_factor_variance(x) ** 0.5
        # these can be looked up in a dictionary simply
        VaR = stats.norm.ppf(q=alpha, loc=0, scale=1)
        phi = stats.norm.pdf(x=VaR, loc=0, scale=1)
        es = mu - vol * phi / (1 - conf)
        return es


    def find_optimal_weights(self):

        target = self.portfolio.config['MODEL']['opt_function']
        if target == 'sharpe':
            f_ = lambda x: -1 * self._get_sharpe(x) + self._add_penalty(x, l=self.reg_lambda)
        elif target == 'sortino':
            f_ = lambda x: -1 * self._get_sortino(x) + self._add_penalty(x, l=self.reg_lambda)
        elif target == 'expected_shortfall_univariate':
            # this calculates it on portfolio level
            f_ = lambda x: -1 * self.model.get_portfolio_factor_return(x) / self._get_expected_shortfall(x, self.conf)

        else:
            raise Exception(f'Target opt. function {target} does not exist')

        # individual constraints
        n_remove = 2
        top_weights = self.portfolio.weights_constraints
        constraint_matrix = np.eye(self.m)


        constraints_list = [sco.LinearConstraint(constraint_matrix, lb=0, ub=top_weights)]


        # sum to one constraint
        constraint_matrix = np.ones((1, self.m))  # Coefficients for all elements of x
        constraint_bound = 1                      # Bound for the sum of x
        constraint_sum_to_one = sco.LinearConstraint(constraint_matrix, lb=constraint_bound, ub=constraint_bound)
        constraints_list.append(constraint_sum_to_one)


        # n_attempts = 20
        # optimal_weights = pd.DataFrame(index=self.portfolio.assets, data=np.zeros([self.m, n_attempts]))
        # max_func = np.zeros(n_attempts)
        # for attempt in range(n_attempts):
        # print(f'Optimization attempt: {attempt}')

        optimal_weights = pd.Series(index=self.portfolio.assets, data=np.zeros(self.m))


        x0 = self.weights + np.random.uniform(0, 1, size=self.m)
        x0[x0 < 0] = 0
        x0 /= x0.sum()
        res = sco.minimize(f_, x0=x0, method='trust-constr', tol=1e-6, constraints=constraints_list)
        # optimal_weights.iloc[:, attempt] = res.x.round(3)
        optimal_weights.iloc[:] = res.x.round(3)
        # max_func[attempt] = res.fun

        return optimal_weights, res


