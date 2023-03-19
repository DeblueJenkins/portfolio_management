import numpy as np
from portfolio_management.optimization.functions import *
from scipy.optimize import minimize, LinearConstraint
from abc import abstractmethod

class Optimizer:

    def __init__(self,  mu: np.array, sigma: np.ndarray, r: np.float):
        self.mu = mu
        self.n_assets = len(mu)

        assert sigma.shape == (self.n_assets, self.n_assets)

        self.sigma = sigma
        self.r = np.repeat(r, repeats=self.n_assets)


    def _minimize(self, cost_function):
        x0 = np.repeat((1/self.n_assets), repeats=self.n_assets)
        # cons = LinearConstraint(np.ones(self.n_assets), ub=1)
        constr_function = lambda w: np.sum(w) - 1
        cons = {"type": "eq", "fun": constr_function}
        res = minimize(fun=cost_function,
                       x0=x0,
                       method='SLSQP',
                       constraints=[cons],
                       bounds=[(0,1) for i in range(self.n_assets)],
                       options={'maxiter': 50,
                                'disp': True})
        return res

    def get_portfolio_weights(self, target='sharpe', method = 'analytical'):
        if target == 'sharpe':
            if method == 'nummerical':
                cost_function = lambda w: sharpe_ratio(w, self.mu, self.sigma, self.r[0], minimize=True)
                res = self._minimize(cost_function)
                w = res.x
            elif method == 'analytical':
                w = get_tangency_w(self.sigma, self.mu, self.r)
            else:
                raise AttributeError('Method must be specified')

        return w

        # if method == 'analytic':
        #    w = self.cost_function(mu, sigma, r, method = 'analytic')
        #
        # elif method == 'nummerical':
        #     f, cons = self.cost_function(mu, sigma, r, method = 'nummerical')
        #     res = minimize(fun=f,
        #                    x0=np.repeat((1/n), repeats=n),
        #                    method='L-BFGS-B',
        #                    constraints=cons,
        #                    options={'maxiter': 10,
        #                             'disp': True})
        #     w = res.x
        #     print(w)
        #
        #
        # return w