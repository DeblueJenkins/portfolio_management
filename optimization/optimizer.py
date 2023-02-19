import numpy as np
from optimization.functions import sharpe_ratio, trenor_ratio, sortino_ratio, mean_variance_target
from scipy.optimize import minimize

class Optimizer:

    def __init__(self, method: str = 'classical M-V'):
        self.method = method

        if self.method == 'Sharpe':
            self.cost_function = sharpe_ratio
        elif self.method == 'Trenor':
            self.cost_function = trenor_ratio
        elif self.method == 'Sortino':
            self.cost_function = sortino_ratio
        elif self.method == 'classical M-V':
            self.cost_function = mean_variance_target


    def get_portfolio_weights(self, mu: np.array, sigma: np.array, r: np.array, method = 'analytic'):
        n = len(mu)
        if method == 'analytic':
           w = self.cost_function(mu, sigma, r, method = 'analytic')

        elif method == 'nummerical':
            f, cons = self.cost_function(mu, sigma, r, method = 'nummerical')
            res = minimize(fun=f,
                           x0=np.repeat((1/n), repeats=n),
                           method='L-BFGS-B',
                           constraints=cons,
                           options={'maxiter': 10,
                                    'disp': True})

            w = res.x
            print(w)


        return w