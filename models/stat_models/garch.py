import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import minimize
from models.stat_models.statmaths import jacobian_2sided, covariance, make_covariance_robust
from scipy.stats import norm
import pandas as pd
from numba import jit
import warnings
from scipy.optimize import minimize
from typing import Dict

class Garch:

    @staticmethod
    def parameter_transform(theta, vector: bool = True):
        """
        applies parameter transforms to ensure that the parameters stay in their respective ranges
        :param theta: parameter vector (w, a, b)
        :param vector:
        :return:
        """
        r = (
            (np.exp(theta[0])),
            (np.exp(theta[1])),
            (1/(1+np.exp(-theta[2])))
        )
        if vector:
            return np.append([], r)
        else:
            return r

    def __init__(self, conf: float = 0.05):
        """
        garch model fitted using the delta method. The delta method transforms the variables which have constraints, to
        unconstrained variables. w, a, b > 0 and a + b < 1, thus we transform w = exp(tw) and a = exp(ta) to make
        w and a positive and set b = 1/(1 + exp(-tb)) to keep it between 0 and 1 (apperantly if stationary keeping b in
        (0,1) makes a + b < 1 hold if alpha is positive). Then we solve for tw, ta and tb and transform them back. This
        allows un constraint optimization which is super fast.
        :param conf: level for t-tests
        """
        self.conf = conf
        self.data = None
        self.n = None
        self.init_sigma2 = None
        self.x0 = None
        self.tx0 = None
        self.x = None
        self.tx = None
        self.sol = None
        self.variances = None
        self.likelihood_values = None
        self.fitted = False

    def garch(self, w: float, a: float, b: float, sigma2: float) -> (np.array, np.array):
        """
        in-sample garch model used for estimation and diagnostic plots
        :param w: omega parameter
        :param a: alpha parameter
        :param b: beta parameter
        :param sigma2: initial value for sigma2
        :return: time series of variances and log-likelihood contributions
        """
        variances = np.zeros(self.n)
        likelihood_values = np.ones(self.n)

        for i1 in range(self.n):

            variances[i1] = sigma2

            likelihood_values[i1] = -0.5 * np.log(sigma2) - 0.5 * self.data[i1] * self.data[i1]/sigma2

            sigma2 = w + b * sigma2 + a * self.data[i1] * self.data[i1]

        likelihood_values = likelihood_values - 0.5 * np.log(2*np.pi)

        return variances, likelihood_values

    def likelihood(self, theta) -> np.array:
        """
        computes the likelihood contributions per observation
        :param theta: parameter vector (w, a, b)
        :return: array of likelihood contributions
        """
        w, a, b = self.parameter_transform(theta)

        data_variances, likelihood_values = self.garch(w=w, a=a, b=b, sigma2=self.init_sigma2)

        return likelihood_values

    def objective_function(self, theta) -> float:
        """
        computes the negative of the mean likelihood given a parameter vector theta
        :param theta: parameter vector (w, a, b)
        :return: negative of the likelihood function
        """
        likelihood_values = self.likelihood(theta=theta)

        return -np.mean(likelihood_values)

    def estimate_model(self, data: np.array, init_value_sigma2: float = None, init_value_theta: tuple = None) -> None:
        """
        estimates the garch model, if one or both of the initial are None the code will assign some
        :param data: numpy array of time series data
        :param init_value_sigma2: initial value of sigma squared
        :param init_value_theta: initial guess of the parameter vector (w, a, b)
        :return: nothing
        """
        self.data = data
        self.n = self.data.shape[0]

        self.set_initial_guesses(init_value_sigma2=init_value_sigma2, init_value_theta=init_value_theta)
        self.tx0 = self.parameter_transform(self.x0)

        self.sol = minimize(self.objective_function, self.x0, method='BFGS', options={'disp': True, 'maxiter': 250})

        self.x = self.sol['x']
        self.tx = self.parameter_transform(self.x)
        self.variances, self.likelihood_values = self.garch(w=self.tx[0], a=self.tx[1],
                                                            b=self.tx[2], sigma2=self.init_sigma2)

        self.fitted = True

    def set_initial_guesses(self, init_value_sigma2: float = None, init_value_theta: tuple = None) -> None:
        """
        sets the initial guesses for the parameter vector and sigma2_0
        :param init_value_sigma2: initial value of sigma squared
        :param init_value_theta: initial guess of the parameter vector (w, a, b)
        :return: nothing
        """
        if init_value_sigma2 is None:
            self.init_sigma2 = np.var(self.data)

        else:
            self.init_sigma2 = init_value_sigma2

        if init_value_theta is None:
            self.x0 = np.log([0.02, 0.05, 0.93])

        else:
            if len(init_value_theta) == 3:
                self.x0 = np.log([init_value_theta[0],
                                  init_value_theta[1],
                                  init_value_theta[2]])

            else:
                print("no, sergej no")

    def plot_in_sample(self) -> None:
        """
        plots the time series + the garch estimated variances
        :return: nothing
        """
        if self.fitted:
            print("parameters:")
            print("w :" + str(self.tx[0]))
            print("a :" + str(self.tx[1]))
            print("b :" + str(self.tx[2]))
            fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
            ax1.plot(self.data, label='data', color="black", linewidth=2)
            ax1.plot(np.sqrt(self.variances), label='vols', color="green", linewidth=2)
            ax1.tick_params(labelrotation=45)
            ax1.set_title("Data vs Garch Volatility")
            ax1.set_xlabel("T ->")
            ax1.grid()
            ax1.legend()
            fig.show()

        else:
            print("fit the model first")

    def test_model_vs_generated_data(self, n: int, w: float, a: float, b: float, s2: float) -> None:
        """
        generates a garch sample and estimates the model on it
        :param n: number of observations
        :param w: omega parameter
        :param a: alpha parameter
        :param b: beta parameter
        :param s2: initial value for sigma2
        :return: nothing
        """
        data, variances = generate_garch_data(n=n, w=w, a=a, b=b, s2=s2)
        self.estimate_model(data=data)
        print("parameters:")
        print("w :" + str(self.tx[0]))
        print("a :" + str(self.tx[1]))
        print("b :" + str(self.tx[2]))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        ax1.plot(self.data, label='data', color="black", linewidth=2)
        ax1.plot(self.variances, label='variances', color="green", linewidth=2)
        ax1.tick_params(labelrotation=45)
        ax1.set_title("Data vs Garch Variances")
        ax1.set_xlabel("T ->")
        ax1.grid()
        ax1.legend()
        ax2.plot(variances, label='data variances', color="black", linewidth=2)
        ax2.plot(self.variances, label='variances', color="green", linewidth=2)
        ax2.tick_params(labelrotation=45)
        ax2.set_title("Data Variances vs Garch Variances")
        ax2.set_xlabel("T ->")
        ax2.grid()
        ax2.legend()
        fig.show()

    def compute_standard_errors(self, robust: bool = False) -> np.array:
        """
        computes the standard errors for the parameters, taking into account the variable transform.
        :param robust: robust or non-robust standard errors.
        :return: array of standard errors for each parameter
        """
        if self.fitted:
            cov_non_robust = covariance(theta=self.x, average_likelihood_func=self.objective_function, n=self.n)
            jac = jacobian_2sided(self.parameter_transform, self.x, True)

            if not robust:
                cov = np.linalg.multi_dot([jac, cov_non_robust, jac.T])

            else:
                cov_robust = make_covariance_robust(cov=cov_non_robust, theta=self.x, likelihood_func=self.likelihood)
                cov = np.linalg.multi_dot([jac, cov_robust, jac.T])

            return np.sqrt(np.diag(cov))

        else:
            print("first fit model on data")
            return np.array([])

    def t_tests(self, robust: bool = False) -> pd.DataFrame:
        """
        performs "t-tests" for the parameters, taking into account the variable transform.
        :param robust: robust or non-robust standard errors.
        :return: data frame with results
        """
        if self.fitted:

            std_errs = self.compute_standard_errors(robust=robust)
            top_quantile = self.tx + norm.ppf(1-self.conf/2) * std_errs
            bot_quantile = self.tx + norm.ppf(self.conf/2) * std_errs
            checks = []
            for a, b in zip(bot_quantile, top_quantile):
                if a <= 0 <= b:
                    checks.append("Not Significant")
                else:
                    checks.append("Significant")

            return pd.DataFrame({"bot quantile:": list(bot_quantile),
                                 "top quantile:": list(top_quantile),
                                 "Results": checks})
        else:
            print("first fit model on data")
            return pd.DataFrame({})


def generate_garch_data(n: int, w: float, a: float, b: float, s2: float, dist: str = 'normal', *args, **kwargs):

    variances = np.zeros(n)
    data = np.zeros(n)

    if dist == 'normal':
        z = scipy.stats.norm.rvs(loc=0, scale=1, size=n)
    elif dist == 't':
        z = scipy.stats.t.rvs(*args, loc=0, scale=1, size=n)
    else:
        raise Exception('distribution of errors can only be normal or t')

    for i in range(n):

        data[i] = np.sqrt(s2) * z[n]
        variances[i] = s2
        s2 = w + b * s2 + a * data[i] ** 2

    return data, variances


def get_ewma_mean(r: np.ndarray, l: float, tau: int):

    assert len(r) == tau
    decay = l ** np.arange(0, tau)
    mean = np.dot(decay, r) / np.sum(decay)
    return mean

def get_ewma_variance(r: np.ndarray, l: float, tau: int):

    assert len(r) == tau
    # this can be improved by using intra-day returns to proxy variance instead of r^2
    decay = l ** np.arange(0, tau)
    sqr_demean_ret = (r - get_ewma_mean(r, l, tau)) ** 2
    var = np.dot(decay, sqr_demean_ret) / np.sum(decay)
    return var


class EWMA:

    def __init__(self, data: np.ndarray, tau: int, realized_variance: np.ndarray = None, realized_dates: np.ndarray = None):
        """

        :param data: np.array, first entry has to be the most recent observation
        :param tau: int, rolling window for EWM
        """
        if isinstance(data, np.ndarray):
            self.data = data.astype(float)
            self.dates = None
        elif isinstance(data, pd.Series):
            self.data = data.values.astype(float)
            self.dates = data.index.values

        self.realized_variance = realized_variance
        self.realized_dates = realized_dates

        warnings.warn('Data must be sorted s.t. first entry is the most recent observation')
        self.tau = tau
        self.T = len(data)

    def fit(self, out=False, objective=None):
        if objective == 'rv':
            objective = self.objective_realized_variance
        else:
            objective = self.objective
        res = self._fit(self.data, self.tau, objective)
        self.l = res.x[0]
        if out:
            return res

    def get_conditional_moments(self, l: float = None, out=True):
        if l is None:
            l = self.l
        elif not isinstance(l, float):
            raise UserWarning('lambda must be provided as a float')

        self.conditional_variance = np.zeros(self.T-self.tau-1)
        self.conditional_mean = np.zeros(self.T-self.tau-1)
        for t in np.arange(0, self.T-self.tau-1):
            # careful, there is a difference in t here, r_{t} - mean_{t-1} / std_{t-1}
            mean = get_ewma_mean(self.data[t+1:t+self.tau+1], l, self.tau)
            var = get_ewma_variance(self.data[t+1:t+self.tau+1], l, self.tau)
            self.conditional_variance[t] = var
            self.conditional_mean[t] = mean

        if self.dates is not None:

            self.conditional_mean = pd.Series(data=self.conditional_mean, index=self.dates[:self.T-self.tau-1], name='ewm_mean')
            self.conditional_variance = pd.Series(data=self.conditional_variance, index=self.dates[:self.T-self.tau-1], name='ewm_var')

        if out:
            return self.conditional_mean, self.conditional_variance


    def objective_realized_variance(self, l, data, tau):

        T = len(data)
        err = np.zeros(T-tau-1)
        for t in np.arange(0, T-tau-1):

            var = get_ewma_variance(data[t+1:t+tau+1], l, tau)
            date_last = pd.Timestamp(self.dates[t+1]).date()
            if date_last not in self.realized_dates:
                continue
            else:
                real_var = self.realized_variance[np.argwhere(self.realized_dates == date_last)]
                real_var = real_var[0][0]
                r = data[t]
                _sqr_error = np.sum((real_var - var) ** 2)
                err[t] = _sqr_error

        ssr = np.sum(err) / (T-tau)
        rmse = np.sqrt(np.mean(err))
        return rmse

    @staticmethod
    def objective(l, data, tau):
        T = len(data)
        rmse = np.zeros(T-tau-1)
        for t in np.arange(0, T-tau-1):

            mean = get_ewma_mean(data[t+1:t+tau+1], l, tau)
            var = get_ewma_variance(data[t+1:t+tau+1], l, tau)
            r = data[t]
            _rmse = ((r - mean) ** 2 - var) ** 2
            rmse[t] = _rmse


        rmse = np.sum(rmse) / (T-tau)
        return rmse


    @staticmethod
    # @jit(nopython=True)
    def _fit(data, tau, objective):
        if objective is None:
            objective = self.objective

        func = lambda x: objective(x, data, tau)
        res = minimize(fun=func, x0=0.8)
        return res

class MultivariateEWMA:

    def __init__(self, data: np.ndarray, tau: int):


        if isinstance(data, pd.DataFrame):
            self.asset_names = data.columns
            self.is_df = True
        elif isinstance(data, np.ndarray):
            self.asset_names = np.arange(data.shape[1])
            self.is_df = False
        self.n_assets = len(self.asset_names)
        self.tau = tau

        marginal_models = {}
        marginal_decay_parameters = {}
        for a in range(self.n_assets):
            if self.is_df:
                model = EWMA(data.iloc[:, a], tau)
            else:
                model = EWMA(data[:, a], tau)
            model.fit()
            marginal_models[self.asset_names[a]] = model
            marginal_decay_parameters[self.asset_names[a]] = model.l




class IntradayVariance:
    pass


class IntradayCovariance:
    pass



