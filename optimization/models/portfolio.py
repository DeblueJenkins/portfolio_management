import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from models.stat_models.statmaths import covariance_matrix, mean, cholesky
from models.stat_models.garch import generate_garch_data
import scipy.optimize as sco
from scipy import stats

def generate_portfolio_data(mu: np.array, cov: np.array, training_size: int, testing_size: int):

    """
    generates training and testing data for a portfolio model
    :param mu: a numpy array of given mean returns
    :param cov: a numpy array of the covariances between all assets
    :param training_size: number of training cases
    :param testing_size: number of test cases

    :return: sample of one period of returns to train the model on and a sample to test the model against
    """

    if np.shape(cov)[0] == np.shape(cov)[1]:
        ValueError("covariance matrix is not square")

    if np.shape(mu)[0] == np.shape(cov)[0]:
        a = cholesky(cov)
        z = np.random.normal(0, 1, (training_size, len(mu)))
        training_sample = mu + np.dot(z, a.T)
        z = np.random.normal(0, 1, (testing_size, len(mu)))
        testing_sample = mu + np.dot(z, a.T)
        return training_sample, testing_sample

    else:
        ValueError("mu is of incorrect dimentions")

def generate_portfolio_garch_data(mu: np.array, initial_variances: np.ndarray, corr: np.array, n_time: int, omega: np.ndarray = None, alpha: np.ndarray = None,
                                  beta: np.ndarray = None, dof: np.ndarray = None):

    n_assets = len(mu)
    assert len(mu) == corr.shape[0] == corr.shape[1]
    if omega is None:
        omega = np.random.uniform(0, 0.0005, size=n_assets)
    if alpha is None:
        alpha = np.random.uniform(0, 0.25, size=n_assets)
    if beta is None:
        beta = np.random.uniform(0, 0.05, size=n_assets)
    if dof is None:
        dof = np.random.uniform(1, 30, n_assets)


    garch_variances = np.zeros((n_assets,n_time))
    data = np.zeros((n_assets,n_time))
    cov = np.zeros((n_assets, n_assets))



    eps = stats.t.rvs(dof, loc=0, scale=1, size=(n_time, n_assets))
    # eps = stats.norm.rvs(loc=0, scale=1, size=(n_time, n_assets))
    # corr_eps = np.dot(eps, a.T).T
    np.fill_diagonal(cov, (omega / (1 - alpha - beta)))
    cov = cov.dot(corr).dot(cov)
    a = cholesky(cov)
    eps = eps.dot(a.T).T

    garch_variances[:, 0] = initial_variances
    data[:, 0] = mu + eps
    for t in np.arange(1, n_time):
        garch_variances[:,t] = omega + np.multiply(alpha, garch_variances[:, t-1]) + np.multiply(beta, eps[:, t-1] ** 2)
    np.fill_diagonal(cov, garch_variances)
    cov = cov.dot(corr).dot(cov)


    return data[:, 1:]



class PortfolioModel(ABC):

    def __init__(self, data):

        self.index = np.array([])
        self.x = np.array([])
        self.n: int = 0
        self.m: int = 0
        self.assets: list = []
        self.mu = np.array([])
        self.cov = np.array([])

        self.handle_input_data(data=data)

        self.weights = np.array([])
        self.fitted = False
        self.mu_p: float = 0.0
        self.vol_p: float = 0.0

    def handle_input_data(self, data) -> None:
        """
        input data handling, ensures that all initial parameters are correctly assigned

        :param data: matrix of data (free of type can be a data frame or numpy array)
        :return: nothing all data is set though
        """

        if isinstance(data, pd.DataFrame):
            self.index = np.array(data.index)
            self.assets = list(data.columns[1:])
            self.x = data.astype(float).values
        elif isinstance(data, np.ndarray):
            self.index = np.arange(np.shape(data)[0])
            self.x = data
            self.assets = np.arange(np.shape(data)[1])
        else:
            raise UserWarning('Data must be either np.array or pd.DataFrame')

        self.n, self.m = self.x.shape

        self.mu = mean(self.x)
        self.cov = covariance_matrix(self.x)

    def estimate_model(self, optimization_type: str, constraint: float = 0.0, allow_short_selling: bool = True) -> None:
        """
        estimates the markowitz portfolio

        :param optimization_type: string either GMVP, MEANCONSTRAINT or VOLCONSTRAINT
        :param constraint: a constraint for mean or vol depending on optimization_type
        :param allow_short_selling: boolean variable false or true
        :return: non but parameters are set
        """

        if constraint <= 0.0:
            ValueError("fill in a proper constraint")

        else:
            if optimization_type.upper() in ["MEANCONSTRAINT", "MEAN CONSTRAINT", "MEAN_CONSTRAINT"]:

                self.weights = self.minimize_sigma_weights(constraint, allow_short_selling=allow_short_selling)
                self.mu_p, self.vol_p = self.compute_mean_vol(w=self.weights)
                self.fitted = True

            elif optimization_type.upper() in ["VOLCONSTRAINT", "VOL CONSTRAINT", "VOL_CONSTRAINT"]:

                self.weights = self.optimize_mean_weights(constraint, allow_short_selling=allow_short_selling)
                self.mu_p, self.vol_p = self.compute_mean_vol(w=self.weights)
                self.fitted = True

            else:
                Exception("There are only three different variations in this model MEAN CONSTRAINT and VOL CONSTRAINT")

    def portfolio_return(self, w: np.array) -> float:
        """
        returns the mean portfolio return given a set of weights

        :param w: array of weights
        :return: mean return
        """

        if len(w) == len(self.mu):
            if np.shape(w) != np.shape(self.mu):
                w = w[:, np.newaxis]
            return float(np.dot(w.T, self.mu))
        else:
            ValueError("Shapes of weight and mean vector are not the same")

    def negative_portfolio_return(self, w: np.array) -> float:
        """
        returns the mean portfolio return given a set of weights (used as an objective function)

        :param w: array of weights
        :return: negative mean return
        """

        return -self.portfolio_return(w=w)

    def portfolio_var(self, w: np.array):
        """
        computes the variance of the portfolio returns given a set of weights
        :param w: array of weights
        :return: estimated portfolio variance
        """

        return np.linalg.multi_dot([w.T, self.cov, w])

    def portfolio_volatility(self, w: np.array) -> float:
        """
        computes the volatility of the portfolio returns given a set of weights
        :param w: array of weights
        :return: estimated portfolio volatility
        """

        return float(np.sqrt(self.portfolio_var(w=w)))

    def compute_mean_vol(self, w: np.array) -> (float, float):
        """
        computes the mean and volatility of the portfolio returns given a set of weights
        :param w: array of weights
        :return: estimated portfolio mean and estimated portfolio volatility
        """

        mu = self.portfolio_return(w)
        vol = self.portfolio_volatility(w)
        return mu, vol

    @abstractmethod
    def minimize_sigma_weights(self, mean_constraint: float, allow_short_selling: bool = True):
        pass

    def minimize_sigma_mean_vol(self, mean_constraint: float, allow_short_selling: bool = True) -> (float, float):
        """
        computes the minimal variance portfolio mean and volatility given a mean constraint

        :param mean_constraint: constraint for the mean
        :param allow_short_selling: boolean variable false or true
        :return: mean and volatility
        """
        w = self.minimize_sigma_weights(mean_constraint=mean_constraint, allow_short_selling=allow_short_selling)
        return self.compute_mean_vol(w)

    @abstractmethod
    def optimize_mean_weights(self, vol_constraint: float, allow_short_selling: bool = True):
        pass

    def optimize_mean_mean_vol(self, vol_constraint: float, allow_short_selling: bool = True) -> (float, float):
        """
        computes the minimal variance portfolio mean and volatility given a volatility constraint

        :param vol_constraint: constraint for the volatility
        :param allow_short_selling: boolean variable false or true
        :return: mean and volatility
        """
        w = self.optimize_mean_weights(vol_constraint=vol_constraint, allow_short_selling=allow_short_selling)
        return self.compute_mean_vol(w)

    def equal_weights(self) -> np.array:
        """
        returns equal weights portfolio
        :return: array of equal weights
        """

        return np.ones(self.m)/self.m

    def get_bounds(self, allow_short_selling: bool, allow_leverage: bool = True):
        """
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


class MarkowitzPortfolio(PortfolioModel):

    def __init__(self, data):

        super().__init__(data=data)

    def estimate_model(self, optimization_type: str, constraint: float = 0.0, allow_short_selling: bool = True) -> None:
        """
        estimates the markowitz portfolio

        :param optimization_type: string either GMVP, MEANCONSTRAINT or VOLCONSTRAINT
        :param constraint: a constraint for mean or vol depending on optimization_type
        :param allow_short_selling: boolean variable false or true
        :return: non but parameters are set
        """

        if optimization_type.upper() == "GMVP":
            self.weights = self.global_minimal_variance_weights(allow_short_selling)
            self.mu_p, self.vol_p = self.compute_mean_vol(w=self.weights)
            self.fitted = True

        else:

            if constraint <= 0.0:
                ValueError("fill in a proper constraint")

            else:
                super().estimate_model(optimization_type=optimization_type, constraint=constraint,
                                       allow_short_selling=allow_short_selling)

    def global_minimal_variance_weights(self, allow_short_selling: bool = True) -> np.array:
        """
        returns the weights for the global mean variance portfolio

        :param allow_short_selling: boolean variable false or true
        :return: a set of weights
        """

        if allow_short_selling:
            ones = np.ones(self.m)[:, np.newaxis]
            inv_cov = np.linalg.inv(self.cov)
            lambda1 = 1/np.linalg.multi_dot([ones.T, inv_cov, ones])
            return np.dot(inv_cov, ones) * lambda1

        else:

            cons = [{'type': 'eq', 'fun': lambda x: sum(x) - 1}]
            res = sco.minimize(self.portfolio_var, x0=self.equal_weights(), method='SLSQP',
                               bounds=self.get_bounds(allow_short_selling=allow_short_selling),
                               constraints=cons, tol=1e-15)
            res = res.x[:, np.newaxis]
            return res

    def global_minimal_variance_portfolio_mean_vol(self, allow_short_selling: bool = True) -> (float, float):
        """
        computes the minimal variance portfolio

        :param mean_constraint: constraint for the mean
        :param allow_short_selling: boolean variable false or true
        :return: mean and volatility
        """

        w = self.global_minimal_variance_weights(allow_short_selling=allow_short_selling)
        return self.compute_mean_vol(w)

    def minimize_sigma_weights(self, mean_constraint: float, allow_short_selling: bool = True) -> np.array:
        """
        computes the minimal variance portfolio weights given a mean constraint

        :param mean_constraint: constraint for the mean
        :param allow_short_selling: boolean variable false or true
        :return: a set of weights
        """

        if allow_short_selling:

            ones = np.ones(np.shape(self.mu))
            inv_cov = np.linalg.inv(self.cov)
            a = np.linalg.multi_dot([ones.T, inv_cov, ones])
            b = np.linalg.multi_dot([self.mu.T, inv_cov, ones])
            c = np.linalg.multi_dot([self.mu.T, inv_cov, self.mu])
            lambda1 = (mean_constraint*a-b)/(a*c - b*b)
            lambda2 = (c-b*mean_constraint)/(a*c - b*b)
            return np.dot(inv_cov, (self.mu * lambda1 + ones * lambda2))

        else:

            cons = [{'type': 'eq', 'fun': lambda x: sum(x) - 1},
                    {'type': 'eq', 'fun': lambda x: np.dot(x.T, self.mu) - mean_constraint}]
            res = sco.minimize(self.portfolio_var, x0=self.equal_weights(), method='SLSQP',
                               bounds=self.get_bounds(allow_short_selling=allow_short_selling),
                               constraints=cons, tol=1e-15)
            res = res.x[:, np.newaxis]
            return res

    def optimize_mean_weights(self, vol_constraint: float, allow_short_selling: bool = True) -> np.array:

        """
        computes the maximum return given a volatility constraint

        :param vol_constraint: constraint for the volatility
        :param allow_short_selling: boolean variable false or true
        :return: a set of weights
        """

        if vol_constraint < self.global_minimal_variance_portfolio_mean_vol(allow_short_selling=allow_short_selling)[1]:
            ValueError("vol_constraint below global minimal variance")

        else:

            var_constraint = pow(vol_constraint, 2)

            cons = [{'type': 'eq', 'fun': lambda x: sum(x) - 1},
                    {'type': 'eq', 'fun': lambda x: np.linalg.multi_dot([x.T, self.cov, x]) - var_constraint}]

            res = sco.minimize(self.negative_portfolio_return, x0=self.equal_weights(), method='SLSQP',
                               bounds=self.get_bounds(allow_short_selling), constraints=cons, tol=1e-15)

            res = res.x[:, np.newaxis]
            return res

    def efficient_frontier(self, top: float = 0.5, bot: float = -0.1) -> None:

        """
        Plots the efficient frontier with randomized weights portfolio and if fitted shows the portfolio on the frontier

        :param top: maximum mean return
        :param bot: minimum mean return
        :return:
        """
        if bot >= top:
            ValueError("top should be greater then bot")

        mean_returns = np.linspace(bot, top, 400)
        vols_frontier = []
        means_frontier = []

        for mean_return in mean_returns:
            m, v = self.minimize_sigma_mean_vol(mean_return)
            vols_frontier.append(v)
            means_frontier.append(m)

        random_weights = np.random.uniform(-1, 1, (5000, self.m))
        weights = [weight/sum(weight) for weight in random_weights]

        means_random = []
        vols_random = []

        for weight in weights:
            m, v = self.compute_mean_vol(weight)
            if (top > m > bot) and (v < vols_frontier[-1]):
                means_random.append(m)
                vols_random.append(v)

        plt.plot(vols_frontier, means_frontier, label="Efficient Frontier", color="k")
        plt.scatter(vols_random, means_random, label="Random Weights", color="b", s=1.5)

        if self.fitted:
            plt.scatter([self.vol_p], [self.mu_p], label="Portfolio", color="r", s=50)

        plt.title("Efficient Frontier vs Random Weights")
        plt.xlabel("Volatility ($\sigma$)")
        plt.ylabel("Portfolio Mean Return ($\mu$)")
        plt.grid()
        plt.legend()
        plt.show()

    def backtests(self, training_sample, testing_sample, optimization_type: str, constraint: float = 0.0,
                  allow_short_selling: bool = True) -> None:

        """
        has to be increased in scope, but assesses the rowling window realized return of a portfolio
        :param training_sample: array/dataframe of training data
        :param testing_sample: array/dataframe of testing data
        :param optimization_type: string stating the optimization type
        :param constraint: optional constraint on mean or volatility
        :param allow_short_selling: boolean set to true if short selling is allowed
        :return: None
        """

        if isinstance(training_sample, pd.DataFrame):
            training_sample = training_sample.astype(float).values
        elif isinstance(training_sample, np.ndarray):
            training_sample = training_sample
        else:
            raise UserWarning('Data must be either np.array or pd.DataFrame')

        if isinstance(testing_sample, pd.DataFrame):
            testing_sample = testing_sample.astype(float).values
        elif isinstance(testing_sample, np.ndarray):
            testing_sample = testing_sample
        else:
            raise UserWarning('Data must be either np.array or pd.DataFrame')

        returns_list = []

        if np.shape(testing_sample)[1] != np.shape(self.x)[1]:
            raise ValueError('number of assets in training_sample (' + str(np.shape(testing_sample)[1]) + ') is not the same as in testing_sample (' + str(np.shape(self.x)[1]) + ') ')
        else:
            for t in range(np.shape(testing_sample)[0]):
                self.handle_input_data(data=training_sample)
                self.estimate_model(optimization_type=optimization_type, constraint=constraint,
                                    allow_short_selling=allow_short_selling)

                y = testing_sample[t][np.newaxis, :]
                realization = float(np.dot(y, self.weights))
                returns_list.append(realization)
                training_sample = np.concatenate((training_sample, y))

        mean_sample = np.mean(returns_list)
        vol_sample = np.std(returns_list)
        plt.hist(returns_list, label="realized returns", density=True, edgecolor='black')
        plt.grid()
        plt.legend()
        plt.title("realized returns")
        plt.show()
        print("Mean Realized Returns: " + str(np.round(mean_sample, 6)))
        print("Volatility Realized Returns: " + str(np.round(vol_sample, 6)))


class SharpePortfolio(PortfolioModel):

    def __init__(self, data, rf: float):

        super().__init__(data=data)
        self.rf = rf

    def estimate_model(self, optimization_type: str, constraint: float = 0.0, allow_short_selling: bool = True, allow_leverage: bool = True, out: bool = False) -> None:
        """
        estimates the sharp portfolio

        :param optimization_type: string either TANGENCY, MEANCONSTRAINT, VOLCONSTRAINT, SHARPE
        :param constraint: a constraint for mean or vol depending on optimization_type
        :param allow_short_selling: boolean variable false or true
        :return: non but parameters are set
        """

        if optimization_type.upper() in ["TANGENCY", "SHARPE"]:
            self.weights = self.get_tangent_portfolio_weights(allow_short_selling, allow_leverage)
            self.mu_p, self.vol_p = self.compute_mean_vol(w=self.weights)
            self.fitted = True

        else:

            if constraint <= 0.0:
                raise ValueError("fill in a proper constraint")

            else:
                super().estimate_model(optimization_type=optimization_type, constraint=constraint,
                                       allow_short_selling=allow_short_selling)
        if out:
            return self.weights

    def sharpe_ratio(self, w: np.ndarray):

        return (np.dot(w.T, self.mu) - self.rf) / self.portfolio_volatility(w=w)

    def negative_sharpe_ratio(self, w: np.ndarray):

        return -self.sharpe_ratio(w=w)

    def portfolio_return(self, w: np.ndarray) -> float:
        """
        returns the mean portfolio return given a set of weights

        :param w: array of weights
        :return: mean return
        """

        return super().portfolio_return(w) + (1 - sum(w)) * self.rf

    def portfolio_excess_return(self, w: np.ndarray, rf: float) -> float:
        """
        returns the mean excess portfolio return given a set of weights

        :param w: array of weights
        :return: mean return
        """

        if len(w) == len(self.mu):
            if np.shape(w) != np.shape(self.mu):
                w = w[:, np.newaxis]
            return float(np.dot(w.T, np.array(self.mu - rf)))
        else:
            ValueError("Shapes of weight and mean vector are not the same")

    def negative_portfolio_return(self, w: np.ndarray) -> float:
        """
        returns the mean portfolio return given a set of weights

        :param w: array of weights
        :return: mean return
        """

        return - self.portfolio_return(w=w)

    def get_tangent_portfolio_weights(self, allow_short_selling: bool = True, allow_leverage: bool = True) -> np.ndarray:
        """
        portfolio with maximum sharp ratio

        :param allow_short_selling: boolean variable false or true
        :return: a set of weights
        """

        if allow_short_selling and allow_leverage:
            ones = np.ones(np.shape(self.mu))
            cov_inv = np.linalg.inv(self.cov)
            top = np.dot(cov_inv, np.subtract(self.mu, self.rf * ones))
            bot = np.linalg.multi_dot([ones.T, cov_inv, np.subtract(self.mu, self.rf * ones)])
            res = top/bot
        else:
            cons = [{'type': 'eq', 'fun': lambda x: sum(x) - 1}]
            res = sco.minimize(self.negative_sharpe_ratio, x0=self.equal_weights(), method='SLSQP',
                               bounds=self.get_bounds(allow_short_selling=allow_short_selling, allow_leverage=allow_leverage),
                               constraints=cons, tol=1e-15)
            res = res.x[:, np.newaxis]

        return res

    def get_tangent_portfolio_weights_mean_vol(self, allow_short_selling: bool = True):
        """

        :param allow_short_selling: boolean variable false or true
        :return: mean and volatility
        """
        w = self.get_tangent_portfolio_weights(allow_short_selling=allow_short_selling)
        return self.compute_mean_vol(w=w)

    def minimize_sigma_weights(self, mean_constraint: float, allow_short_selling: bool = True):

        if allow_short_selling:
            ones = np.ones(np.shape(self.mu))
            cov_inv = np.linalg.inv(self.cov)
            bot = np.linalg.multi_dot([(self.mu - ones * self.rf).T, cov_inv, (self.mu - ones * self.rf)])
            top = (mean_constraint - self.rf) * np.dot(cov_inv, (self.mu - ones * self.rf))
            res = top/bot
        else:

            cons = [{'type': 'eq', 'fun': lambda x: np.dot(x.T, self.mu) + (1 - sum(x)) * self.rf - mean_constraint}]
            res = sco.minimize(self.portfolio_var, x0=self.equal_weights(), method='SLSQP',
                               bounds=self.get_bounds(allow_short_selling=allow_short_selling),
                               constraints=cons, tol=1e-15)
            res = res.x[:, np.newaxis]

        return res

    def optimize_mean_weights(self, vol_constraint: float, allow_short_selling: bool = True) -> np.ndarray:

        """
        computes the maximum return given a volatility constraint

        :param vol_constraint: constraint for the volatility
        :param allow_short_selling: boolean variable false or true
        :return: a set of weights
        """

        var_constraint = pow(vol_constraint, 2)
        cons = [{'type': 'eq', 'fun': lambda x: np.linalg.multi_dot([x.T, self.cov, x]) - var_constraint}]

        res = sco.minimize(self.negative_portfolio_return, x0=self.equal_weights(), method='SLSQP',
                           bounds=self.get_bounds(allow_short_selling), constraints=cons, tol=1e-15)

        res = res.x[:, np.newaxis]
        return res


class KellyPortfolio(SharpePortfolio):

    def __init__(self, data, rf: float):
        super().__init__(data, rf)
    def kelly_criterion(self, w: np.ndarray):

        return self.rf + self.portfolio_excess_return(w, self.rf) - self.portfolio_var(w) / 2

    def estimate_model(self, optimization_type: str = "UNCONSTRAINED_MEAN_OR_VOL", constraint: float = 0.0,
                       allow_short_selling: bool = True, allow_leverage: bool = True, out: bool = False) -> None:


        if optimization_type == 'VOLCONSTRAINT':
            additional_cons = {'type': 'eq', 'fun': lambda x: np.linalg.multi_dot([x.T, self.cov, x]) - constraint}
        elif optimization_type == 'MEANCONSTRAINT':
            additional_cons = {'type': 'eq', 'fun': lambda x: np.dot(x.T, self.mu) + (1 - sum(x)) * self.rf - constraint}
        elif optimization_type == 'UNCONSTRAINED_MEAN_OR_VOL':
            additional_cons = None
        else:
            raise Exception('optimization_type can only be VOLCONSTRAINT, MEANCONSTRAINT, UNCONSTRAINED_MEAN_OR_VOL')

        self.weights = self.optimize_kelly_criterion(allow_short_selling, allow_leverage, additional_cons)
        self.mu_p, self.vol_p = self.compute_mean_vol(w=self.weights)
        self.fitted = True

        if out:
            return self.weights

    def optimize_kelly_criterion(self, allow_short_selling: bool = True, allow_leverage: bool = True, additional_constraint: dict = None):
        if allow_leverage and allow_short_selling:
            w = np.linalg.inv(self.cov).dot(self.mu - self.rf)
            return w
        else:
            cons = [{'type': 'eq', 'fun': lambda x: sum(x) - 1}]
            if additional_constraint is not None:
                cons.append(additional_constraint)
            func = lambda x: -1 * self.kelly_criterion(x)
            res = sco.minimize(func, x0=self.equal_weights(), method='SLSQP', constraints=cons,
                               bounds=self.get_bounds(allow_short_selling, allow_leverage), tol=1e-15)
            w = res.x[:, np.newaxis]
            return w




