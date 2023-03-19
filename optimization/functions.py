import numpy as np
import warnings
from scipy.optimize import minimize



def get_A(sigma_inverse):
    return sigma_inverse.sum().sum()

def get_B(sigma_inverse, mu):
    return sigma_inverse.sum(axis=1) @ mu

def get_C(sigma_inverse, mu):
    return mu.T @ sigma_inverse @ mu

def get_tangency_w(sigma, mu, r):
    sigma_inverted = np.linalg.inv(sigma)
    A = get_A(sigma_inverted)
    B = get_B(sigma_inverted, mu)
    w = (sigma_inverted @ (mu - r)) / (B - A*r)
    return w

def get_mean_variance_weights_analytically(mu, sigma, m):

    n = len(mu)

    sigma1 = np.linalg.inv(sigma)
    A = np.ones(n) @ sigma1 @ np.ones(n)
    B = mu @ sigma1 @ np.ones(n)
    C = mu @ sigma1 @ mu

    l = (A * m - B) / (A * C - B ** 2)
    g = (C - B * m) / (A * C - B ** 2)

    w = sigma1 @ (l * mu + g * np.ones(n))

    return w

mu_test = np.array([0.05, 0.07, 0.15, 0.27])
m_test = 0.1
# mu_test = mu_test.reshape(n_test,1)
sigma_test = np.array([[0.0049, 0.00672, 0.0105, 0.0168],
                       [0.00672, 0.0144, 0.0252, 0.036],
                       [0.0105, 0.0252, 0.09, 0.144],
                       [0.0168, 0.036, 0.144, 0.36]])

w_test = get_mean_variance_weights_analytically(mu_test, sigma_test, m_test)
assert np.sum(np.isclose(w_test, np.array([0.528412108,
                                    0.172888075,
                                    0.159764343,
                                    0.138935474,])))

def mean_variance_target(mu: np.ndarray, sigma: np.ndarray, m: np.float, method: str = 'nummerical', l: np.float = 2):
    """
    Classical mean-variance (Markowitz)
    :param mu:
    :param sigma: np.ndarray
    :param rf: np.float, risk-free rate
    :param m: np.float, target return
    :param method: str, nummerical or analytical
    :param l: risk-aversion, l > 0
    :return:
    """

    assert l > 0




    if method == 'analytic':

        if sigma is None:
            raise UserWarning('User must provide var-cov in matirx form')
        if not isinstance(sigma, np.ndarray):
            raise UserWarning('Sigma is covariance matrix, it needs to be np.array')
        if np.isclose(np.linalg.det(sigma), 0):
            warnings.warn('Covariance matrix close to singular, trying nummerical optimization', UserWarning)


        w = get_mean_variance_weights_analytically(mu, sigma, m)


        return w

    elif method == 'nummerical':

        n = len(mu)
        f = lambda w: w.T @ sigma @ w
        cons1 = lambda w: w.T @ mu - m
        cons2 = lambda w: w.T @ np.ones(n) - np.ones(n)
        cons = [{'type': 'eq'}, {'fun': cons1},
                {'type': 'eq'}, {'fun': cons2},]

        return f, cons


def sharpe_ratio(w: np.array, mu: np.array, sigma: np.ndarray, rf: np.array, minimize=True):
    """

    :param mu: np.array, expected value returns
    :param sigma: np. array, variance
    :param rf: np.array, risk-free rate
    :return: np.array, sharpe ratio
    """
    if minimize:
        return -1 * (w @ mu - rf) ** 2 / (w.T @ sigma @ w)
    else:
        return (w @ mu - rf) / np.sqrt((w.T @ sigma @ w))

def trenor_ratio(mu: np.array, beta: np.array, rf: np.array):
    """

    :param mu: np.array, expected value returns
    :param sigma: np. array, variance
    :param rf: np.array, risk-free rate
    :return: np.array, sharpe ratio
    """

    return (mu - rf) / beta

def sortino_ratio(mu: np.array, target: np.array, rf: np.array):
    """
    The ratio S is calculated as

        S = R − T D R S={\frac {R-T}{DR}} ,

    where R R is the asset or portfolio average realized return, T T is the target or required rate of return for the
    investment strategy under consideration (originally called the minimum acceptable return MAR), and D R DR is the
    target semi-deviation (the square root of target semi-variance), termed downside deviation. D R DR is expressed
    in percentages and therefore allows for rankings in the same way as standard deviation.

    :param mu: np.array, expected or realized returns
    :param sigma: np. array, variance
    :param rf: np.array, risk-free rate
    :return: np.array, sortino ratio
    """

    pass