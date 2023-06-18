import numpy as np
from scipy.stats import t
import pandas as pd
import matplotlib.pyplot as plt


class LinearRegressionModel:

    def __init__(self, x: np.array, y: np.array, conf: float = 0.05, reg_param: float = None):
        """
        simple multi variate regression model

        :param x: matrix of explanatory variables
        :param y: vector of dependent variables
        :param conf: alpha for parameter tests
        :param reg_param: regularization parameter
        """
        y = np.array(y).reshape(len(y), 1)

        if np.shape(x)[0] == len(y):
            x = np.array(x).reshape(np.shape(x)[0], np.shape(x)[1])
        elif np.shape(x)[1] == len(y):
            x = np.array(x).reshape(np.shape(x)[1], np.shape(x)[0])
        else:
            print("X and y are not correct dimensions")

        x = np.insert(x, 0, 1, axis=1)
        self.x = x
        self.y = y
        self.n = len(self.y)
        self.conf = conf
        self.reg_param = reg_param
        self.xTx = np.dot(x.T, x)

        if reg_param is None:
            self.beta = self.ols()
        else:
            self.beta = self.ridged_ols(self.reg_param)

        self.residuals = np.dot(self.x, self.beta) - self.y
        self.sigma = np.sqrt(np.dot(self.residuals.T, self.residuals)/(len(y) - len(self.beta) + 1))
        self.mse = self.compute_mse(self.beta)

    def ols(self) -> np.array:
        """
        computes the standard ols estimator

        :return: beta
        """
        return np.linalg.multi_dot([np.linalg.inv(self.xTx), self.x.T, self.y])

    def ridged_ols(self, reg_param: float) -> np.array:
        """
        computes the standard ols estimator with regularisation

        :param reg_param:
        :return: beta
        """
        reg_mat = self.xTx + reg_param * np.ones(len(self.xTx))
        return np.linalg.multi_dot([np.linalg.inv(reg_mat), self.x.T, self.y])

    def compute_mse(self, beta: np.array) -> float:
        """
        computes the MSE based on a beta vector (can be used to compare regularised vs non-regularised estimators)

        :param beta: vector of parameters
        :return: mse
        """
        epsilon = np.dot(self.x, beta) - self.y
        return float(np.dot(epsilon.T, epsilon)/self.n)

    def compute_standard_errors(self) -> np.array:
        """
        computes the standard errors

        :return: np.array of standard errors per beta parameter
        """
        if self.reg_param is not None:
            print("Do not know whether this works for ridged regression")

        std_err = np.ones(shape=(np.shape(self.beta)))
        xtx_inv = np.linalg.inv(self.xTx)
        for i in range(len(self.beta)):
            std_err[i] = self.sigma * np.sqrt(xtx_inv[i][i])

        return std_err

    def t_tests(self) -> pd.DataFrame:
        """
        performs the t-tests for each parameter

        :return: pd.DataFrame of test results
        """
        std_err = self.compute_standard_errors()
        df = len(self.y) - len(self.beta) + 1
        top_quantile = self.beta + t.ppf(1-self.conf/2, df=df) * std_err
        bot_quantile = self.beta + t.ppf(self.conf/2, df=df) * std_err
        checks = []
        for a, b in zip(bot_quantile, top_quantile):
            if a <= 0 <= b:
                checks.append("Not Significant")
            else:
                checks.append("Significant")

        return pd.DataFrame({"bot quantile:": list(bot_quantile),
                             "top quantile:": list(top_quantile),
                             "Results": checks})

    def r_square(self) -> float:
        """
        coefficient of determination % explained variance in sample

        :return: float
        """
        ss_res = np.dot(self.residuals.T, self.residuals)
        ss_tot = np.dot((self.y - np.mean(self.y)).T, self.y - np.mean(self.y))
        return float(1 - ss_res/ss_tot)

    def plot(self) -> None:
        """
        plots the real values vs the predictions

        :return: none
        """
        predictions = np.dot(self.x, self.beta)
        r2 = round(self.r_square(), 5)
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
        ax1.scatter(self.y, predictions, label='scatter', color="black", linewidth=2)
        ax1.tick_params(labelrotation=45)
        ax1.set_title("Scatter Plot Predictions $r^{2} = $" + str(r2))
        ax1.set_xlabel("Real Valued")
        ax1.set_ylabel("Predictions")
        ax1.grid()
        ax1.legend()

        fig.show()


def generate_regression_data(betas: np.array, sigma: float, n: int = 100000):
    """

    :param betas: vector of betas
    :param sigma: variance of the outputs
    :param n: number of observations
    :return: matrix of generated x explanatory variables and a vector of y dependent variables.
    """
    b = betas[:, np.newaxis]
    m = len(b) - 1
    x = np.random.normal(0, 1, size=(n, m))
    x_r = np.c_[np.ones(n), x]
    y = np.dot(x_r, b) + np.random.normal(0, sigma, size=(n, 1))
    return x, y


class MultiOutputLinearRegressionModel:

    def __init__(self, x: np.array, y: np.array, tickers: list = None):
        """
        Linear Regression with multiple y variables

        :param x: matrix of explanatory variables
        :param y: vector of dependent variables
        :param tickers: list of ticker names
        """
        self.n = np.shape(y)[0]
        self.n_y_s = np.shape(y)[1]
        self.tickers = None
        self.set_tickers(tickers=tickers)

        if np.shape(x)[0] == self.n:
            x = np.array(x).reshape(np.shape(x)[0], np.shape(x)[1])
        elif np.shape(x)[1] == self.n:
            x = np.array(x).reshape(np.shape(x)[1], np.shape(x)[0])
        else:
            print("X and y are not correct dimensions")

        x = np.insert(x, 0, 1, axis=1)
        self.x = x
        self.y = y
        xtx_inv = np.linalg.inv(np.dot(self.x.T, self.x))
        self.betas = np.linalg.multi_dot([xtx_inv, self.x.T, self.y])

    def set_tickers(self, tickers: list):
        """
        sets the tickers

        :param tickers: list of tickers
        :return: none
        """
        if self.n_y_s == len(tickers):
            self.tickers = tickers
        else:
            self.tickers = None
            print("length tickers is: " + str(len(tickers)) + " number of assets is " + str(self.n_y_s))

    def plot_single(self, index: int, name: str = None) -> None:
        """
        plots the summary of a single regression based on its column position

        :param index: integer for the index
        :param name: possible name for the plot
        :return: none
        """
        if name is None:
            name = str(index)

        betas = self.betas[:, index].T
        y = self.y[:, index].T
        errors = np.dot(self.x, betas) - y
        ss_res = np.dot(errors.T, errors)
        ss_tot = np.dot((y - np.mean(y)).T, y - np.mean(y))
        r_square = 1 - ss_res/ss_tot

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.scatter(y, np.dot(self.x, betas), label='scatter', color="black", linewidth=2)
        ax1.tick_params(labelrotation=45)
        ax1.set_title("Scatter Plot Predictions for " + name + " ($r^{2} = $" + str(round(r_square, 5)) + ")")
        ax1.set_xlabel("Real Valued")
        ax1.set_ylabel("Predictions")
        ax1.grid()
        ax1.legend()

        ax2.plot(errors, label="regression errors", color="black", linewidth=2)
        ax2.tick_params(labelrotation=45)
        ax2.set_title("Time series regression errors " + name)
        ax2.set_xlabel("x")
        ax1.set_ylabel("y")
        ax2.grid()
        ax2.legend()

        fig.show()

    def plot_ticker(self, ticker: str):
        """
        plots the summary of a single regression based on its ticker

        :param ticker: string with the requested ticker
        :return: none
        """
        if self.tickers is not None:
            reg_index = self.tickers.index(ticker)
            self.plot_single(index=reg_index, name=ticker)
        else:
            print("first set tickers using set_tickers()")


