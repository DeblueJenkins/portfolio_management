import numpy as np
from scipy.stats import t
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Union
import warnings
from dataclasses import dataclass, field
from abc import ABC, abstractmethod, abstractproperty
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.robust.norms import *


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


@dataclass
class GeneralizedLinearModel(ABC):
    x: np.ndarray
    y: np.ndarray
    add_intercept: bool = True
    method: str = 'ols'
    conf: float = 0.95


    def __post_init__(self):

        if self.add_intercept:
            self.x = np.insert(self.x, 0, 1, axis=1)

        try:
            self.n_y1, self.n_y2 = np.shape(self.y)
        except:
            self.n_y1 = len(self.y)
            self.n_y2 = 1

        self.n_x1, self.n_x2 = np.shape(self.x)

        if self.n_x1 == self.n_y1:
            self.x = np.array(self.x).reshape(self.n_x1, self.n_x2)
        elif self.n_x2 == self.n_y1:
            self.x = np.array(self.x).reshape(self.n_x2, self.n_x1)
        else:
            raise Exception("X and y are not correct dimensions")

        self.assets = self.y.columns
        self.dates = self.y.index.values
        self.y = self.y.to_numpy()

        self.betas = np.ndarray([self.n_x2, self.n_y2])
        self.residuals = np.ndarray([self.n_x1, self.n_y2])
        self.pvalues = np.ndarray([self.n_x2, self.n_y2])
        self._reshape_y()

    def _reshape_y(self):
        self.y = self.y.reshape(self.n_y1, self.n_y2)

    @abstractmethod
    def fit(self) -> None:
        """
        Must set parameters self.betas
        :return:
        """
        pass

    def get_errors(self, out: bool = False) -> Union[None, np.ndarray]:
        self.residuals = np.dot(self.x, self.betas) - self.y
        if out:
            return self.residuals

    def get_sigma(self) -> float:
        if len(self.residuals) == 0:
            self.get_errors()
        return np.sqrt(np.dot(self.residuals.T, self.residuals) / (len(self.y) - len(self.betas) + 1))

    def diagnostics(self, out=False):
        self.get_errors()
        self.get_sigma()
        self.mse = self.compute_mse()
        self.r2, self.r2_adj = self.get_r_square()
        if out:
            return {'mse': self.mse, 'r2': self.r2, 'r2_adj': self.r2_adj}

    def compute_mse(self) -> float:
        """
        computes the MSE based on a beta vector (can be used to compare regularised vs non-regularised estimators)
        :return: mse
        """
        return (self.residuals ** 2).mean(axis=0)
    def get_r_square(self) -> float:
        """
        coefficient of determination % explained variance in sample
        :return: float
        """
        # ss_res = np.dot(self.residuals.T, self.residuals)
        ss_res = (self.residuals ** 2).sum(axis=0)
        # ss_tot = np.dot((self.y - np.mean(self.y)).T, self.y - np.mean(self.y))
        ss_tot = ((self.y - np.mean(self.y)) ** 2).sum(axis=0)

        r2 = 1 - ss_res/ss_tot
        r2_adj = 1 - (ss_res/self.df_res) / (ss_tot/self.df_total)
        return r2, r2_adj


class LinearRegressionModel(GeneralizedLinearModel):

    def __init__(self, *args, l1: float = 0, w: np.ndarray = None, **kwargs):
        super().__init__(*args, **kwargs)

        if self.method == 'ols':
            self.p = self.x.shape[1] - 1
        elif self.method == 'ridge':
            self.p = self.x.shape[1] - 2

        if w is None:
            w = np.repeat(1, self.n_y1)
        if w.ndim != 1:
            raise Exception('weights must be provided as a vector')

        self.W = np.diag(w)
        self.l1 = l1
        self.df_res = self.n_y1 - self.p
        self.df_total = self.n_y1 - 1
        self._reshape_y()
        self.residuals = np.array([])
        self.betas = np.array([])

        ## TO DO:
        # self.betas, self.residuals, self.mse should be setter/getters


    def fit(self, out: bool = False) -> np.ndarray:
        """
        computes the standard ols estimator with regularisation
        :param reg_param:
        :return: beta
        """
        xTx = self.x.T.dot(self.W).dot(self.x)
        reg_mat = xTx + self.l1 * np.ones(len(xTx))
        self.betas = np.linalg.multi_dot([np.linalg.inv(reg_mat), self.x.T, self.W, self.y])
        if out:
            return self.betas

    def plot(self) -> None:
        """
        plots the real values vs the predictions

        :return: none
        """
        predictions = np.dot(self.x, self.betas)
        r2, r2_adj = self.get_r_square()
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
        ax1.scatter(self.y, predictions, label='scatter', color="black", linewidth=2)
        ax1.tick_params(labelrotation=45)
        ax1.set_title("Scatter Plot Predictions $r^{2} = $" + str(r2))
        ax1.set_xlabel("Real Valued")
        ax1.set_ylabel("Predictions")
        ax1.grid()
        ax1.legend()

        fig.show()

class MultiOutputLinearRegressionModel(LinearRegressionModel):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        """
        Linear Regression with multiple y variables

        :param x: matrix of explanatory variables
        :param y: vector of dependent variables
        :param tickers: list of ticker names
        """

    def _set_tickers(self, tickers: List):
        if (isinstance(tickers, list)) and (self.n_y2 == len(tickers)):
            return np.array(tickers)

    def plot_single(self, name: str, tickers: List) -> None:
        """
        plots the summary of a single regression based on its column position

        :param index: integer for the index
        :param name: possible name for the plot
        :return: none
        """

        tickers = self._set_tickers(tickers)
        index = np.where(tickers == name)[0][0]

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

class IterativelyWeightedLeastSquares(GeneralizedLinearModel):

    def __init__(self, x, y, add_intercept: bool = True, **kwargs):

        super().__init__(x, y, add_intercept)

        self.n_y2 = np.shape(self.y)[1]
        self.models = [None] * self.n_y2


        for m in range(self.n_y2):
            self.models[m] = RLM(exog=self.x,
                                 endog=self.y[:,m],
                                      **kwargs)

        self.df_res = self.models[m].df_resid
        self.df_total = self.models[m].df_model

    def fit(self, *args, **kwargs):

        self.results = [None] * self.n_y2

        for m in range(self.n_y2):

            self.results[m] = self.models[m].fit(*args, **kwargs)
            self.betas[:, m] = self.results[m].params
            self.pvalues[:, m] = self.results[m].pvalues.round(4)


