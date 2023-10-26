import matplotlib.pyplot as plt
import numpy as np
import warnings
import pandas as pd
from abc import ABC, abstractmethod


class PcaBaseClass(ABC):

    @staticmethod
    def get_svd(x: np.array) -> (np.array, np.array, np.array):
        """
        static method used in Pca model only, applied in both pca models.
        :param x: A matrix of data

        :return: singular_values, eig_vals, eig_vecs
        """
        u, s, v = np.linalg.svd(x)
        n = x.shape[0]
        singular_values = s

        stability_condition_1 = np.any(singular_values == 0)
        stability_condition_2 = len(singular_values) != len(np.unique(singular_values))

        if stability_condition_1 or stability_condition_2:
            warnings.warn('Possible instability in the SVD')

        eig_vecs = v.T
        eig_vals = np.power(s, 2) / (n - 1)

        return singular_values, eig_vals, eig_vecs

    def __init__(self, data, demean: bool) -> None:

        """
        Initiate the abstract base pca class. This class is never used it, it holds all methods that are used in both models
        :param data: matrix of data (free of type can be a data frame or numpy array)
        :param demean: boolean variable True demeans the data, when false it will not
        """

        self.index = np.array([])
        self.raw_data = np.array([])
        self.demean = demean
        self.x = np.array([])
        self.n: int = 0
        self.m: int = 0
        self.assets: list = []

        self.handle_input_data(data=data)

    def handle_input_data(self, data) -> None:
        """
        input data handling, ensures that all initial parameters are correctly assigned
        :param data: matrix of data (free of type can be a data frame or numpy array)
        :return: nothing all data is set though
        """

        if isinstance(data, pd.DataFrame):
            self.index = np.array(data.index)
            self.assets = list(data.columns[1:])
            self.raw_data = data.astype(float).values
        elif isinstance(data, np.ndarray):
            self.index = np.arange(len(data))
            self.raw_data = data
            self.assets = np.arange(len(data))
        else:
            raise UserWarning('Data must be either np.array or pd.DataFrame')

        if self.demean:
            self.x = self.raw_data - self.raw_data.mean(axis=0)
        else:
            self.x = self.raw_data

        self.n, self.m = self.x.shape

    @abstractmethod
    def components(self, n: int):
        pass


class PcaHandler(PcaBaseClass):

    def __init__(self, X, demean: bool = True, method : str = 'svd', covariance: str = 'mle'):
        """
        Principle Component Object
        :param data: matrix of data (free of type can be a data frame or numpy array)
        :param demean: boolean variable True demeans the data, when false it will not

        """
        if method == 'svd':
            if X.shape[0] == X.shape[1]:
                raise Exception('When method is svd, please provide rectangular matrix (data)')
        elif method == 'pca':
            if X.shape[0] != X.shape[1]:
                raise Exception('When method is pca, please provide square matrix (covariance)')
            if np.sum(X != X.T):
                raise Exception('Covariance matrix must be symmetric')
        else:
            raise Exception('Method must be either svd (on data) or pca (on covariance matrix)')

        super().__init__(data=X, demean=demean)
        self.method = method
        if self.method == 'svd':
            self.cov_data = np.cov(self.x.T)
        else:
            if covariance == 'mle':
                self.cov_data = np.cov(self.raw_data.T)
            else:
                raise Exception('only mle covariance is supported')
        self.cov_pca = np.array([])
        if method == 'svd':
            self._get_svd()
        elif method == 'pca':
            self._get_pca()

    def _get_pca(self):
        _, s, v = np.linalg.svd(self.raw_data)
        self.eig_vals = s
        self.eig_vecs = v.T

    def _get_svd(self) -> None:
        self.singular_values, self.eig_vals, self.eig_vecs = self.get_svd(self.x)

    def benchmark_test(self) -> None:

        """
        Compares the PCA covariance to the empirical covariance.

        """
        self.cov_pca = np.linalg.multi_dot([self.eig_vecs, np.diag(self.eig_vals), self.eig_vecs.T])
        print(sum(self.cov_pca - self.cov_data))

    def components(self, n: int, data: np.ndarray = None) -> np.array:
        """
        Computes principle components

        :param n: number of factors
        :return: n pca vectors factor
        """
        if self.method == 'pca':
            if data is None:
                raise Exception('Data matrix must be provided when method is svd')


        return np.dot(self.x, self.eig_vecs[:, :n])

    def plot(self, n: int) -> None:
        """
        plots the variance explained per component.
        :param n: number of factors
        :return:
        """

        explain_ratios = np.round(self.eig_vals/sum(self.eig_vals), 3)[0:n]
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
        ax1.bar(np.arange(0, n), explain_ratios, color="blue", align="center", alpha=0.5, edgecolor="black")
        ax1.set_ylabel("explained variance")
        ax1.grid()
        ax1.set_title("Scree Plot")


class R2Pca(PcaBaseClass):

    def __init__(self, data, rolling_window: int, demean: bool = True):
        """
        R2Pca model from Robust Rolling PCA: Managing Time Series and Multiple Dimensions 2023, March 25

        :param data: matrix of data (free of type can be a data frame or numpy array)
        :param demean: boolean variable True demeans the data, when false it will not
        """

        self.eigenvector_dict = {}
        self.rolling_window = rolling_window

        super().__init__(data=data, demean=demean)

    def components(self, n: int) -> np.array:
        """
        Computes principle components

        :param n: number of factors
        :return: n pca vectors factor
        """

        x = self.x[:self.rolling_window, :]
        singular_values, eig_vals, eig_vecs = self.get_svd(x=x)
        v_t_pre = eig_vecs[:, :n]
        components = np.dot(x, v_t_pre)[-1, :].reshape((1, n))

        date = self.index[self.rolling_window - 1]

        self.eigenvector_dict[date] = v_t_pre
        i = 1

        while i + self.rolling_window <= self.n:

            x = self.x[:i+self.rolling_window, :]
            singular_values, eig_vals, eig_vecs = self.get_svd(x=x)
            v_t = eig_vecs[:, :n]
            date = self.index[self.rolling_window + i - 1]
            self.eigenvector_dict[date] = v_t

            order = []
            for j in range(n):
                j_max = np.argmax(abs(np.dot(v_t_pre.T, v_t[:, j])))
                order.append(j_max)
                if np.dot(v_t_pre[:, j], v_t[:, j_max]) < 0:
                    v_t[:, j_max] = -v_t[:, j_max]

            v_t = v_t[:, order]
            new_components = np.dot(x, v_t)
            new_row = new_components[-1, :].reshape((1, n))
            components = np.concatenate((components, new_row))
            v_t_pre = v_t
            i += 1

        return components
