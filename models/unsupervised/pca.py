import matplotlib.pyplot as plt
import numpy as np
import warnings
import pandas as pd

def get_svd(x):
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

class PcaHandler:

    def __init__(self, data: pd.DataFrame, demean: bool = True):
        """
        Principle Component Object
        :param data: pd.DataFrame,
        then first column needs to be dates (str or datetime), rest needs to be prices (floats)
        :param demean: boolean, whether to demean it or not; will work also with numpy array but dates
        will be lost
        """
        if isinstance(data, pd.DataFrame):
            self.index = np.array(data.index)
            self.raw_data = data.astype(float).values
        elif isinstance(data, np.ndarray):
            self.index = np.arange(len(data))
            self.raw_data = data
        else:
            raise UserWarning('Data must be either np.array or pd.DataFrame')

        if demean:
            self.x = self.raw_data - self.raw_data.mean(axis=0)
        else:
            self.x = self.raw_data
        self.n, self.m = self.x.shape
        self._get_svd()

    def _get_svd(self):
        self.singular_values, self.eig_vals, self.eig_vecs = get_svd(self.x)

    def benchmark_test(self) -> None:

        """
        Compares the PCA covariance to the empirical covariance.
        """

        self.cov_data = np.cov(self.x, rowvar=False)
        self.cov_pca = np.linalg.multi_dot([self.eig_vecs, np.diag(self.eig_vals), self.eig_vecs.T])
        print(sum(self.cov_pca - self.cov_data))

    def components(self, n: int) -> np.array:
        """
        Computes principle components
        :param n: number of factors
        :return: n pca vectors factor
        """
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










