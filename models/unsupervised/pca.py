import matplotlib.pyplot as plt
import numpy as np


class PcaHandler:

    def __init__(self, data: np.array):
        """
        Principle Component Object

        :param data: numpy array containing the explanatory variables
        """
        self.raw_data = data
        self.x = data - data.mean(axis=0)
        self.n = np.shape(self.x)[0]
        u, s, v = np.linalg.svd(self.x)
        self.singular_values = s
        self.eig_vecs = v.T
        self.eig_vals = np.power(s, 2) / (self.n - 1)

    def benchmark_test(self) -> None:
        """
        Compares the PCA covariance to the emperical covariance

        :return:
        """
        np_cov = np.cov(self.x, rowvar=False)
        pca_cov = np.linalg.multi_dot([self.eig_vecs, np.diag(self.eig_vals), self.eig_vecs.T])
        print(sum(pca_cov - np_cov))

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
