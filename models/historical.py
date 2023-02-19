from abc import abstractmethod
import numpy as np
from portfolio_management.models.data.variables import PriceVariables


class Model:

    def __init__(self, ret: np.array):
        self.r = ret
        self.n_times, self.n_assets = ret.shape
        self.weights = np.zeros(self.n_assets)

    @abstractmethod
    def get_expected_value(self):
        pass

    @abstractmethod
    def get_variance(self):
        pass

    @abstractmethod
    def get_covariance(self):
        pass


class Historical(Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.get_variance()
        self.get_covariance()
        self.get_expected_value()

    def get_expected_value(self):
        self.mu = np.mean(self.r, axis=0)
        return self.mu


    def get_variance(self):
        self.sigma = np.std(self.r, axis=0)
        return self.sigma

    def get_covariance(self):
        self.cov = np.cov(self.r.T)
        self.corr = np.corrcoef(self.r.T)
        return self.cov



class Stochastic(Model):

    pass



