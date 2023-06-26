from models.unsupervised.pca import PcaHandler
import pandas as pd
import numpy as np
from models.stat_models.linearregression import LinearRegressionModel
import matplotlib.pyplot as plt

class RollingModel:

    def __init__(self, data: pd.DataFrame, rolling_window: int, demean: bool = True):
        """
        This is a rolling window mean model. at each rolling window, [t, t+n], you estimate PCA (or some unsupervised
         factor model) and you run OLS (or some estimation), but (very important), you only save the residual e_{t+n},
         and repeat.
        :param data: pd.DataFrame, first column should be dates, rest should be returns (floats)
        :param rolling_window: int
        :param demean: bool, whether to demean the data
        """

        if isinstance(data, pd.DataFrame):
            self.index = data.index.values
            self.raw_data = data.iloc[:, 1:].astype(float).values
            self.assets = data.columns[1:]
        elif isinstance(data, np.array):
            self.index = np.arange(len(data))
            self.raw_data = data
            self.assets = np.arange(len(data))
        else:
            raise UserWarning('Data must be either np.array or pd.DataFrame')

        if demean:
            self.x = self.raw_data - self.raw_data.mean(axis=0)
        else:
            self.x = self.raw_data
        self.n, self.m = self.x.shape

        self.rolling_window = rolling_window

    def estimate(self, config: dict, RIC: str = None):

        """
        :param config: dictionary with configurations
        example:
        config = {
            'PCA': True,
            'n_components': 7,
            'OLS': True
            }
        :param RIC: if int, it will estimate for the int-indexed asset, if str it will estimate for the RIC
        :return:
        """

        if isinstance(RIC, int):
            idx = RIC
        else:
            idx = np.where(self.assets == RIC)[0][0]

        self.y = self.x[:, idx]
        if config['PCA']:
            self.singular_values = {}
            self.eig_vals = {}
            self.eig_vecs = {}

        self.size_of_rolling_windows = []
        i = 0
        # good to know this upfront, no time now
        errors = []
        all_errors = {}
        for i in range(self.n):
            print(i)
            try:
                x = self.x[:i+self.rolling_window, :]
                y = self.y[:i+self.rolling_window]
                date = self.index[i+self.rolling_window]
                self.size_of_rolling_windows.append(len(x))
                i += 1
                if config['PCA']:
                    pca_model = PcaHandler(x)
                    self.singular_values[date] = pca_model.singular_values
                    self.eig_vals[date] = pca_model.eig_vals
                    self.eig_vecs[date] = pca_model.eig_vecs

                    X = pca_model.components(config['n_components'])

                if config['OLS']:
                    ols = LinearRegressionModel(X, y)
                    errors.append(ols.residuals[-1][0])
                    all_errors[date] = ols.residuals.flatten()
            except IndexError:
                continue

        self._y = self.y[:self.n-self.rolling_window]
        self.errors = np.array(errors)
        ss_res = np.dot(self.errors.T, self.errors)
        ss_tot = np.dot((self._y - np.mean(self._y)).T, self._y - np.mean(self._y))
        self.r_sqr = float(1 - ss_res/ss_tot)
        self.all_errors = pd.DataFrame.from_dict(all_errors, orient='index').T


    def plot_resids(self):

        r2 = round(self.r_sqr, 5)
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
        ax1.plot(self.errors, label='resids', color="black", linewidth=2)
        ax1.tick_params(labelrotation=45)
        ax1.set_title("Scatter Plot Predictions $r^{2} = $" + str(r2))
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Predictions")
        ax1.grid()
        ax1.legend()

        fig.show()

