from models.unsupervised.pca import PcaHandler
import pandas as pd
import numpy as np
from models.stat_models.linearregression import LinearRegressionModel
import matplotlib.pyplot as plt
import multiprocessing as mp


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

        self.y = None
        self.singular_values = {}
        self.eig_vals = {}
        self.eig_vecs = {}

        self.size_of_rolling_windows = []

        self.rolling_window = rolling_window

        # indexes are date of estimation, columns are prediction dates
        self.errors_insample_all = pd.DataFrame()
        self.errors_outsample_all = pd.DataFrame()

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

        The algorithm can be described as follows:
            1. Estimate PCA in the window and retrieve desired number of PCs
            2. Estimate OLS but save only last residual (to avoid data leakage)
            3. Move window by one period, and repeat
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
            self.factors = {}

        self.size_of_rolling_windows = []
        # good to know this upfront
        errors_insample_last = []
        errors_insample_all = {}
        all_factor_loadings = {}
        for i in range(self.n):
            print(i)
            try:
                x = self.x[:i+self.rolling_window, :]
                y = self.y[:i+self.rolling_window]
                date_fit = self.index[i+self.rolling_window]
                self.size_of_rolling_windows.append(len(x))
                i += 1
                if config['PCA']:
                    pca_model = PcaHandler(x)
                    self.singular_values[date_fit] = pca_model.singular_values
                    self.eig_vals[date_fit] = pca_model.eig_vals
                    self.eig_vecs[date_fit] = pca_model.eig_vecs
                    X = pca_model.components(config['n_components'])
                    self.factors[date_fit] = X
                # this should be outside, independent of this loop
                if config['OLS']:
                    ols = LinearRegressionModel(X, y)
                    params = ols.beta.flatten()

                    # these are errors with no data leakage, so e_{t}|OLS(PC_{t-n,t}), e_{t+1}|OLS(PC_{t-n,t+1}), etc.
                    errors_insample_last.append(ols.residuals[-1][0])
                    # these are usual errors (with data leakage), so e_{t-n:t}|OLS(PC_{t-n,t}, e_{t-n:t+1}|OLS(PC_{t-n,t+1}
                    errors_insample_all[date_fit] = ols.residuals.flatten()
                    all_factor_loadings[date_fit] = params

            except IndexError:
                continue

        self.factor_loadings = pd.DataFrame.from_dict(all_factor_loadings, orient='index')
        self.errors_insample_all = pd.DataFrame.from_dict(errors_insample_all, orient='index')
        self.errors_insample_all.columns = self.index[:-1]
        self._get_out_of_sample_errors()

        self._y = self.y[:self.n-self.rolling_window]
        self.errors_insample_last = np.array(errors_insample_last)
        self.errors_insample_last = pd.Series(data=self.errors_insample_last, index=self.index[self.rolling_window:])
        ss_res = np.dot(self.errors_insample_last.T, self.errors_insample_last)
        ss_tot = np.dot((self._y - np.mean(self._y)).T, self._y - np.mean(self._y))
        df_res = len(self.errors_insample_last) - self.m
        df_tot = len(self.errors_insample_last) - 1
        self.r_sqr = float(1 - ss_res/ss_tot)
        self.r_sqr_adj = float(1 - (ss_res/df_res)/(ss_tot/df_tot))

    def _get_out_of_sample_errors(self):
        """
        A out of sample backtesting algorithm. Slightly complex algorithm due to the rolling nature of testing.
        The idea is to replicate production behaviour. Assume we have data x_t(0) to x_t(N). At time t(i), we estimate
        the model on x_t(0) to x_t(i) (in case of PCA, also get factors f_t(0) to f_t(i)), and we assume the factor
        loadings remain constant through time going forward. When we move to t(i+1), we now have data (factors) f_t(0):f_t(i+1),
         but we have the factor loadings from t(i), b_t_(i). So, we can estimate our prediction as f_t_(i+1) * b_t_(i) to get the error,
        e_t_(i+1). Keeping our estimation date fixed at t(i), we can then move forward to t(i+2), t(i+3) until we run out of data.
        Then, we move our estimation date to t(i+1), we repeat the process. The outcome of this algorithm is a matrix
        containing as indices the estimation dates, and as columns the prediction dates.

        Apart from the fact that this is the only proper way of getting out of sample errors, another goal of this
        algo is to find out the right rebalancing period, i.e., what is j such that E[x_t(j)|model] = 0.

        :return: pd.DataFrame
        """
        out_sample_errors = {}
        n_estimations = len(self.factor_loadings)
        for i in range(n_estimations):
            out_of_sample_at_one_estimation = {}
            date_estimation = self.factor_loadings.index[i]
            dates_after_estimation = self.factor_loadings.index[i+1:].values
            factor_loadings = self.factor_loadings.loc[date_estimation].values
            for j in range(len(dates_after_estimation)):
                date_next = dates_after_estimation[j]
                n_data_used, n_factors = self.factors[date_next].shape
                intercept = np.ones(n_data_used).reshape(n_data_used, 1)
                factors = np.concatenate([intercept, self.factors[date_next]], axis=1)
                y_next = self.y[np.where(self.index==date_next)][0]
                # take only last
                y_pred = factors[-1,:].dot(factor_loadings)
                out_of_sample_at_one_estimation[date_next] = y_next - y_pred
            out_sample_errors[date_estimation] = out_of_sample_at_one_estimation
        # index
        self.out_sample_errors = pd.DataFrame.from_dict(out_sample_errors).T
    def get_out_of_sample_error_k_period_ahead(self, k):
        """
        In out of sample matrix, for ith row take (i+k)th column
        :param n:
        :return:
        """
        if self.out_sample_errors is None:
            self._get_out_of_sample_errors()
        n,m = self.out_sample_errors.shape
        errors = []
        for i in range(n-k):
            errors.append(self.out_sample_errors.iloc[i,i+k])
        return errors

    def plot_resids(self):

        r2 = round(self.r_sqr, 5)
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
        ax1.plot(self.errors_insample_last, label='resids', color="black", linewidth=2)
        ax1.tick_params(labelrotation=45)
        ax1.set_title("Scatter Plot Predictions $r^{2} = $" + str(r2))
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Predictions")
        ax1.grid()
        ax1.legend()

        fig.show()

