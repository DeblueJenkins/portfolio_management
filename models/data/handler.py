import pandas as pd
import numpy as np
from models.unsupervised.pca import PcaHandler, R2Pca
import yaml
import warnings

class DataHandler:

    """
    This class should handle the data. It should load from table and do merges and joins as needed.
    """

    def __init__(self, data: pd.DataFrame, data_rates: pd.Series, date_col: str, horizon: int):

        """
        :param data: pd.DataFrame, this needs to be a dataframe indexed by time and assets
        :param date_col: str, time column
        """

        self.data = data
        self.rates = data_rates
        self.horizon = horizon
        self.get_horizon_adjusted_rate()
        self.data.set_index(date_col, inplace=True)
        # self.asset_cols = [_ for _ in data.columns if _ != date_col]
        # to do: to be datetime
        # self.data[date_col] = self.data[date_col].apply(lambda x: )
        # also; converison to float is not working here
        # self.data[self.asset_cols] = self.data[self.asset_cols].astype({_:'float' for _ in self.asset_cols })
        self.data = self.data.astype(float)
        self.date_col = date_col

    def get_horizon_adjusted_rate(self):
        # rates data is daily overnight, so they need to be transformed to
        # the same time unit as are the returns calculcated; since returns are calculated
        # based on the investment horizon, we will also do so for the rates
        # convert dailies
        self.rates = self.rates / 365
        self.rates = (1 + self.rates) ** self.horizon - 1




    def get_excess_returns(self, period: int, out: bool = True):

        rets = self.get_returns(period, out=True)

        warnings.warn('FedRates are joined on random benchmark (whichever comes first). Further filtering is needed!')
        rets = rets.join(self.rates, how='left')
        excess_rets = pd.DataFrame(index=rets.index,
                                   columns=rets.columns[:-1],
                                   data=np.subtract(rets.iloc[:, :-1].values, rets.iloc[:, -1].values[:, np.newaxis]))

        return excess_rets

    def get_returns(self, period: int, out: bool = True):
            
        # to do: can be sped up with numpy only
        T = len(self.data)
        self.returns = {}
        for t in np.arange(0, T-period, period):

            self.returns[self.data.index[t+period]] = np.log(self.data.iloc[t+period,:].astype(float).values / self.data.iloc[t,:].astype(float).values)
        self.returns = pd.DataFrame.from_dict(self.returns, orient='index')
        self.returns.columns = self.data.columns

        if out:
            return self.returns.copy()

    def get_overlapping_returns(self, period: int, out: bool = True):
        self.returns = np.log(self.data / self.data.shift(period))
        self.returns.dropna(axis=0, how='all', inplace=True)
        if out:
            return self.returns.copy()


    def get_pca_factors(self, n_components: int, data: np.ndarray = None, method: str ='ordinary'):
        if data is None:
            if hasattr(self, 'returns'):
                data = self.returns
            else:
                raise Exception('If data is not provided, get_returns() need to be run first!')

        if method == 'ordinary':

            self._pca_model = PcaHandler(data, demean=True)
        elif method == 'r2pca':
            pass
        else:
            raise Exception('method is not implemented yet')

        self.factors = self._pca_model.components(n_components).copy()
        self.factor_loading_estimator = np.multiply(self._pca_model.eig_vals, self._pca_model.eig_vecs)
        return self.factors, self._pca_model.eig_vals[:n_components]



