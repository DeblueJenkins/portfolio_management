import pandas as pd
import numpy as np
from models.unsupervised.pca import PcaHandler, R2Pca
import yaml
class DataHandler:

    """
    This class should handle the data. It should load from table and do merges and joins as needed.
    """

    def __init__(self, data: pd.DataFrame, date_col: str):

        """
        :param data: pd.DataFrame, this needs to be a dataframe indexed by time and assets
        :param date_col: str, time column
        """

        self.data = data
        self.data.set_index(date_col, inplace=True)
        # self.asset_cols = [_ for _ in data.columns if _ != date_col]
        # to do: to be datetime
        # self.data[date_col] = self.data[date_col].apply(lambda x: )
        # also; converison to float is not working here
        # self.data[self.asset_cols] = self.data[self.asset_cols].astype({_:'float' for _ in self.asset_cols })
        self.data = self.data.astype(float)
        self.date_col = date_col


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

        factors = self._pca_model.components(n_components).copy()
        return factors



