import pandas as pd
import numpy as np

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
        self.asset_cols = [_ for _ in data.columns if _ != date_col]
        # to do: to be datetime
        # self.data[date_col] = self.data[date_col].apply(lambda x: )
        self.data[self.asset_cols] = self.data[self.asset_cols].astype(float)

    def get_returns(self, period: int, out: bool = True):
            
        # to do: can be sped up with numpy only
        T = len(self.data)
        self.returns = {}
        for t in np.arange(0, T-period, period):
            self.returns[self.data.loc[t+period,self.date_col]] = np.log(self.data.loc[t+period,self.asset_cols].values / self.data.loc[t,self.asset_cols].values)
        self.returns = pd.DataFrame.from_dict(self.returns, orient='index')
        self.returns.columns = self.data.columns[1:]
        if period == 12:
            assert np.isclose(self.returns.iloc[0,0],np.log(79.848712 / 72.458651))

        if out:
            return self.returns.copy()



