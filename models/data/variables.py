import pandas as pd
import numpy as np


class Variable():

    def __init__(self, start_date: pd.Timestamp, end_date: pd.Timestamp):
        """
        This should be a big data class. This class will build a big data pd.DataFrame which contains inputs for
        the models. As inputs, the interest_rates.py and stocks.py will be taken. It will be initiated per ticker/symbol.
        The main attribute is self.data, which is a dictionary containing either tables (pd.DataFrames) or other dictionaries
        (depending on the data structure).

        :param data: pd.DataFrame
        :param start_date:
        :param end_date:
        """

        self.start_date = start_date
        self.end_date = end_date

class PriceVariables(Variable):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_price_variables(self, data: pd.DataFrame,
                            adjusted_close: str = 'AdjClose',
                            close: str = 'Close',
                            high: str = 'High',
                            low: str = 'Low',
                            open: str = 'Open'):

        if not isinstance(data.index.values[0], (pd.Timestamp, np.datetime64)):
            raise UserWarning('Data index must be pd.Timestamp or np.datetime64')

        for col in [adjusted_close, close, high, low, open]:
            if col not in data.columns:
                raise UserWarning(f'Data provided is missing {col}')

        data = data.loc[self.start_date:self.end_date, :]

        """ 
        :param data: 
        :param adjusted_close: 
        :param close: 
        :param high: 
        :param low: 
        :param open: 
        :return: 
        """

        data.loc[:, 'adjusted_returns'] = np.log(data.loc[:, adjusted_close].astype(float) / data.loc[:, adjusted_close].shift(1).astype(float))
        data.loc[:, 'returns'] = np.log(data.loc[:, close].astype(float) / data.loc[:, close].shift(1).astype(float))
        data.loc[:, 'high_minus_low'] = data.loc[:, high].astype(float) - data.loc[:, low].astype(float)
        data.loc[:, 'close_minus_open'] = data.loc[:, close].astype(float) - data.loc[:, open].astype(float)

        self.big_data = data


