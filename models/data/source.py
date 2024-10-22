from abc import abstractmethod
import pandas as pd
import yfinance as yf
import warnings
import requests
from .mapper import MAPPER
import time
import eikon as ek
import datetime as dt
import os
import numpy as np
import warnings

warnings.filterwarnings("ignore")


def load_fed_rates_from_excel(path):

    def convert_dates(date):
        x = date.split('/')
        return f'{x[2]}-{x[0]}-{x[1]}'


    data = pd.read_csv(path)
    data.dropna(axis=0, how='all', inplace=True)
    data['Effective Date'] = data['Effective Date'].apply(lambda x: convert_dates(x))
    data = data.groupby('Effective Date').first().reset_index()
    data.set_index('Effective Date', inplace=True)
    data = data.loc[:, 'Rate (%)'] / 100
    data.name = 'FedRate'


    return data

def load_risk_free_from_ff_data(path):

    data = pd.read_csv(path)




class APIError(Exception):
    """An API Error Exception"""

    def __init__(self, response_message):
        self.msg = response_message

    def __str__(self):
        return "APIError: status={}".format(self.msg)

def _wrap_dt_parser(x):
    return dt.datetime.strptime(x, '%Y%m%d')

class FamaFrenchData:

    USE_COLS_MAPPER = {
        'Developed_5_Factors_Daily':
            {
                'factors': ['Mkt-RF', 'SMB', 'HML', 'CMA', 'RF'],
                'read_csv_params': {
                    'skiprows': 3,
                    'index_col': 0,
                },
            },
         'Developed_MOM_Factor_Daily':
             {
                'factors': ['WML'],
                 'read_csv_params': {
                     'skiprows': 3,
                     'index_col': 0
                 },
             },
        'F-F_Research_Data_5_Factors_2x3_daily':
            {
                'factors': ['Mkt-RF','SMB','HML','RMW','CMA','RF'],
                'read_csv_params': {
                    'skiprows': 3,
                    'index_col': 0
                },
            },
        '5_Industry_Portfolios_Daily':
            {
                'factors': ['Cnsmr','Manuf','HiTec','Hlth','Other'],
                'read_csv_params': {
                    'skiprows': 9,
                    'index_col': 0,
                    # 'skipfooter': 1,
                    'nrows': 25795,
                    'date_parser': _wrap_dt_parser
                }
            }
    }

    def __init__(self, path_to_input_folder: str,
                 filenames: list):
        """

        :param path_to_input_folder:
        :param filenames: dictionary contains the names of the files as keys and the
        Excell cells (A, B, C) to use as values of the dict
        """

        self.path = path_to_input_folder
        self.filenames = filenames
        # index must be date col
        self._load_files()
        self._transform_files()
        self._set_date_index()

    def _set_date_index(self):

        self.data.index = self.data.index.astype(str)

    def _load_files(self):
        self.factor_names = []
        self._data = {}
        self._size = {}
        self.factor_names = []
        for i, f in enumerate(self.filenames):
            kwargs = FamaFrenchData.USE_COLS_MAPPER[f]['read_csv_params']
            factor_names = FamaFrenchData.USE_COLS_MAPPER[f]['factors']
            df_factors = pd.read_csv(filepath_or_buffer=fr'{self.path}\\{f}.csv',
                                     **kwargs,
                                     parse_dates=True)
            if len(factor_names) > 1:
                for fac in factor_names:
                    self._data[fac] = df_factors.loc[:, fac]
                    self._data[fac].dropna(inplace=True)
                    # self._data[fac] = self._repair_empty_spaced_col(self._data[fac])
                    self._data[fac] = self._data[fac].replace({-99.99:np.nan}) / 100
                    self._size[fac] = len(self._data[fac])
                    self.factor_names.append(fac)
            else:
                self._data[factor_names[0]] = df_factors.replace({-99.99:np.nan}) / 100
                self.factor_names.append(factor_names[0])

    def _convert_index_to_timestamp(self):
        for fac in self.factor_names:
            self._data[fac].index = self._data[fac].apply


    def _repair_empty_spaced_col(self, df):
        def _repair_func(x):
            if isinstance(x, str):
                if x is None:
                    return np.nan
                else:
                    return float(x.replace(' ',''))

        df = df.apply(_repair_func)
        return df

    def _transform_files(self):
        if hasattr(self, '_data'):
            max_size_key = max(self._size, key=self._size.get)
            # find which df is the largest, so that we start with that
            # one, also remove it and put it at first place (to start the merging with it)
            self.factor_names.remove(max_size_key)
            self.factor_names.insert(0, max_size_key)
            for i, fac in enumerate(self.factor_names):
                if i == 0:
                    df = self._data[max_size_key]
                else:
                    if fac != max_size_key:
                        df = pd.merge(df, self._data[fac], how='left', left_index=True, right_index=True)
            self.data = df
        else:
            raise UserWarning('Loading files has failed, self._data has not been set.')



class Eikon:

    def __init__(self, path_api_key: str = r'C:\Users\serge\OneDrive\Documents\apikeys.csv'):

        ek_api_key = pd.read_csv(path_api_key, names=['api', 'key'], index_col=0)
        ek.set_app_key(ek_api_key.loc['reuters'].values[0])

        self.api = ek

    def _parse_date_field(self, df: pd.DataFrame, date_field: str, date_field_alias: str = 'Calc Date'):

        df.loc[:, date_field] = df.loc[:, date_field_alias].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d'))
        df.loc[:, 'Month'] = df.loc[:, date_field].apply(lambda x: x.month)
        df.loc[:, 'Year'] = df.loc[:, date_field].apply(lambda x: x.year)

        return df

    def download_timeseries(self, rics: list, field: list = ['TR.PriceClose', 'Price Close'],
                            date_field: list = ['TR.PriceClose.calcdate', 'Calc Date'], params: dict = None,
                            save_config: dict = {'save': True, 'path': r'C:\Users\serge\IdeaProjects\portfolio_management\models\data\csv' },
                            out: bool = True, set_as_data_data_attribute: bool = True, overwrite: bool = False):

        ## There should be a class attribute name mapper between TR.fields and their corresponding column names
        ## s.t. data_field and field don't have to be lists

        if not isinstance(rics, list):
            raise AttributeError('rics parameter must be list, if it is single RIC, convert to list first')
        data_dict = {}
        for ric in rics:
            print(ric)
            _ric_exists = None
            fname = f'{ric}.csv'
            _files = os.listdir(save_config['path'])
            if fname in _files:
                print('CSV File Found')
                if overwrite:
                    print('Overwriting')
                    pass
                else:
                    continue

            try:
                print('CSV File Not Found, Downloading')
                data_dict[ric] = ek.get_data(ric, [field[0], date_field[0]], parameters=params)[0]
                data_dict[ric] = self._parse_date_field(data_dict[ric], date_field=date_field[0], date_field_alias=date_field[1])
                print(f"Percentage of NaN values: {np.round(100 * data_dict[ric].isna().sum() / len(data_dict[ric]), 2)}")
                if save_config['save']:
                    data_dict[ric].to_csv(f"{save_config['path']}\{ric}.csv")
                    print(f"Saved: {save_config['path']}\{ric}.csv")
            except Exception as e:
                print('Exception occurred')
                print(e)

        data = self._merge_individual_timeseries(data_dict,field[1], date_field[1])
        if set_as_data_data_attribute:
            self.data_dict = data_dict
            self.data = data
        if out:
            return self.data.copy()


    def load_timeseries(self, rics: list, load_path: str, field: list = ['TR.PriceClose', 'Price Close'],
                        date_field: list = ['TR.PriceClose.calcdate', 'Calc Date'], out: bool = True):
        data_dict = {}
        for ric in rics:
            try:
                data_dict[ric] = pd.read_csv(fr'{load_path}\{ric}.csv')
                # print(f'Loaded: {ric}')
            except Exception as e:
                print(e)
        self.data_dict = data_dict
        self.data = self._merge_individual_timeseries(data_dict,field[1], date_field[1])
        if out:
            return self.data.copy()

    def _merge_individual_timeseries(self, data_dict, field_alias: str = 'Price Close',
                                     date_field_alias: str = 'Calc Date'):

        # first one
        first_ric = list(data_dict.keys())[0]
        _df1 = list(data_dict.values())[0]
        df_all = _df1.rename(columns={field_alias: first_ric})
        df_all = df_all[[date_field_alias, first_ric]]

        for i, (ric, df) in enumerate(data_dict.items()):
            try:
                if i > 0:
                    df = df.rename(columns={field_alias: ric})
                    df = df[[ric, date_field_alias]]
                    df_all = pd.merge(df_all, df, on=date_field_alias, how='left')
            except Exception as e:
                print(f'Exception for {ric}: {repr(e)}')

            # print(df_all.count())
        return df_all


    def get_index_constituents(self, index: str = '.SPX', date: str = '20230321'):

        all_rics = []
        t0 = time.time()
        if isinstance(index, str):
            ind = []
            ind.append(index)
            index = ind
        elif not (isinstance(index, list) or isinstance(index, str)):
            raise UserWarning('index must be string or list of str')

        for i in range(len(index)):
            try:
                temp_rics, err = self.api.get_data(index[i], ['TR.IndexConstituentRIC' , 'TR.IndexConstituentName'], {'SDate': date})
                # all_rics, err = ek.get_data(indices_rics[2], ['TR.IndexConstituentRIC' , 'TR.IndexConstituentName'])
                print(f'Retrieved {time.time() - t0}')
                e = None
            except Exception as e:
                print(e)
                continue
            if e is None:
                all_rics.append(temp_rics['Constituent RIC'].to_list())

        return all_rics[0]


    def download_fixed_time_drivers(self, rics: list, save_path: str,
                                    fields: dict = {'GICS Sector Name': 'TR.GICSSector',
                                                    'Country of Headquarters': 'TR.HeadquartersCountry'},
                                    out: bool = True):

        if not isinstance(rics, list):
            raise AttributeError('rics parameter must be list, if it is single RIC, convert to list first')

        try:
            _request = self.api.get_data(rics, list(fields.values()))
            self.data_fixed = _request[0]
            self.data_fixed.set_index('Instrument', inplace=True)
            _msg = _request[1]
            print(_msg)
        except Exception as e:
            print(e)



        self.data_fixed.to_csv(f'{save_path}')

        if out:
            return self.data_fixed

    def load_fixed_time_drivers(self, rics: list, load_path: str, fields: list = ['GICS Sector Name', 'Country of Headquarters'],
                                out: bool = False):

        self.data_fixed = pd.read_csv(f'{load_path}')
        self.data_fixed.set_index('Instrument', inplace=True)
        self.data_fixed = self.data_fixed.loc[rics, fields]
        if out:
            return self.data_fixed





class EikonIndustryRegionClassifier(Eikon):

    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)


    def get_classifications(self, rics: list, ):

        results_sectors = {}
        results_market_cap = {}
        #  'TR.CompanyMarketCap'
        for ric in rics:
            self.api.get_data(ric, fields=['TR.GICSSector'])






# this should have AlphaVantageStock as a subclass, but CCL right now
class AlphaVantage:

    API_KEY = "HE4GW6VF8BJM3MB0"
    # function=TIME_SERIES_INTRADAY&symbol=IBM&interval=' \
    #       f'5min&apikey={api_key}'

    COLUMN_NAMES_MAP = {
        '1. open': 'Open',
        '2. high': 'High',
        '3. low': 'Low',
        '4. close': 'Close',
        '5. adjusted close': 'AdjClose',
        '6. volume': 'Volume',
        '7. dividend amount': 'DividendAmount',
        '8. split coefficient': 'SplitCoef',
    }

    URL = f'https://www.alphavantage.co/query'

    def __init__(self, name: str, symbol: str = None, region='Frankfurt', match_score: float=0.5, save=True):
        """
        

        
        :param keywords: list of strings with which th
        :param match_score:
        """

        
        self.data = {}
        self.name = name
        self.match_score = match_score

        params_tickers = {
            'function': 'SYMBOL_SEARCH',
            'keywords': name,
            'apikey': f'{AlphaVantage.API_KEY}'
        }

        response, code = self._get_request(params_tickers)
        self.ExchangeInfo = pd.DataFrame.from_dict(response['bestMatches'])
        self.ExchangeInfo.set_index('1. symbol', inplace=True)



        self.region = region
        if symbol is None:
            idx = self.ExchangeInfo.index[self.ExchangeInfo['4. region'] == self.region]
            # slicing 0 here will select the index with the best match since the order is pserved after the get request
            try:
                # multiple symbols found
                self.symbol = idx[0]
            except IndexError:
                # only one symbol found
                self.symbol = idx
        else:
            self.symbol = symbol

    @classmethod
    def _get_request(cls, params):
        r = requests.get(cls.URL, params=params)
        if r.status_code != 200:
            raise UserWarning('Something went wrong with the API call')
        return r.json(), r.status_code


    def get_balance_sheet(self, symbol : str = None):
        """
        Get income statement. Symbol availability can be searched in self.ExchangeInfo. Testing only works
        with symbol = 'IBM' (that's demo).
        :param symbol: str, if symbol is None, function will only return,
        :return: pd.DataFrame
        """
        if symbol is None:
            symbol = self.symbol

        params = {
            'function': 'BALANCE_SHEET',
            'symbol': symbol,
            'apikey': AlphaVantage.API_KEY
        }

        print(f'Attempting to retrieve balance sheet for symbol {symbol}')
        response, code = self._get_request(params)
        if len(response) == 0:
            print(f'Could not retrieve balance sheet for symbol {symbol}. Attempting other symbols.')

            for s in self.ExchangeInfo.index:
                params['symbol'] = s
                response, code = self._get_request(params)
                if len(response) > 0:
                    print(f'Retrieved balance sheet for symbol {s}')
                    self.AnnualBalanceSheet = pd.DataFrame.from_dict(response['annualReports'])
                    self.QuarterlyBalanceSheet = pd.DataFrame.from_dict(response['quarterlyReports'])
                    break
                else:
                    print('No balance sheet available')
        else:
            self.AnnualBalanceSheet = pd.DataFrame.from_dict(response['annualReports'])
            self.QuarterlyBalanceSheet = pd.DataFrame.from_dict(response['quarterlyReports'])

    def get_income_statements(self, symbol: str = None):
        """
        Get income statement. Symbol availability can be searched in self.ExchangeInfo. Testing only works
        with symbol = 'IBM' (that's demo).
        :param symbol: str, if symbol is None, function will only return,
        :return: pd.DataFrame
        """
        if symbol is None:
            symbol = self.symbol

        params = {
            'function': 'INCOME_STATEMENT',
            'symbol': symbol,
            'apikey': AlphaVantage.API_KEY
        }

        print(f'Attempting to retrieve income statement for symbol {symbol}')
        response = self._get_request(params)
        if len(response) == 0:
            print(f'Could not retrieve income statement for symbol {symbol}. Attempting other symbols.')

            for s in self.ExchangeInfo.index:
                params['symbol'] = s
                response = self._get_request(params)
                if len(response) > 0:
                    print(f'Retrieved income statement for symbol {s}')
                    self.AnnualIncomeStatement = pd.DataFrame.from_dict(response['annualReports'])
                    self.QuarterlyIncomeStatement = pd.DataFrame.from_dict(response['quarterlyReports'])
                    break
                else:
                    print('No income statement available')
        else:
            self.AnnualIncomeStatement = pd.DataFrame.from_dict(response['annualReports'])
            self.QuarterlyIncomeStatement = pd.DataFrame.from_dict(response['quarterlyReports'])


    def _get_time_series(self, symbol, params, interval='daily'):
        response, code = self._get_request(params)
        if interval in ['daily', 'weekly', 'monthly']:
            col = f'Time Series ({interval.capitalize()})'

        else:
            raise UserWarning('User must specify interval')
        try:
            MetaData = response['Meta Data']
            TimeSeriesDaily = pd.DataFrame.from_dict(response[col]).T
            TimeSeriesDaily.loc[: , 'Symbol'] = symbol
            TimeSeriesDaily.index = pd.to_datetime(TimeSeriesDaily.index)
            TimeSeriesDaily = TimeSeriesDaily.iloc[::-1]
            TimeSeriesDaily.rename(columns=AlphaVantage.COLUMN_NAMES_MAP, inplace=True)
            del response
            return MetaData, TimeSeriesDaily
        except KeyError:
            raise APIError(response)

    def get_time_series(self, symbol: str = None, load: bool = False, inputsize: str ='compact', datatype: str ='json',
                              dividend_adjusted : bool = True, freq : str = 'daily'):

        self.freq = freq
        if symbol is None:
            symbol = self.symbol

        """
        Get daily time series. Symbol availability can be searched in self.ExchangeInfo
        :param symbol: str
        :param inputsize:
        :param datatype:
        :param dividend_adjusted:
        :return: pd.DataFrame
        """

        if dividend_adjusted:
            function = f'TIME_SERIES_{self.freq.swapcase()}_ADJUSTED'
        else:
            function = f'TIME_SERIES_{self.freq.casefold()}'

        params = {'function': f'{function}',
                  'symbol': f'{symbol}',
                  'inputsize': f'{inputsize}',
                  'datatype' : f'{datatype}',
                  'apikey': f'{AlphaVantage.API_KEY}'}

        if not load:
            self.MetaData, self.TimeSeries = self._get_time_series(symbol, params, interval=self.freq)
            # self.MetaData.to_csv(fr'input\{self.name}_MetaData_{self.freq.capitalize()}.csv')
            self.TimeSeries.to_csv(fr'.\input\{self.name}_TimeSeriesData_{self.freq.capitalize()}.csv')




class YahooData:

    def __init__(self, name: str, symbol: str = None):
        super().__init__()

        self.name = name
        if symbol is None:
            self.symbol = MAPPER[self.name['YH']]

        self._load_data()

    @abstractmethod
    def _load_data(self):
        pass


class YahooStockData(YahooData):

    def __init__(self, *args, start_date: pd.Timestamp, end_date: pd.Timestamp):
        self.start_date = start_date
        self.end_date = end_date
        super().__init__(*args)



    def _load_data(self):

        yahoo_object = yf.Ticker(self.symbol)

        # get all stock info (slow)
        yahoo_object.info
        # fast access to subset of stock info (opportunistic)
        yahoo_object.fast_info

        # get historical market data
        self.history = yahoo_object.history(period="1mo")

        # show meta information about the history (requires history() to be called first)
        yahoo_object.history_metadata

        # show actions (dividends, splits, capital gains)
        yahoo_object.actions
        yahoo_object.dividends
        yahoo_object.splits
        yahoo_object.capital_gains  # only for mutual funds & etfs

        # show share count
        # - yearly summary:
        yahoo_object.shares
        # - accurate time-series count:
        start = self.start_date.strftime('%Y-%m-%d')
        end = self.end_date.strftime('%Y-%m-%d')
        df = yahoo_object.get_shares_full(start=start, end=end)

        # show financials:
        # - income statement
        income = yahoo_object.income_stmt
        yahoo_object.quarterly_income_stmt
        # - balance sheet
        yahoo_object.balance_sheet
        yahoo_object.quarterly_balance_sheet
        # - cash flow statement
        yahoo_object.cashflow
        yahoo_object.quarterly_cashflow
        # see `Ticker.get_income_stmt()` for more options

        # show holders
        yahoo_object.major_holders
        yahoo_object.institutional_holders
        yahoo_object.mutualfund_holders

        # show earnings
        yahoo_object.earnings
        yahoo_object.quarterly_earnings

        # show sustainability
        yahoo_object.sustainability

        # show analysts recommendations
        yahoo_object.recommendations
        yahoo_object.recommendations_summary
        # show analysts other work
        yahoo_object.analyst_price_target
        yahoo_object.revenue_forecasts
        yahoo_object.earnings_forecasts
        yahoo_object.earnings_trend

        # show next event (earnings, etc)
        yahoo_object.calendar

        # Show future and historic earnings dates, returns at most next 4 quarters and last 8 quarters by default.
        # Note: If more are needed use yahoo_object.get_earnings_dates(limit=XX) with increased limit argument.
        yahoo_object.earnings_dates

        # show ISIN code - *experimental*
        # ISIN = International Securities Identification Number
        yahoo_object.isin

        # show options expirations
        yahoo_object.options

        # show news
        yahoo_object.news

        # get option chain for specific expiration
        opt = yahoo_object.option_chain('YYYY-MM-DD')
        # data available via: opt.calls, opt.puts


