from abc import abstractmethod
import pandas as pd
import yfinance as yf
import warnings
import requests
from .mapper import MAPPER

warnings.filterwarnings("ignore")

class APIError(Exception):
    """An API Error Exception"""

    def __init__(self, response_message):
        self.msg = response_message

    def __str__(self):
        return "APIError: status={}".format(self.msg)


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


