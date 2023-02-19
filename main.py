from models.data.source import YahooStockData
from models.data.mapper import MAPPER
import pandas as pd
from portfolio_management.models.historical import Historical
from models.data.source import AlphaVantage
from models.data.variables import PriceVariables
from optimization.optimizer import Optimizer
import numpy as np

# names = list(MAPPER.keys())

names = ['Microsoft', 'IBM', 'Amazon']

n_assets = len(names)

config = {
    'start_date' : pd.Timestamp('2022-09-27'),
    'end_date' : pd.Timestamp('2023-02-17'),
    'data': 'download'
}

# data = YahooStockData(**config)


# this is a problem, not sure where to put this, and how to manage it further on
cnt = 0
for name in names:
    print(name)

    if config['data'] == 'download':

        data_source = AlphaVantage(name=name, save=True)

        data_source.get_time_series()

        # These will only work with the IBM symbol (demo works with IBM only)

        # data_source.get_income_statements()
        # data_source.get_balance_sheet()



        data_ts = data_source.TimeSeries
    elif config['data'] == 'load':

        data_ts = pd.read_csv(fr"input\{name}_TimeSeriesData_Daily.csv",
                              index_col=0)
        data_ts.index = data_ts.index.values.astype(np.datetime64)


    variates = PriceVariables(start_date = config['start_date'],
                              end_date = config['end_date'])
    variates.get_price_variables(data = data_ts)



    if cnt == 0:
        joined_data = variates.big_data['returns'].to_frame(name=name).copy()
    else:
        joined_data = joined_data.join(variates.big_data['returns'].to_frame(name=name).copy())

    cnt += 1


model = Historical(joined_data.dropna().values)

opt = Optimizer(method = 'classical M-V')


weights = opt.get_portfolio_weights(mu = model.mu,
                                    sigma = model.cov,
                                    r = 0.01,
                                    method = 'analytic')

print('Optimal weights:')
print({k:v for k,v in zip(names, weights)})



