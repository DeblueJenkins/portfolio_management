from models.stat_models.garch import EWMA
from models.data.source import Eikon
import numpy as np
import pickle
from models.data.handler import DataHandler
import matplotlib.pyplot as plt
import datetime as dt
import os
import pandas as pd


data_path = r'C:\Users\HP\Documents\GitHub\portfolio_management\models\data'
path_apikeys = r'C:\Users\HP\IdeaProjects\FinIgor\data\apikeys.csv'
eikon_api = Eikon(path_apikeys)

index = '.SPX'

get_constituents = False
download = False

univariate = True

if get_constituents:
    rics_list = eikon_api.get_index_constituents()
    with open(fr'{data_path}\rics_of_{index.replace(".","")}.pkl', 'wb') as f:
        pickle.dump(rics_list, f)
        print('Pickled.')

else:
    rics_list = pickle.load(open(fr'{data_path}\rics_of_{index.replace(".","")}.pkl', 'rb'))
    print('Unpickled.')

rics_list = rics_list[:100]





params_download = {
    'rics': rics_list,
    'field': ['TR.PriceClose', 'Price Close'],
    'date_field': ['TR.PriceClose.calcdate', 'Calc Date'],
    'save_config': {
        'path': os.path.join(data_path, 'csv'),
        'save': True
    },
    'params': {
        'SDate':'1999-12-31',
        'EDate': '2023-07-01',
        'Curn':'Native',
    },
}

params_load = {
    'rics': rics_list,
    'field': ['TR.PriceClose', 'Price Close'],
    'date_field': ['TR.PriceClose.calcdate', 'Calc Date'],
    'load_path': os.path.join(data_path, 'csv')

}

config = {
    'PCA': True,
    'n_components': 7,
    'OLS': True
}
if download:
    data = eikon_api.download_timeseries(**params_download)
else:
    data = eikon_api.load_timeseries(**params_load)

# sample weight can be exp. weighted

preprocessor = DataHandler(data=data, date_col=params_load['date_field'][1])

X = preprocessor.get_returns(period=1)

univariate = True

if univariate:

    rets = X['MCD.N'].iloc[::-1]
    model = EWMA(rets, tau=60)
    res = model.fit(out=True)

    ewm_mean, ewm_vars = model.get_conditional_moments()
    simple_rolling_means, simple_rolling_vars = model.get_conditional_moments(l=1.0)

    end_date = '2023-05-27'
    start_date = '2022-04-27'

    intraday_prices = eikon_api.api.get_timeseries(rics='MCD.N', fields=['CLOSE'], start_date=start_date, end_date=end_date, interval='minute')
    intraday_returns = np.log(intraday_prices / intraday_prices.shift(1))
    del intraday_prices
    intraday_returns.loc[:, 'day'] = [i.date() for i in intraday_returns.index]
    intraday_returns.loc[:, 'sqr_rets'] = intraday_returns['CLOSE'] ** 2
    realized_variance = intraday_returns.groupby('day').sum()
    realized_variance.loc[:, 'new_index'] = [str(i + dt.timedelta(days=1)) for i in realized_variance.index.values]
    # this will make it comparable
    realized_variance.set_index('new_index', inplace=True)
    realized_variance.rename({'sqr_rets': 'realized_variance'}, axis=1, inplace=True)
    realized_variance['realized_vol'] = realized_variance['realized_variance'] ** 0.5

    df = pd.merge(np.sqrt(realized_variance['realized_variance']), np.sqrt(ewm_vars),
                  left_index=True, right_index=True)
    df = pd.merge(df, np.sqrt(simple_rolling_vars),
                  left_index=True, right_index=True)
    df.plot(figsize=(11,8))
    plt.show()




else:
    rets = X.iloc[::-1, :]





