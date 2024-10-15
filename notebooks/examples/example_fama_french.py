from models.data.source import FamaFrenchData
from models.data.source import Eikon
import pickle
import numpy as np
import matplotlib.pyplot as plt
import statsmodels
import sklearn
from statsmodels.regression.linear_model import OLS
import pandas as pd


data_path = r'C:\Users\serge\IdeaProjects\portfolio_management\models\data\fama-french-factors'
filenames = ['Developed_5_Factors_Daily', 'Developed_MOM_Factor_Daily',
             '5_Industry_Portfolios_Daily']

fm = FamaFrenchData(path_to_input_folder=data_path, filenames=filenames)
# print(fm.data)





PATH_API_KEYS = r'C:\Users\serge\OneDrive\reuters\apikeys.csv'

api = Eikon(PATH_API_KEYS)

# cons = api.get_index_constituents('.SP500', date='20240915')
cons = pickle.load(open(r'C:\Users\serge\IdeaProjects\portfolio_management\models\data\rics_of_SPX.pkl', 'rb'))


params = {
    'rics': cons,
    # 'field': ['TR.PriceClose', 'Price Close'],
    'field': ['TR.PriceClose', 'Price Close'],
    'date_field': ['TR.PriceClose.calcdate', 'Calc Date'],
    'load_path': r'C:\Users\serge\IdeaProjects\portfolio_management\models\data\csv\prices',
    # 'save_config': {'save': True, 'path': r'C:\Users\serge\IdeaProjects\portfolio_management\models\data\csv\prices' },
    # 'params': {
    #     'SDate':'2000-12-31',
        # 'EDate': '2024-08-29',
    # }
}

data = api.load_timeseries(**params, out=True)
data.set_index('Calc Date', inplace=True)



# industry and region

params = {
    'rics': cons,
    # 'fields': {'GICS Sector Name': 'TR.GICSSector'},
    'fields': ['GICS Sector Name'],
    'load_path': r'C:\Users\serge\IdeaProjects\portfolio_management\models\data\csv\fixed_factors\industry.csv',
}

data_industry = api.load_fixed_time_drivers(**params, out=True)



returns = np.log(data / data.shift(1))

returns.dropna(inplace=True, how='all', axis=0)
returns.dropna(inplace=True, how='all', axis=1)
returns.dropna(inplace=True, how='any', axis=1)

n = 0

while n < 20:

    idx = np.random.choice(returns.columns)
    r = returns.loc[:,idx]
    print(idx)
    data = fm.data.join(r, how='left')
    data.dropna(inplace=True)
    data['intercept'] = np.ones(len(data))


    features = ['HML', 'SMB', 'WML', 'CMA',
                'Cnsmr','Manuf','HiTec','Hlth','Other']
    _results = {}
    _params = {}


    exog_col = 'intercept'
    exog = [exog_col]
    res_old = None
    for f in features:
        exog += [f]
        exog_col += f'-{f}'
        y = data[idx] - data['RF']
        model = OLS(y, data[exog], hasconst=True)
        res = model.fit(cov_type='HC1')
        _res = {}
        _res['aic'] = np.round(res.aic)
        _res['r2'] = np.round(res.rsquared_adj,3)
        _res['corr(e,y)'] = np.round(np.corrcoef(res.resid, y)[0,1],4)
        # if res_old is not None:
        #     _res['lm_test_against_prev'] = res.compare_lm_test(res_old)
        # else:
        #     _res['lm_test_against_prev'] = np.nan
        res_old = res
        _results[exog_col] = _res
        _params[exog_col] = {np.round(k,3):np.round(v,3) for k,v in zip(res.params,res.pvalues)}

    table_metrics = pd.DataFrame.from_dict(_results).T
    table_params = pd.DataFrame.from_dict(_params).T

    print(res.summary())
    print(table_metrics)

    if res.pvalues[0] < 0.1:
        print(f'Non-null alpha for {idx}')

    n += 1




# Let's find out which factor was outperforming the rest of the factors in the last 6-months,
# (possibly by amount f (hyperparameter)), then construct a portfolio which longs companies that have above
# beta_u (hyperparameter) exposure to the factor and short companies that have below beta_l (hyperparameter)
# exposure to the factor; we are going to hold this portfolio until n_horizon (hyperparameter) and check the performance

# let's test the strategy first on the period after the FED rate hikes (Jan 2020 - Sep 2024)
# We will roll 9-months of training and test on following 3-months; this will give us 9*21=189 data points
# to estimate 5 parameters, which should be alright.


# let's also see how stable are the factor loadings, i.e., the exposures through time



returns_strategy = returns.loc['2020-01-01':, :]
factors_strategy = fm.data.loc['2020-01-01':, :]

n, m = returns_strategy.shape

train_size = 90
test_size = 30

strategy_returns = {}
for t in np.arange(0, n-train_size-test_size, test_size):
    print(f'Backtesting')
    start_train_date = returns_strategy.index[t]
    end_train_date = returns_strategy.index[t+train_size]
    end_test_date = returns_strategy.index[t+train_size+test_size]
    print(f'Estimation: {start_train_date} to {end_train_date}')
    print(f'Testing from {end_train_date} to {end_test_date}')
    _returns = returns_strategy.iloc[t:t+train_size, :]
    _factors = factors_strategy.iloc[t:t+train_size,:][['Mkt-RF', 'SMB', 'HML', 'CMA', 'WML', 'RF']]
    exposures = {}
    for i in range(m):
        _r = _returns.iloc[:,i]
        _name = _returns.columns[i]
        _data = _factors.join(_r, how='left')
        _data['intercept'] = np.ones(train_size)
        _y = _data[_name] - _data['RF']

        _model = OLS(_y, _data[['Mkt-RF', 'SMB', 'HML', 'CMA', 'WML', 'intercept']], hasconst=True)
        _res = _model.fit(cov_type='HC1')
        exposures[_name] = _res.params

    exposures = pd.DataFrame.from_dict(exposures)
    # this is the strategy now
    _cols_factors = ['SMB', 'HML', 'CMA', 'WML']
    cum_returns_factors = 1 - (1 + _data[_cols_factors]).product()
    best_performing_factor = _cols_factors[np.argmax(cum_returns_factors)]
    # find 10 most exposed and 10 least exposed
    exposures_best_performing = exposures.loc[best_performing_factor, :].sort_values()
    bottom_10 = exposures_best_performing.iloc[:10].index
    top_10 = exposures_best_performing.iloc[-10:].index

    bottom_10_returns = 1 - (1 + returns_strategy.iloc[t+train_size:t+train_size+test_size, :][bottom_10]).product()
    top_10_returns = 1 - (1+returns_strategy.iloc[t+train_size:t+train_size+test_size, :][top_10]).product()


    my_portfolio = - (top_10_returns / 10).sum() + (bottom_10_returns / 10).sum()
    strategy_returns[returns_strategy.index[t+train_size+test_size]] = my_portfolio


pd.Series(strategy_returns).plot()
plt.show()

