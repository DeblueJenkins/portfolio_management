from models.data.source import FamaFrenchData, Eikon, FRED
import pickle
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.regression.linear_model import OLS
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import warnings



def get_corr(X, col='y'):

    sectors = X['sector'].unique()
    m = len(sectors)
    _res = np.zeros((m, m))

    X['sector'] = X['sector'].astype(int)
    X_grp = X[['sector', 'time', col]].groupby(['sector', 'time']).agg(lambda x: x.mean())

    for i in range(m):
        for j in range(i):
            y1 = X_grp.loc[(sectors[i], slice(None)),:]
            y1 = y1[y1[col] != 0.0]
            y2 = X_grp.loc[(sectors[j], slice(None)),:]
            y2 = y2[y2[col] != 0.0]
            if len(y1) > len(y2):
                y = pd.merge(y2,y1, how='left', suffixes=('_2', '_1'), on='time')
            else:
                y = pd.merge(y1,y2, how='left', suffixes=('_1', '_2'), on='time')

            _corr = y[[f'{col}_1', f'{col}_2']].corr().iloc[0,1]
            _res[i,j] = _corr
            _res[j,i] = _corr
            np.fill_diagonal(_res, 1)

    res = pd.DataFrame(index=sectors, columns=sectors, data=_res)
    return res

def preprocess_design_matrix(returns, data_industry, gdp, vix):

    unique_industries = list(set(list(data_industry.values.flatten())))
    n_industry = len(unique_industries)
    industry_map = {j:i for i,j in zip(range(n_industry), unique_industries)}

    # join with industry indicators
    X = returns.T.join(data_industry, how='left')
    X = X.replace({'GICS Sector Name': industry_map})
    X = X.T

    # convert to long format
    X = (X.T.reset_index() \
         # .iloc[1:,:]\
         # .explode('index')\
         .rename({'index':'asset', 'GICS Sector Name': 'sector'}, axis=1))

    X = pd.melt(X, id_vars=['asset', 'sector'],
                var_name='time', value_name='value')

    # join with fama-french
    X = pd.merge(X, ff, on='time', how='left')

    # drop nans
    X.dropna(inplace=True)
    X = X[X.sector != 0]

    X['value'] = X['value'].astype(float)
    X['y'] = X['value'] - X['RF']
    X['alpha'] = np.ones(len(X))

    # get dummies
    D_industry = pd.get_dummies(X['sector']).astype(int)
    D_industry = D_industry.iloc[:, :-1]
    D_industry.columns = list(industry_map.keys())[1:-1]


    # get time columns
    X['time'] = X['time'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d'))
    X['year'] = X['time'].apply(lambda x: x.year)
    X['year_month'] = X['time'].apply(lambda x: f'{x.year}{x.month:02}')

    X = pd.merge(X, gdp[['year_month', 'regime', 'growth']], left_on='year_month', right_on='year_month', how='left')
    # we can forward fill since the gdp is reported on the first of the month
    # although interpolation would be better most likely
    X['regime'] = X['regime'].fillna(method='ffill')
    X['growth'] = X['growth'].fillna(method='ffill')

    D_crisis = pd.get_dummies(X['regime']).astype(int).iloc[:,[0,2,3]]

    X = X.join(D_industry, how='left').dropna()
    X = X.join(D_crisis, how='left').dropna()

    X = pd.merge(X, vix[['VIX_h', 'DATE']], left_on='time', right_on='DATE', how='left')
    warnings.warn('Losing Jan and Feb 2001 with this join since of a mismatch of dates. Can be fixed tho. ')

    n_lag_1 = X.groupby('time').count().iloc[0,0]
    X['VIX_h_l1'] = X['VIX_h'].shift(n_lag_1)

    X.dropna(inplace=True)

    return X, industry_map, list(D_industry.columns), list(D_crisis.columns)


def get_vix_data(path, n_horizon):

    vix = pd.read_csv(path)
    vix['DATE'] = vix['DATE'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d'))
    vix['year_month'] = vix['DATE'].apply(lambda x: f'{x.year}{x.month:02}')
    vix.replace({'VIXCLS': {'.': np.nan}}, inplace=True)
    vix['VIXCLS'] = vix['VIXCLS'].astype(float) / 100
    vix['VIX_var_h'] = (vix['VIXCLS'] ** 2).rolling(n_horizon).sum()
    vix['VIX_h'] = np.sqrt(vix['VIX_var_h'])


    vix.dropna(inplace=True)
    return vix



def get_gdp_data(path):

    gdp = pd.read_csv(path)
    # gdp.set_index('DATE', inplace=True)
    gdp['DATE'] = gdp['DATE'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d'))
    gdp['year_month'] = gdp['DATE'].apply(lambda x: f'{x.year}{x.month:02}')
    gdp['growth'] = (gdp['GDP'] - gdp['GDP'].shift(1)) / gdp['GDP'].shift(1)
    gdp.dropna(inplace=True)

    q_neg = np.quantile(gdp.growth[gdp.growth < 0], q=0.1)
    q_pos = np.quantile(gdp.growth[gdp.growth > 0], q=0.9)

    gdp['regime'] = ''
    gdp.loc[gdp['growth'] < q_neg, 'regime'] = 'extreme_crisis'
    gdp.loc[(gdp['growth'] > q_neg) & (gdp['growth'] < 0), 'regime'] = 'normal_crisis'
    gdp.loc[(gdp['growth'] > 0) & (gdp['growth'] < q_pos), 'regime'] = 'normal_boom'
    gdp.loc[gdp['growth'] > q_pos, 'regime'] = 'extreme_boom'


    return gdp



if __name__ == '__main__':

    path_data = r'C:\Users\serge\OneDrive\portfolio_management\data\csv'
    path_api_eys = r'C:\Users\serge\OneDrive\reuters\apikeys.csv'
    cons = pickle.load(open(r'C:\Users\serge\OneDrive\portfolio_management\rics_of_SPX.pkl', 'rb'))
    filenames = ['Developed_5_Factors_Daily', 'Developed_MOM_Factor_Daily',
             '5_Industry_Portfolios_Daily']
    # returns horizon, 21 would be monthly, etc.
    n_horizon = 21
    start_date = '2000-12-31'
    end_date = '2024-08-29'


    api = Eikon(path_api_eys)
    # cons = api.get_index_constituents('.SP500', date='20240915')
    fm = FamaFrenchData(path_to_input_folder=fr'{path_data}\fama-french-factors', filenames=filenames)
    gdp = get_gdp_data(path=fr'{path_data}\macro\fred\GDP.csv')
    vix = get_vix_data(path=fr'{path_data}\macro\fred\VIXCLS.csv', n_horizon=n_horizon)

    macro_indicators = [
        'GDP',
        'COMREPUSQ159N',
        'CPALTT01USM661S',
        'MABMM301USM189S',
        'MABMM301USM657S',
        'RBUSBIS',
        'LRUN64TTUSM156S',
        'IRLTLT01USM156N',
        'USALOLITONOSTSAM',
    ]

    fred = FRED(path=fr'{path_data}\macro\fred',
                tickers=macro_indicators,
                start_date=start_date,
                end_date=end_date)

    params = {
        'rics': cons,
        # 'field': ['TR.PriceClose', 'Price Close'],
        'field': ['TR.PriceClose', 'Price Close'],
        'date_field': ['TR.PriceClose.calcdate', 'Calc Date'],
        'load_path': fr'{path_data}\prices',
        # 'save_config': {'save': True, 'path': fr'{path_data}\macro' },
        # 'params': {
        #     'SDate': start_date,
        #     'EDate': end_date,
        # }
    }

    data = api.load_timeseries(**params, out=True)
    data.set_index('Calc Date', inplace=True)

    params = {
        'rics': cons,
        # 'fields': {'GICS Sector Name': 'TR.GICSSector'},
        'fields': ['GICS Sector Name'],
        'load_path': fr'{path_data}\fixed_factors\industry.csv',
    }

    data_industry = api.load_fixed_time_drivers(**params, out=True)


    # this converts the returns to the right horizon
    ff = (1 + fm.data).rolling(n_horizon).apply(lambda x: x.prod()) - 1
    ff = ff.reset_index().rename({'index':'time', 'Mkt-RF':'MktRF'}, axis=1)

    returns = np.log(data / data.shift(n_horizon))[::n_horizon].iloc[1:,:]
    returns.dropna(inplace=True, how='any', axis=1)
    # returns.dropna(inplace=True, how='all', axis=1)
    returns.dropna(inplace=True, how='any', axis=0)

    # returns = returns[returns.index > '2023-01-01']
    # returns = returns.iloc[:, :100]

    X, industry_map, col_industry, col_regime = preprocess_design_matrix(returns, data_industry, gdp, vix)

    # correlation between industries (groups)
    corr_matrix = get_corr(X)
    corr_matrix.index = [{v:k for k,v in industry_map.items()}[i] for i in corr_matrix.index]
    corr_matrix.columns = [{v:k for k,v in industry_map.items()}[i] for i in corr_matrix.columns]

    model_ols = OLS(endog=X['y'],
                    exog=X[['alpha', 'MktRF', 'SMB', 'HML', 'WML', 'VIX_h']],
                    hasconst=True)
    res_ols = model_ols.fit(cov_type='cluster', cov_kwds={'groups': X['regime']})

    X['y_hat_ols'] = res_ols.fittedvalues
    X['e_ols'] = res_ols.resid

    # check error clusters (group-based)
    sns.scatterplot(X, x='y_hat_ols', y='e_ols', hue='sector')
    plt.show()

    # check error cluster (regime-based, i.e., time)
    sns.scatterplot(X, x='y_hat_ols', y='e_ols', hue='regime')
    plt.show()

    # check correlation between VIX and y given different GDP growth (regimes)
    sns.scatterplot(X, x='y', y='VIX_h', hue='regime')
    plt.show()

    # specify and fit random effects model
    model_fe = OLS(endog=X['y'],
                   exog=X[['alpha', 'MktRF', 'SMB', 'HML', 'WML', 'VIX_h'] + col_regime],
                   hasconst=True)

    # changes absolutely nothing
    # res_fe = model_fe.fit()
    res_fe = model_fe.fit(cov_type='cluster', cov_kwds={'groups': X['regime']})

    X['y_hat_fe'] = res_fe.fittedvalues
    X['e_fe'] = res_fe.resid

    # specify and fit random effects model with heteroskedasticity-robust errors

    # res_fe_robust = model_fe.fit(cov_type='HC3')

    # check error clusters (group-based)
    sns.scatterplot(X, x='y_hat_fe', y='e_fe', hue='sector')
    plt.show()

    # check error cluster (regime-based, i.e., time)
    sns.scatterplot(X, x='y_hat_fe', y='e_fe', hue='regime')
    plt.show()

    # check correlation between VIX and y given different GDP growth (regimes)
    sns.scatterplot(X, x='y', y='VIX_h', hue='regime')
    plt.show()

    sns.scatterplot(X, x='y', y='e_fe')
    plt.show()


    # specify and fit random effects model
    model_re = MixedLM(endog=X['y'],
                       exog=X[['alpha', 'MktRF', 'SMB', 'HML', 'WML', 'VIX_h']],
                       groups=X['sector'])
    res_re = model_re.fit()
    # md = smf.mixedlm("y ~ MktRF + SMB + HML + WML", X,  groups=X["sector"])
    # res = md.fit()




    # plots

    # y vs e
    fig, axs = plt.subplots(3,1,sharex=True, sharey=True)
    axs[0].scatter(X['y'], res_re.resid)
    axs[1].scatter(X['y'], res_fe.resid)
    axs[2].scatter(X['y'], res_fe_robust.resid)
    plt.show()

    fig, axs = plt.subplots(3,1,sharex=True, sharey=True)
    axs[0].scatter(res_re.fittedvalues, res_re.resid)
    axs[1].scatter(res_fe.fittedvalues, res_fe.resid)
    axs[2].scatter(res_fe_robust.fittedvalues, res_fe_robust.resid)

    plt.show()

    fig, axs = plt.subplots(3,1,sharex=True, sharey=True)
    sns.distplot(res_re.resid, ax=axs[0])
    sns.distplot(res_fe.resid, ax=axs[1])
    sns.distplot(res_fe_robust.resid, ax=axs[2])
    plt.show()

    # assessing the errors

    # the clusters that we see in the y_hat vs e plots are due to crises

    X['y_hat_fe'] = res_fe.fittedvalues

    X[X['y_hat_fe'] < -0.3].groupby('year').count()
    X[(X['y_hat_fe'] > -0.3) & (X['y_hat_fe'] < -0.2)].groupby('year').count()
    X[(X['y_hat_fe'] > -0.2) & (X['y_hat_fe'] < -0.1)].groupby('year').count()