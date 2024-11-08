from models.data.source import FamaFrenchData
from models.data.source import Eikon
import pickle
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.regression.linear_model import OLS
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



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

def preprocess_design_matrix(returns, data_industry, ):

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
    D = pd.get_dummies(X['sector']).astype(int)
    D = D.iloc[:, :-1]
    D.columns = list(industry_map.keys())[1:-1]

    return X, D, industry_map


if __name__ == '__main__':

    path_data = r'C:\Users\serge\OneDrive\portfolio_management\data\csv'
    path_api_eys = r'C:\Users\serge\OneDrive\reuters\apikeys.csv'
    cons = pickle.load(open(r'C:\Users\serge\OneDrive\portfolio_management\rics_of_SPX.pkl', 'rb'))
    filenames = ['Developed_5_Factors_Daily', 'Developed_MOM_Factor_Daily',
             '5_Industry_Portfolios_Daily']
    # returns horizon, 21 would be monthly, etc.
    n_horizon = 21

    api = Eikon(path_api_eys)
    fm = FamaFrenchData(path_to_input_folder=fr'{path_data}\fama-french-factors', filenames=filenames)

    # cons = api.get_index_constituents('.SP500', date='20240915')



    params = {
        'rics': cons,
        # 'field': ['TR.PriceClose', 'Price Close'],
        'field': ['TR.PriceClose', 'Price Close'],
        'date_field': ['TR.PriceClose.calcdate', 'Calc Date'],
        'load_path': fr'{path_data}\prices',
        # 'save_config': {'save': True, 'path': r'C:\Users\serge\IdeaProjects\portfolio_management\models\data\csv\prices' },
        # 'params': {
        #     'SDate':'2000-12-31',
            # 'EDate': '2024-08-29',
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

    X, D, industry_map = preprocess_design_matrix(returns, data_industry)

    # correlation between industries (groups)
    corr_matrix = get_corr(X)
    corr_matrix.index = [{v:k for k,v in industry_map.items()}[i] for i in corr_matrix.index]
    corr_matrix.columns = [{v:k for k,v in industry_map.items()}[i] for i in corr_matrix.columns]

    # specify and fit random effects model
    model_re = MixedLM(endog=X['y'],
                       exog=X[['alpha', 'MktRF', 'SMB', 'HML', 'WML']],
                       groups=X['sector'])
    res_re = model_re.fit()
    # md = smf.mixedlm("y ~ MktRF + SMB + HML + WML", X,  groups=X["sector"])
    # res = md.fit()

    # specify and fit random effects model
    model_fe = OLS(endog=X['y'],
                   exog=pd.concat([X[['alpha', 'MktRF', 'SMB', 'HML', 'WML']], D], axis=1),
                   hasconst=True)

    # changes absolutely nothing
    res_fe = model_fe.fit()

    # specify and fit random effects model with heteroskedasticity-robust errors
    res_fe_robust = model_fe.fit(cov_type='cluster', cov_kwds={'groups': X['sector']})


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
    sns.displot(res_re.resid, ax=axs[0])
    sns.displot(res_fe.resid, ax=axs[1])
    sns.displot(res_fe_robust.resid, ax=axs[2])
    plt.show()

    # assessing the model