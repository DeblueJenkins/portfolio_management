from models.data.source import FamaFrenchData
from models.data.source import Eikon
import pickle
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
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
fm = FamaFrenchData(path_to_input_folder=data_path, filenames=filenames)



returns = np.log(data / data.shift(1))

returns.dropna(inplace=True, how='all', axis=0)
returns.dropna(inplace=True, how='all', axis=1)
returns.dropna(inplace=True, how='any', axis=1)

returns = returns[returns.index > '2023-01-01']
returns = returns.iloc[:, :100]

unique_industries = list(set(list(data_industry.values.flatten())))
n_industry = len(unique_industries)
_industry_map = {j:i for i,j in zip(range(n_industry), unique_industries)}

returns = returns.reset_index().rename({'Calc Date': 'time'})


design_matrix = returns.T.join(data_industry, how='left')
design_matrix.dropna(inplace=True, how='any', axis=0)
design_matrix = design_matrix.replace({'GICS Sector Name': _industry_map})
design_matrix = design_matrix.T

# design_matrix = design_matrix.join(fm.data[['HML', 'SMB', 'WML', 'CMA', 'Mkt-RF']], how='left')
# design_matrix.dropna(inplace=True)

design_matrix_reset = design_matrix.T.reset_index().explode('index').rename({'index':'asset', 'GICS Sector Name': 'sector'}, axis=1)

X = pd.melt(design_matrix_reset, id_vars=['asset', 'sector'],
            var_name='time', value_name='value')

md = smf.mixedlm("value ~ time", X,  groups=X["sector"])
md.fit()

