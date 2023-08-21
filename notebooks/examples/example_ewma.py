from models.data.source import Eikon
from models.stat_models.garch import EWMA, get_ewma_variance
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import eikon as ek

path_apikeys = r'C:\Users\serge\OneDrive\Documents\apikeys.csv'
ek_api_key = pd.read_csv(path_apikeys, names=['api', 'key'], index_col=0)

ek.set_app_key(ek_api_key.loc['reuters'].values[0])
data = ek.get_timeseries('AMD.OQ', interval='daily', start_date = "2000-03-01", end_date = "2023-08-20")['HIGH']
data_hourly = ek.get_timeseries('AMD.OQ', interval='hour', start_date = "2021-08-20", end_date = "2023-08-20")['HIGH']


rets = np.log(data.shift(1) / data)
rets_hourly = pd.DataFrame(np.log(data_hourly.shift(1) / data_hourly))
rets_hourly['date'] = [pd.Timestamp(i).date() for i in rets_hourly.index.values]
rets_hourly['sqr_rets'] = rets_hourly.HIGH ** 2
RV = rets_hourly.groupby('date').sum('sqr_rets')
RV = RV['sqr_rets']
RV.plot()


rets = rets[::-1]
rets.dropna(inplace=True)

model = EWMA(rets, tau=125)
model.fit()

model2 = EWMA(rets, tau=len(RV), realized_variance=RV.to_numpy(), realized_dates=RV.index.values)
model2.fit(objective='rv')


func = lambda x: np.sqrt(get_ewma_variance(x, l=model.l, tau=len(x)))
func2 = lambda x: np.sqrt(get_ewma_variance(x, l=model2.l, tau=len(x)))
vols = {}
vols['ewma_traditional'] = rets[::-1].rolling(125).apply(func)
vols['ewma_rv'] = rets[::-1].rolling(125).apply(func2)

vols['std'] = rets[::-1].rolling(125).apply(np.std)
vols['rv'] = np.sqrt(RV)
vols = pd.DataFrame.from_dict(vols)
vols.dropna(how='any').plot()
plt.show()

# i think the ewma_rv needs more parameters
# would be interesting to a GARHC model fit on RV difference
# or just RV AR(p) model