from models.data.source import Eikon
import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import arch.data.sp500
import datetime as dt
import matplotlib.pyplot as plt


PATH_API_KEYS = r'C:\Users\serge\OneDrive\reuters\apikeys.csv'
api = Eikon(PATH_API_KEYS)

START_DATE = '1990-12-31'
END_DATE = '2024-04-27'

data_raw = api.api.get_data('.SP500', fields=['TR.PriceClose.calcdate', 'TR.PriceClose'], parameters={'SDate': START_DATE,
                                                                                                  'EDate': END_DATE,
                                                                                                  'Curn': 'Native'})

data = data_raw[0]
data['returns'] = 100 * np.log(data['Price Close'] / data['Price Close'].shift(1))
data['Calc Date'] = pd.DatetimeIndex(data['Calc Date'])
data.set_index('Calc Date', inplace=True)
data = data.drop(['Instrument', 'Price Close'], axis=1).dropna()

# returns = returns.to_numpy()[1:]

model = arch_model(y=data, mean='zero', vol='GARCH', p=2, q=1, power=2, dist='t')
res = model.fit(update_freq=5)
res.summary()

sim_forecasts = res.forecast(start="2024-04-28", method="simulation", horizon=30)
print(sim_forecasts.residual_variance.dropna().head())