import pandas as pd
import numpy as np
import datetime as dt
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
prev_path = "\\".join(list(os.getcwd().split('\\')[0:-1]))
if module_path not in sys.path: sys.path.append(prev_path)
import matplotlib.pyplot as plt
from optimization.models.portfolio import MarkowitzPortfolio, generate_portfolio_data, generate_portfolio_garch_data, KellyPortfolio, SharpePortfolio
import numpy as np

m = np.array([0.00042243, 0.00051672, 0.00026824, 0.00048209, 0.00064111,
              0.0009473 , 0.00011558, 0.00079135, 0.00020627, 0.00028736])


s = np.array([[4.01633077e-04, 2.88469913e-04, 9.20412470e-05, 1.39007357e-04,
               8.70685391e-05, 1.19824140e-04, 1.10731123e-04, 9.40674827e-05,
               2.11125041e-04, 1.22474972e-04],
              [2.88469913e-04, 7.99345692e-04, 7.73573044e-05, 1.33743811e-04,
               8.81861326e-05, 1.73681572e-04, 1.28403944e-04, 7.73591642e-05,
               3.31483246e-04, 1.66472091e-04],
              [9.20412470e-05, 7.73573044e-05, 1.43316371e-04, 5.86198133e-05,
               6.50543207e-05, 4.48140341e-05, 8.26460095e-05, 5.26800256e-05,
               7.52407309e-05, 4.57669236e-05],
              [1.39007357e-04, 1.33743811e-04, 5.86198133e-05, 3.12030862e-04,
               4.68730549e-05, 1.49923588e-04, 6.35742676e-05, 1.34408563e-04,
               1.16620707e-04, 1.44659075e-04],
              [8.70685391e-05, 8.81861326e-05, 6.50543207e-05, 4.68730549e-05,
               3.45608103e-04, 3.48388692e-05, 8.12828624e-05, 3.50902419e-05,
               8.14822307e-05, 4.31172382e-05],
              [1.19824140e-04, 1.73681572e-04, 4.48140341e-05, 1.49923588e-04,
               3.48388692e-05, 7.22375415e-04, 4.86216287e-05, 1.32628769e-04,
               1.29592656e-04, 2.34347085e-04],
              [1.10731123e-04, 1.28403944e-04, 8.26460095e-05, 6.35742676e-05,
               8.12828624e-05, 4.86216287e-05, 2.01400889e-04, 6.03298437e-05,
               9.75211915e-05, 5.72384601e-05],
              [9.40674827e-05, 7.73591642e-05, 5.26800256e-05, 1.34408563e-04,
               3.50902419e-05, 1.32628769e-04, 6.03298437e-05, 4.17461940e-04,
               7.84363171e-05, 1.27768168e-04],
              [2.11125041e-04, 3.31483246e-04, 7.52407309e-05, 1.16620707e-04,
               8.14822307e-05, 1.29592656e-04, 9.75211915e-05, 7.84363171e-05,
               5.17377921e-04, 1.36615567e-04],
              [1.22474972e-04, 1.66472091e-04, 4.57669236e-05, 1.44659075e-04,
               4.31172382e-05, 2.34347085e-04, 5.72384601e-05, 1.27768168e-04,
               1.36615567e-04, 3.85728990e-04]])

prices = np.array([651.2, 558.1, 112.9, 299.86, 491.55, 125.89, 319.9, 405.64,
                   114.1727328, 250.79])



r = 0.043

training_sample, testing_sample = generate_portfolio_data(mu=m, cov=s, training_size=3000, testing_size=500)
sim_returns = generate_portfolio_garch_data(mu=m, cov=s, n_time=1000)
sim_prices = np.zeros((sim_returns.shape[0], sim_returns.shape[1] + 1))
sim_prices[:,0] = prices
for t in np.arange(1, 1000):
    sim_prices[:, t] = sim_prices[:, t-1] * np.exp(sim_returns[:, t-1])

fig, axs = plt.subplots(sim_returns.shape[0], 1, figsize=(11, 8))
for i in range(sim_returns.shape[0]):
    axs[i].plot(sim_returns[i, :])
plt.show()

fig, axs = plt.subplots(sim_prices.shape[0], 1, figsize=(11,8))
for i in range(sim_prices.shape[0]):
    axs[i].plot(sim_prices[i,:])
plt.show()

results = {}

model_kelly = KellyPortfolio(training_sample, r)
model_sharpe = SharpePortfolio(training_sample, r)

results['weights_sharpe_allow_short_allow_lev'] = model_sharpe.estimate_model('SHARPE', allow_short_selling=True, allow_leverage=True, out=True)
results['weights_sharpe_allow_short_disallow_lev'] = model_sharpe.estimate_model('SHARPE', allow_short_selling=True, allow_leverage=False, out=True)
results['weights_sharpe_disallow_short_disallow_lev'] = model_sharpe.estimate_model('SHARPE', allow_short_selling=False, allow_leverage=False, out=True)
results['weights_sharpe_disallow_short_allow_lev'] = model_sharpe.estimate_model('SHARPE', allow_short_selling=False, allow_leverage=True, out=True)


results['weights_kelly_allow_short_allow_lev'] = model_kelly.estimate_model('UNCONSTRAINED_MEAN_OR_VOL', True, True, out=True)
results['weights_kelly_allow_short_disallow_lev'] = model_kelly.estimate_model('UNCONSTRAINED_MEAN_OR_VOL', True, False, out=True)
results['weights_kelly_disallow_short_disallow_lev'] = model_kelly.estimate_model('UNCONSTRAINED_MEAN_OR_VOL', False, False, out=True)
results['weights_kelly_disallow_short_allow_lev'] = model_kelly.estimate_model('UNCONSTRAINED_MEAN_OR_VOL', True, False, out=True)

print(results)