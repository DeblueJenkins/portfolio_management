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


corr = np.array([[ 1.        ,  0.09085473, -0.69770486, -0.66825076, -0.61282197,
                -0.44232506,  0.19836368, -0.41650365, -0.25751082,  0.28424596],
              [ 0.09085473,  1.        ,  0.06752966, -0.1311518 ,  0.58351884,
                0.50202211, -0.67070902, -0.70147012, -0.51269642, -0.78612631],
              [-0.69770486,  0.06752966,  1.        ,  0.07888222,  0.32957044,
               0.12379964, -0.02069225,  0.06900167, -0.04297018, -0.25689535],
              [-0.66825076, -0.1311518 ,  0.07888222,  1.        ,  0.44789692,
               0.32031553, -0.31267621,  0.26208864,  0.38876085, -0.21610211],
              [-0.61282197,  0.58351884,  0.32957044,  0.44789692,  1.        ,
               0.79119842, -0.55206198,  0.03859865, -0.04110355, -0.8058536 ],
              [-0.44232506,  0.50202211,  0.12379964,  0.32031553,  0.79119842,
               1.        , -0.31915587, -0.14184056, -0.33462193, -0.71248885],
              [ 0.19836368, -0.67070902, -0.02069225, -0.31267621, -0.55206198,
                -0.31915587,  1.        ,  0.25385289, -0.12091004,  0.49900973],
              [-0.41650365, -0.70147012,  0.06900167,  0.26208864,  0.03859865,
               -0.14184056,  0.25385289,  1.        ,  0.70926938,  0.4277049 ],
              [-0.25751082, -0.51269642, -0.04297018,  0.38876085, -0.04110355,
               -0.33462193, -0.12091004,  0.70926938,  1.        ,  0.28445784],
              [ 0.28424596, -0.78612631, -0.25689535, -0.21610211, -0.8058536 ,
                -0.71248885,  0.49900973,  0.4277049 ,  0.28445784,  1.        ]])


prices = np.array([651.2, 558.1, 112.9, 299.86, 491.55, 125.89, 319.9, 405.64,
                   114.1727328, 250.79])



r = 0.043

training_sample, testing_sample = generate_portfolio_data(mu=m, corr=corr, training_size=3000, testing_size=500)
sim_returns = generate_portfolio_garch_data(mu=m, corr=corr, n_time=1000)
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