from models.unsupervised.pca import *
from pathlib import Path
from models.data.source import Eikon
from models.data.handler import DataHandler
import os
import matplotlib.pyplot as plt
from models.stat_models.linearregression import *

path_data = fr'{Path(__file__).parents[2]}\models\data'
path_apikeys = r'C:\Users\serge\OneDrive\reuters\apikeys.csv'

rics = [x[:-4] for x in os.listdir(os.path.join(path_data, 'csv'))]

params = {
    'rics': rics,
    'field': ['TR.PriceClose', 'Price Close'],
    'date_field': ['TR.PriceClose.calcdate', 'Calc Date'],
    'load_path': os.path.join(path_data, 'csv')
}

eikon_api = Eikon(path_apikeys)

data = eikon_api.load_timeseries(**params)

preprocessor = DataHandler(data=data,
                           date_col=params['date_field'][1])

period = 7
n_assets = 50
returns = preprocessor.get_returns(period=period)
returns.dropna(axis=1, inplace=True)
T, N = returns.shape

# stability of PCs per time

window = period
n_components = 5
t_start = 200
last_factors = pd.DataFrame(index=returns.index[t_start+period:],
                            data=np.zeros((T-window-t_start, n_components)))

betas = np.zeros((T-window-t_start, n_components+1, n_assets))
betas_r2a = np.zeros((T-window-t_start, n_components+1, n_assets))
eigenvals = pd.DataFrame(index=returns.index[t_start+period:],
                         data=np.zeros((T-window-t_start, n_components)))

for t in np.arange(t_start, T-window):
    print(t)
    r = returns.iloc[:t+window, :n_assets]
    pca = PcaHandler(X=r, demean=True, method='svd')
    r2pca = R2Pca(data=r, rolling_window=window)
    # pca.benchmark_test()
    factors = pca.components(n=n_components)
    factors_r2pca = r2pca.components(n=n_components)
    last_factors.iloc[t-t_start, :] = factors[-1,:]
    eigenvals.iloc[t-t_start, :] = pca.eig_vals[:n_components]
    x_pc = (factors[window-1:, :] - factors[window-1:, :].mean(axis=0)) / factors[window-1:, :].std(axis=0)
    x_r2pc = (factors_r2pca - factors_r2pca.mean(axis=0)) / factors_r2pca.std(axis=0)
    ols = MultiOutputLinearRegressionModel(y=r.iloc[window-1:, :], x=x_pc)
    ols_r2a = MultiOutputLinearRegressionModel(y=r.iloc[window-1:, :], x=x_r2pc)
    betas[t-t_start, :, :] = ols.fit(out=True)
    betas_r2a[t-t_start, :, :] = ols_r2a.fit(out=True)


# time dynamics of PC factors

fig, axs = plt.subplots(3,2, sharex=True, figsize=(11,8))
fig.suptitle(f'Time-series of PCs, with rolling window {window} days')

for i, ax in enumerate(axs.flat[:-1]):
    r2factors = np.concatenate([np.zeros(window-1), factors_r2pca[:, i]])
    time = returns.index.values[1:]
    ax.plot(time, r2factors, label='PCA')
    ax.plot(time, factors[:,i], label='R2 Robust PCA')
    ax.set_xticks(time[::10])
    ax.set_xticklabels(time[::10], rotation=45)
    ax.grid(True)

plt.legend()
plt.show()

# sensitivity of factor loadings (betas) through time

fig, axs = plt.subplots(3,2, sharex=True, figsize=(11,8))
fig.suptitle('Sensitivity of factor loadings (betas) through time')
for i, ax in enumerate(axs.flat):
    time = returns.index[t_start+period:]
    ax.plot(time, betas[:, i, 12:13], label='PCA')
    ax.plot(time, betas_r2a[:, i, 12:13], label='R2 Robust PCA')
    ax.set_xticks(time[::10])
    ax.set_xticklabels(time[::10], rotation=45)
    ax.grid(True)
    ax.set_ylabel(f'Factor loading (beta) no. {i}')
plt.legend()
plt.show()

# sensitivity of PC factors on asset matrix

cnt = 0
n_attempts = 100
pc_bootstrapped_r2 = np.zeros((n_attempts, T-window+1, n_components))
while cnt < n_attempts:
    print(f'Bootstrap attempt: {cnt}')
    idx_sample = np.random.choice(np.arange(0, returns.shape[1]), size=50)
    r = returns.iloc[:, idx_sample]
    # pca = PcaHandler(X=r, demean=True, method='svd')
    r2pca = R2Pca(data=r, rolling_window=window)
    factors = r2pca.components(n=n_components)
    pc_bootstrapped_r2[cnt, :, :] = factors
    cnt += 1



