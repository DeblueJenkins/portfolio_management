from models.data.source import Eikon
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from examples_rics_indices import rics_examples
from multiprocessing.pool import ThreadPool
import multiprocessing as mp
import matplotlib.pyplot as plt
import pickle
from models.data.handler import DataHandler

data_path = r'C:\Users\serge\IdeaProjects\portfolio_manager\portfolio_management\models\data'
path_apikeys = r'C:\Users\serge\OneDrive\Documents\apikeys.csv'

eikon_api = Eikon(path_apikeys)

index = '.SPX'

get_constituents = False
download = False

if get_constituents:
    rics_list = eikon_api.get_index_constituents()
    with open(fr'{data_path}\rics_of_{index.replace(".","")}.pkl', 'wb') as f:
        pickle.dump(rics_list, f)
        print('Pickled.')

else:
    rics_list = pickle.load(open(fr'{data_path}\rics_of_{index.replace(".","")}.pkl', 'rb'))
    print('Unpickled.')

rics_list = np.random.choice(rics_list, 150)





params_download = {
    'rics': rics_list,
    'field': ['TR.PriceClose', 'Price Close'],
    'date_field': ['TR.PriceClose.calcdate', 'Calc Date'],
    'save_config': {
        'path': r'C:\Users\serge\IdeaProjects\portfolio_manager\portfolio_management\models\data\csv',
        'save': True
    },
    'params': {
        'SDate':'1999-12-31',
        'EDate': '2021-06-04',
        'Curn':'Native',
    },
}

params_load = {
    'rics': rics_list,
    'field': ['TR.PriceClose', 'Price Close'],
    'date_field': ['TR.PriceClose.calcdate', 'Calc Date'],
    'load_path': r'C:\Users\serge\IdeaProjects\portfolio_manager\portfolio_management\models\data\csv'

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

## returns

# X = preprocessor.get_overlapping_returns(period=25)
# X = preprocessor.get_returns(period=25)

## prices
X = (data - data.mean(axis=0)) / data.std(axis=0)


X = X.dropna(axis=1).T.values

idx = np.random.choice(np.arange(X.shape[0]), 30)
rnd_eql_portfolio_1 = X[idx, :].mean(axis=0)
idx = np.random.choice(np.arange(X.shape[0]), 30)
rnd_eql_portfolio_2 = X[idx, :].mean(axis=0)



X.shape

def fit_kmeans(k, data):
    # print(f'Fitting n_clusters: {k}')
    model = KMeans(n_clusters=k, n_init=50, verbose=False, random_state=0, algorithm='elkan')
    model.fit(data)
    return model.inertia_

wrapper = lambda x: fit_kmeans(x, X)

pool = ThreadPool(mp.cpu_count())
res = pool.map(wrapper, np.arange(1,10,1))

fig = plt.figure(figsize=(11,8))
plt.plot(res, ls=':', marker='o')
plt.ylabel('Sum of squared errors')
plt.xlabel('Number of clusters k')
plt.grid(True)
plt.show()

k = input('Specify number of cluster k: ')
k = int(k)

model = KMeans(n_clusters=k, n_init=50, verbose=False, random_state=0, algorithm='elkan')
model.fit(X)

print(np.bincount(model.labels_))

df_labels = pd.Series(index=X.T.index, data=model.labels_, name='cluster')
df = pd.merge(X.T, df_labels, left_index=True, right_index=True)

data_reduced = df.groupby('cluster').mean().T

print(data_reduced.corr())

data_reduced.plot(figsize=(11,8), marker='o', ls=':')
