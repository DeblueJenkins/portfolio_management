from models.data.source import Eikon
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from examples_rics_indices import rics_examples

data_path = r'C:\Users\serge\IdeaProjects\portfolio_manager\data\fama-french-factors'
path_apikeys = r'C:\Users\serge\OneDrive\Documents\apikeys.csv'

rics_list = list(rics_examples.values())

params = {
    'rics': rics_list,
    'field': ['TR.PriceClose', 'Price Close'],
    'date_field': ['TR.PriceClose.calcdate', 'Calc Date'],
    'load_path': r'C:\Users\serge\IdeaProjects\portfolio_manager\portfolio_management\models\data\csv',
    'params': {
        'SDate':'1999-12-31',
        'EDate': '2021-06-04',
        'Curn':'Native',
    },
}

config = {
    'PCA': True,
    'n_components': 7,
    'OLS': True
}
eikon_api = Eikon(path_apikeys)
save_path = params['load_path']
params.pop('load_path')
data = eikon_api.download_timeseries(**params, save_config={'save': True, 'path': save_path})
data.set_index('Calc Date', inplace=True)

# sample weight can be exp. weighted
model = KMeans(n_clusters=5, n_init=10, verbose=True, random_state=0, algorithm='elkan')
model.fit(data.T)


df = pd.DataFrame(data=np.concatenate([data, model.labels_.reshape(len(data),1)], axis=1),
                  columns=rics_list + ['labels'])
