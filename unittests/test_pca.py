import pytest
from models.unsupervised.pca import RollingPcaHandler, PcaHandler
import numpy as np
import pandas as pd
@pytest.fixture
def rolling_pca_handler():

    T = 250
    N = 80
    window = 50
    n_components = 5
    data = np.random.rand(T,N)
    dates = pd.date_range('2000-01-01', freq='W', periods=T)
    data = pd.DataFrame(index=dates, data=data)
    return RollingPcaHandler(PcaHandler, data, n_components, demean=True, rolling_window=window)


def test_estimate(rolling_pca_handler):

    pcs = rolling_pca_handler.estimate(method='ordinary')

    dim1 = rolling_pca_handler.T - rolling_pca_handler.rolling_window
    assert pcs.shape[0] == dim1