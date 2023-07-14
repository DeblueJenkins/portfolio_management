import pytest
from models.unsupervised.pca import PcaHandler
from models.stat_models.linearregression import LinearRegressionModel, generate_regression_data
from models.stat_models.garch import Garch, generate_garch_data
import pandas as pd
import os
import numpy as np


data = pd.read_csv(os.path.join(os.getcwd(), "testdata", "testdata.csv"), index_col=0)


def test_pca_handler():
    test_model = PcaHandler(data=data, demean=True)
    test_model.set_benchmark_comp()
    test_matrix = test_model.cov_data - test_model.cov_pca
    for i in range(np.shape(test_matrix)[0]):
        for j in range(np.shape(test_matrix)[1]):
            assert pytest.approx(test_matrix[i][j], rel=1e-10) == 0.0


def test_lin_reg():
    test_betas = np.array([10, 20, 30, 40, 50])
    sigma_test = 0.4
    x, y = generate_regression_data(betas=test_betas, sigma=sigma_test, n=500000)
    test_model = LinearRegressionModel(x=x, y=y)
    for i in range(len(test_betas)):
        assert pytest.approx(test_betas[i], rel=1e-3) == test_model.beta[i]
    assert pytest.approx(sigma_test, rel=1e-3) == test_model.sigma

