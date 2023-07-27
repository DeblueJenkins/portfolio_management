import pytest
from models.unsupervised.pca import PcaHandler
from models.stat_models.linearregression import LinearRegressionModel, MultiOutputLinearRegressionModel
from models.stat_models.linearregression import generate_regression_data
from models.stat_models.garch import Garch, generate_garch_data
from optimization.models.portfolio import MarkowitzPortfolio
import pandas as pd
import os
import numpy as np


data = pd.read_csv(os.path.join(os.getcwd(), "testdata", "testdata.csv"), index_col=0)
test_model = PcaHandler(data=data, demean=True)


def test_pca_handler():

    test_model.set_benchmark_comp()
    test_matrix = test_model.cov_data - test_model.cov_pca
    for i in range(np.shape(test_matrix)[0]):
        for j in range(np.shape(test_matrix)[1]):
            assert pytest.approx(test_matrix[i][j], rel=1e-10) == 0.0


def test_multi_linear():
    test_multi_lin_model = MultiOutputLinearRegressionModel(x=test_model.components(n=7),
                                                            y=np.array(data),
                                                            tickers=list(data.columns))
    assert test_multi_lin_model.n == 396
    assert test_multi_lin_model.n_y_s == 69
    test_matrix = pd.read_csv(os.path.join(os.getcwd(), "testdata", "multi_linear_test_betas.csv"), index_col=0)
    test_matrix = np.array(test_matrix) - test_multi_lin_model.betas
    for i in range(np.shape(test_matrix)[0]):
        for j in range(np.shape(test_matrix)[1]):
            assert pytest.approx(test_matrix[i][j], rel=1e-10) == 0.0


def test_lin_reg():
    test_betas = np.array([10, 20, 30, 40, 50])
    sigma_test = 0.4
    x, y = generate_regression_data(betas=test_betas, sigma=sigma_test, n=500000)
    test_reg_model = LinearRegressionModel(x=x, y=y)
    for i in range(len(test_betas)):
        assert pytest.approx(test_betas[i], rel=1e-2) == test_reg_model.beta[i]
    assert pytest.approx(sigma_test, rel=1e-2) == test_reg_model.sigma


def test_garch():
    test_params = [0.005, 0.095, 0.9]
    garch_data, variances = generate_garch_data(n=50000, w=test_params[0], a=test_params[1], b=test_params[2], s2=1.0)
    test_garch_mod = Garch()
    test_garch_mod.estimate_model(data=garch_data)
    for i in range(len(test_params)):
        assert pytest.approx(test_garch_mod.tx[i], abs=1e-2) == test_params[i]


def test_markowitz_portfolio():

    data = np.array([[0.08993464,  0.00934476,  0.14527779,  0.27913437],
                     [0.22507284,  0.31655741,  0.33848865,  0.30662318],
                     [-0.12077173,  0.07518168,  0.03553274,  0.11836059],
                     [0.21105498, -0.1847318, -0.21850225, -0.28724726],
                     [-0.00727867,  0.03587309,  0.0653784,  0.2312663 ],
                     [0.22017315,  0.19787159,  0.15891429,  0.09286841],
                     [0.00406632,  0.12835258,  0.16634935,  0.3003768 ],
                     [0.15713029,  0.0428573,  0.08962161,  0.11046548],
                     [-0.03538538, -0.06407222, -0.08220379, -0.01101051],
                     [0.01155979, -0.0206244, -0.12775172, -0.13768249],
                     [0.17122321,  0.20053017,  0.12929159,  0.09065406],
                     [-0.04897752,  0.16246763,  0.00742009, -0.04326068],
                     [0.1715825,  0.1046309,  0.14100799,  0.18035126],
                     [0.15623638,  0.10399021,  0.1425084,  0.20566116],
                     [0.03823017, -0.0209014, -0.03315573, -0.00611012]])

    model = MarkowitzPortfolio(data)

    m1, v1 = model.global_minimal_variance_portfolio_mean_vol(False)
    assert pytest.approx(m1, rel=1e-8) == 0.08108017159940333
    assert pytest.approx(v1, rel=1e-8) == 0.09097079952957589

    m2, v2 = model.global_minimal_variance_portfolio_mean_vol(True)
    assert pytest.approx(m2, rel=1e-8) == 0.14556930730487985
    assert pytest.approx(v2, rel=1e-8) == 0.04464961045010424

    m3, v3 = model.minimize_sigma_mean_vol(allow_short_selling=False, mean_constraint=0.3)
    assert pytest.approx(m3, rel=1e-8) == 0.09536337000098741
    assert pytest.approx(v3, rel=1e-8) == 0.16915898241613025

    m4, v4 = model.minimize_sigma_mean_vol(allow_short_selling=True, mean_constraint=0.3)
    assert pytest.approx(m4, rel=1e-8) == 0.29999999999999727
    assert pytest.approx(v4, rel=1e-8) == 0.19034548746893312

    m5, v5 = model.optimize_mean_mean_vol(allow_short_selling=False, vol_constraint=0.3)
    assert pytest.approx(m5, rel=1e-8) == 0.09536337000031798
    assert pytest.approx(v5, rel=1e-8) == 0.16915898241493366

    m6, v6 = model.optimize_mean_mean_vol(allow_short_selling=True, vol_constraint=0.3)
    assert pytest.approx(m6, rel=1e-8) == 0.3931619557311211
    assert pytest.approx(v6, rel=1e-8) == 0.3000000000000107
