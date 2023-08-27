import numpy as np
import time
from linear_programming import AbstractModel

class ModelSelector:


    def __init__(self, model: AbstractModel):
        self.model = model

    def get_exponential_weights(self, n, decay):
        weights = decay ** np.arange(1, n+1)
        weights = weights / np.sum(weights)
        return weights

    # this should be in ModelSelector class
    def tune_hyperparams(self):
        # decay_candidates = np.arange(0.05, 0.35, 0.05)
        lambda_candidates = np.arange(0,1,0.05)
        trials = {}
        # for d in decay_candidates:
        for l in lambda_candidates:
            metrics = self.cross_validation(l, data=self.data)
            trials[np.round(l,2)] = metrics['rmse_diff']
        return trials
        # this should be in ModelSelector class
    def cross_validation(self, l, data, size_of_train: int = 200):
        # this is essentiallu model selection procedure
        # there need to be sample weights here for OLS
        # L1 or L2 regularization
        # other params
        # this should be done for the individual regressions
        T, n = data.shape
        errors_in_sample = []
        errors_out_sample = []
        t0 = time.time()
        for i in range(T-size_of_train):
            data_train = data[:i+size_of_train, :]
            pca_model = PcaHandler(data_train, demean=True)
            factors_train = pca_model.components(self.config['MODEL']['main_factors']['PCA']).copy()
            # weights = self.get_exponential_weights(len(data_train), decay)
            regr_model = MultiOutputLinearRegressionModel(x=factors_train,
                                                          y=data_train,
                                                          method=self.config['MODEL']['regression_method'],
                                                          l1=l,
                                                          w=None)
            regr_model.fit()
            # choice to be made, whether to predict with already fit params or not
            data_test = data[:i+size_of_train+1, :]
            pca_model_test = PcaHandler(data_test, demean=True)
            factors_test = pca_model_test.components(self.config['MODEL']['main_factors']['PCA']).copy()
            y_pred_out_sample = regr_model.predict(factors_test[-1])
            y_pred_in_sample_last = regr_model.predict(factors_train[-1])
            in_sample_error = data[size_of_train] - y_pred_in_sample_last
            out_sample_error = data[size_of_train+1] - y_pred_out_sample

            errors_in_sample.append(in_sample_error)
            errors_out_sample.append(out_sample_error)

        print(f'CV finished: {time.time() - t0}')

        errors_in_sample = np.array(errors_in_sample).flatten()
        errors_out_sample = np.array(errors_out_sample).flatten()

        metrics = {
            'rmse_in': np.sqrt(np.mean(errors_in_sample ** 2)),
            'rmse_out': np.sqrt(np.mean(errors_out_sample ** 2)),
            'mae_in': np.mean(np.abs(errors_in_sample)),
            'mae_out': np.mean(np.abs(errors_out_sample))
        }

        metrics['rmse_diff'] = metrics['rmse_in'] - metrics['rmse_out']
        metrics['mae_diff'] = metrics['mae_in'] - metrics['mae_out']

        return metrics