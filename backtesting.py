import numpy as np
import matplotlib.pyplot as plt


class PerformanceAssesser:

    def __init__(self, start_date, end_date, weights):

        self.start_date = start_date
        self.end_date = end_date
        self.weights = weights
        self.n = len(weights)


    def get_historical_returns(self, returns: np.ndarray):
        # the start date and end date are the same by design of the
        # returns matrix
        self.historical_returns = self.weights.T.dot(returns.T)
        self.historical_mean_return = np.mean(self.historical_returns)
        self.historical_vol = np.std(self.historical_returns)

    def get_benchmark_index_returns(self, eikon, index_ric, field, date_field):

        params = {
                'SDate': self.start_date,
                'EDate': self.end_date,
                'Curn':'Native',
            }

        data = eikon.api.get_data(index_ric, [field, date_field], parameters=params)[0]
        logr = np.log(data['Price Close'] / data['Price Close'].shift(1))



