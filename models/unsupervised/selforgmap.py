import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import warnings

# data science (I did not have sklean installed lol)
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler


class SelfOrgMap:

    def __init__(self, df: pd.DataFrame, map_dimensions: tuple, learning_rate: float = 0.5, sigma: float = 5,
                 random_seed: int = 42):

        plt.style.use('fivethirtyeight')
        warnings.filterwarnings('ignore')

        self.df = df
        self.x = np.array(self.df).T
        self.tickers = list(self.df.columns)
        scaler = MinMaxScaler()
        self.x_tilde = scaler.fit_transform(self.x)
        self.som = MiniSom(map_dimensions[0], map_dimensions[1], self.x_tilde.shape[1], learning_rate=learning_rate,
                           sigma=sigma, random_seed=random_seed)
        self.som.random_weights_init(self.x_tilde)
        self.result = None

    def run(self, show_plot: bool = True):

        self.som.train_batch(self.x_tilde, 10000, verbose=True)

        if show_plot:
            plt.figure(figsize=(25, 25))
            for ix in range(len(self.x_tilde)):
                winner = self.som.winner(self.x_tilde[ix])
                plt.text(winner[0], winner[1], self.tickers[ix], bbox=dict(facecolor='white', alpha=0.5, lw=0))
            plt.imshow(self.som.distance_map())
            plt.grid(False)
            plt.title('Self Organizing Map')

        x_axis_array = []
        y_axis_array = []

        for ix in range(len(self.x_tilde)):
            winner = self.som.winner(self.x_tilde[ix])
            x_axis_array.append(winner[0])
            y_axis_array.append(winner[1])

        self.result = pd.DataFrame({"x_map": x_axis_array, "y_map": y_axis_array}, index=self.tickers)

        return self.result

