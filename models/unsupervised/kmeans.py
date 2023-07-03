import pandas as pd


class KMeansHandler:

    def __init__(self, data: pd.DataFrame, standardize=True):
        """
        This is a k-means clustering handler which inherits from
        :param data: pd.DataFrame, indexed by RIC constituent and columns need to be time.
        Be careful with this.
        :param standardize: bool, if True, standardize the data
        """

        self.data = data
        if standardize:
            self.data = (self.data - self.data.mean(axis=1)) / self.data.std(axis=1)

