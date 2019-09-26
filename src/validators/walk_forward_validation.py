import pandas as pd
from sklearn.model_selection import BaseCrossValidator


class WalkForwardValidator(BaseCrossValidator):

    split_dates = {}

    def __init__(self, n_splits, date_col):
        self.n_splits = int(n_splits)
        self.date_col = date_col

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def _iter_test_indices(self, X=None, y=None, groups=None):
        iterator = []
        max_date = self.date_col.max()
        max_splits = len(self.date_col.unique()) - 1
        if self.n_splits > max_splits:
            raise ValueError('Too many splits, maximal nr of splits is {0}'.format(max_splits))

        split_dates = pd.date_range(end=max_date, periods=self.n_splits, freq=pd.offsets.MonthBegin())
        for i, date in enumerate(split_dates):
            self.split_dates[i] = date.date()
            test_indices = X.loc[self.date_col >= date.date(), :].index.values.astype(int)
            iterator.append(test_indices)

        return iterator
