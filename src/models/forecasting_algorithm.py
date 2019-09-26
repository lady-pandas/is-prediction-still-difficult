from abc import ABC, abstractmethod
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.base import RegressorMixin

import numpy as np
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error, make_scorer


def weighted_mape(y_true, y_pred, sample_weight):
    if sample_weight is None:
        sample_weight = [1]*len(y_pred)
    else:
        sample_weight = sample_weight.loc[y_true.index.values].values

    return mean_absolute_error(y_true, y_pred, sample_weight=sample_weight)


class ForecastingAlgorithm(ABC):

    def __init__(self, confidence_level=95, cv=5, model_params=None, sample_weight_series=None):
        self.confidence_level = confidence_level
        self.cv = cv
        params = model_params if model_params is not None else self.default_params
        self.model = self.model_class(**params)

        self.sample_weight_series = sample_weight_series
        score_params = {"sample_weight": self.sample_weight_series}
        self.scorer = make_scorer(weighted_mape, greater_is_better=False, **score_params)

        self.gs = None
        if not issubclass(self.model_class, RegressorMixin):
            raise TypeError("Provided model class is not scikit-learn Estimator: {}".format(self.model_class()))

    @property
    @abstractmethod
    def model_class(self):
        pass

    @property
    @abstractmethod
    def default_params(self):
        pass

    @property
    @abstractmethod
    def default_grid_search_params(self):
        pass

    def get_features_importances(self, X):
        # Default setup:
        features = pd.DataFrame({
            'Feature': X.columns,
            'Importance': None
        })
        return features

    def get_levels(self):
        lower_level = (100.0 - self.confidence_level) / 2.0
        upper_level = 100.0 - lower_level
        return lower_level, upper_level

    def cross_val_score(self, X, y):
        if type(y) != pd.Series:
            y = pd.Series(y, index=X.index)

        y_vals = []
        for train_index, val_index in self.cv.split(X):
            y_val, y_val_down, _ = self.fit_predict(X.loc[train_index, :], y.loc[train_index],
                                                    X.loc[val_index, :])
            y_val = pd.DataFrame(y_val, index=val_index)
            y_vals.append(y_val)

        y_vals = pd.concat(y_vals).sort_index()
        y = y.sort_index()

        return weighted_mape(y, y_vals, self.sample_weight_series)

    def predict_ci(self, X_train, y_train, X_test):
        stdev = np.sqrt(sum((self.model.predict(X_train) - y_train) ** 2) / (len(y_train) - 2))
        stdev_scaled = norm.ppf(self.get_levels()[1] / 100.0) * stdev
        test_predictions = self.model.predict(X_test)
        return test_predictions - stdev_scaled, test_predictions + stdev_scaled

    def do_grid_search(self, X, y, grid_search_params=None):
        if type(y) != pd.Series:
            y = pd.Series(y)

        gs_params = grid_search_params if grid_search_params is not None else self.default_grid_search_params
        gs = GridSearchCV(self.model_class(**self.default_params), gs_params, cv=self.cv, scoring=self.scorer)
        gs.fit(X, y)

        self.model = gs.best_estimator_
        self.gs = gs

    def fit_predict(self, X_train, y_train, X_test):
        self.model.fit(X_train, y_train)
        features_importances = self.get_features_importances(X_train)
        y_test_pred = self.model.predict(X_test)
        y_test_down_pred, _ = self.predict_ci(X_train, y_train, X_test)

        return y_test_pred, y_test_down_pred, features_importances
