from sklearn.linear_model import LinearRegression
import pandas as pd

from src.models.forecasting_algorithm import ForecastingAlgorithm


class FA_LinearRegression(ForecastingAlgorithm):

    model_class = LinearRegression
    default_params = {'fit_intercept': True}
    default_grid_search_params = {'fit_intercept': [True, False]}

    def get_features_importances(self, X):
        features = pd.DataFrame({
            'Feature': list(X.columns) + ['Intercept'],
            'Importance': list(self.model.coef_) + [self.model.intercept_]
        })
        return features
