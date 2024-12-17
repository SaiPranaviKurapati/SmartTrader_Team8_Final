import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin

class XGBRegressorWrapper(BaseEstimator, RegressorMixin):
    _estimator_type = "regressor"
    
    def __init__(self, **kwargs):
        self.model = xgb.XGBRegressor(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)
    
    def get_params(self, deep=True):
        return self.model.get_params(deep=deep)

    def set_params(self, **params):
        self.model.set_params(**params)
        return self
