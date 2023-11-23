from catboost import CatBoostClassifier
from types import MethodType

class CatBoostWrapper:
    def __init__(self, model):
        self.model = model

    def fit(self, X, y, sample_weight=None):
        return self.model.fit(X, y, sample_weight=sample_weight)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

        ## how to be used in main
        ## catboost_model = CatBoostClassifier
        ## wrapper_model = CatBoostWrapper(catboost_model)
        ## cp.conformancePrediction(wrapper_model.model, X_uncertainty, y_uncertainty, X_test, y_test)

