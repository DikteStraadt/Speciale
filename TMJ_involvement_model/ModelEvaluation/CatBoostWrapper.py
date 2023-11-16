from catboost import CatBoostClassifier
from types import MethodType

class CatBoostWrapper:
    def __init__(self, model):
        self.model = model
        self.model_feature_names_in_ = getattr(model, 'feature_names_', None)

        # dummy method to avoid issues with MAPIE
        self.model.predict_proba = MethodType(lambda self, X: self.predic(X, prediction_type='Probability'), self.model)

        ## how to be used in main
        ## catboost_model = CatBoostClassifier
        ## wrapper_model = CatBoostWrapper(catboost_model)
        ## cp.conformancePrediction(wrapper_model.model, X_uncertainty, y_uncertainty, X_test, y_test)

