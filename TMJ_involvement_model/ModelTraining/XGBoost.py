import pandas as pd
from matplotlib import pyplot
from FeatureEngineering import FeatureSelection as f
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, make_scorer, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
from Utils import Report as r
from ModelEvaluation import Evaluation as e

class XGBoostClassifier:

    def __init__(self, X_train, X_test, y_train, y_test, config):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.config = config

    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):

        # ids_X_train = self.X_train["ID"]
        # ids_X_test = self.X_test["ID"]
        self.X_train = self.X_train.drop(columns=['ID'])
        self.X_test = self.X_test.drop(columns=['ID'])

        data_fs = f.feature_selection(self.X_train, self.y_train, self.X_test, xgb.XGBClassifier(), self.config)

        self.X_train = data_fs[0]
        self.X_test = data_fs[1]

        if self.config['n_categories'] == 2:
            xgboost_objective = 'binary:logistic'
        elif self.config['n_categories'] == 3:
            xgboost_objective = 'multi:softmax'

        model = Pipeline(steps=[
            ("xgboost", xgb.XGBClassifier()),
        ])

        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'f1_micro': make_scorer(f1_score, average='micro'),
            'f1_macro': make_scorer(f1_score, average='macro'),
            'f1_weighted': make_scorer(f1_score, average='weighted'),
        }

        fit_params = {"early_stopping_rounds": 50}

        param = {
            'xgboost__max_depth': [3],
            'xgboost__eta': [0.2],
        }

        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param,
            #num_boost_round=100000,
            n_iter=self.config["iterations"],
            #fit_params=fit_params,
            cv=self.config["cv"],
            n_jobs=-1,
            random_state=42,
            scoring=scoring,
            refit='f1_weighted',
            verbose=self.config["verbose"]
        )

        random_search_model = random_search.fit(self.X_train, self.y_train)

        # self.X_train = pd.concat([ids_X_train, data], axis=1)
        # self.X_test = pd.concat([ids_X_test, data], axis=1)

        e.evaluation("xgboost", random_search_model, self.X_test, self.y_test)

        return data