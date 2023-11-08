from matplotlib import pyplot
from FeatureEngineering import FeatureSelection as f
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, make_scorer, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
from Utils import Report as r
from ModelEvaluation import Evaluation as e

class XGBoostClassifier:

    def __init__(self, X_train, X_test, y_train, y_test, target, config):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.target = target
        self.config = config

    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):

        data_fs = f.feature_selection(data, self.X_train, self.X_test, xgb.XGBClassifier(), self.target, self.config)

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
            'xgboost__enable_categorical': [True],
            'xgboost__max_depth': [3, 7, 10],
            'xgboost__eta': [0.01, 0.1, 0.2],
            'xgboost__objective': [xgboost_objective],
            'xgboost__min_child_weight': [1, 5, 15, 30, 100, 200],
            'xgboost__colsample_bytree': [0.8, 0.9, 1],
            'xgboost__n_estimators': [100, 200, 300, 500, 700, 1000],
            'xgboost__reg_alpha': [0.5, 0.2, 1],
            'xgboost__reg_lambda': [2, 3, 5],
            'xgboost__gamma': [1, 2, 3],
            'xgboost__random_state': [42]
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

        e.evaluation("xgboost", random_search_model, self.X_train, self.X_test, self.y_test)

        return data