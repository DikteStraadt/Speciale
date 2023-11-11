import pandas as pd
from matplotlib import pyplot
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from FeatureEngineering import FeatureSelection as f
from ModelEvaluation import Evaluation as e
from Utils import Report as r

class RandomForest:

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

        data_fs = f.feature_selection(data, self.X_train, self.X_test, RandomForestClassifier(), self.target, self.config)

        self.X_train = data_fs[0]
        self.X_test = data_fs[1]

        model = Pipeline(steps=[
            ("random forest", RandomForestClassifier()),
        ])

        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'f1_micro': make_scorer(f1_score, average='micro'),
            'f1_macro': make_scorer(f1_score, average='macro'),
            'f1_weighted': make_scorer(f1_score, average='weighted'),
        }

        param = {
            'random forest__n_estimators': [100, 200, 300, 500, 700, 1000, 100000],
            'random forest__max_depth': [None, 3, 7, 10],
            'random forest__min_samples_split': [2, 5, 10],
            'random forest__min_samples_leaf': [1,5,10,20,50],
            'random forest__max_features': ['sqrt', 'log2', 0.5],
            'random forest__bootstrap': [True, False],
            'random forest__class_weight': [[1, 2, 0.5], 'balanced', [0.5, 2, 1]],
            'random forest__min_impurity_decrease': [0.0, 1e-5, 1e-4, 0.01, 0.1],
            'random forest__criterion': ['gini', 'entropy'],
            'random forest__ccp_alpha': [0.0, 0.01, 0.1],
            'random forest__random_state': [42],
        }

        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param,
            n_iter=self.config["iterations"],
            cv=self.config["cv"],
            n_jobs=-1,
            random_state=42,
            scoring=scoring,
            refit='f1_weighted',
            verbose=self.config["verbose"]
        )

        random_search_model = random_search.fit(self.X_train, self.y_train)

        e.evaluation("random forest", random_search_model, self.X_train, self.X_test, self.y_test)

        return data