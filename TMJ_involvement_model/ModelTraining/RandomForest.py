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

    def __init__(self, X_train, X_test, y_train, y_test, config):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.config = config

    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):

        self.X_train = self.X_train.drop(columns=['ID'])
        self.X_train = self.X_train.drop(columns=['ageatvisitation'])
        self.X_train = self.X_train.drop(columns=['difftdate'])
        self.X_test = self.X_test.drop(columns=['ID'])
        self.X_test = self.X_test.drop(columns=['ageatvisitation'])
        self.X_test = self.X_test.drop(columns=['difftdate'])

        data_fs = f.feature_selection(self.X_train, self.y_train, self.X_test, RandomForestClassifier(), self.config)

        self.X_train = data_fs[0]
        self.X_test = data_fs[1]

        r.write_to_report("(random forest) n_features", len(self.X_train.columns))
        r.write_to_report("(random forest) feature names", self.X_train.columns.tolist())

        model = Pipeline(steps=[
            ("random forest", RandomForestClassifier()),
        ])

        param = {
            'random forest__n_estimators': [100, 300, 700, 1000],
            'random forest__max_depth': [None, 3, 7, 10],
            'random forest__min_samples_split': [2, 5, 10],
            'random forest__min_samples_leaf': [1, 5, 10, 20, 50],
            'random forest__max_features': ['sqrt', 'log2', 0.5],
            'random forest__bootstrap': [True, False],
            'random forest__class_weight': ['balanced'],
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
            scoring='f1_macro',
            refit='f1_macro',
            verbose=self.config["verbose"]
        )

        random_search.fit(self.X_train, self.y_train)

        e.evaluation("random forest", random_search, self.X_test, self.y_test)

        return data