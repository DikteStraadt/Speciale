import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, make_scorer, accuracy_score, f1_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from FeatureEngineering import FeatureSelection as f
import Report as r
from mlxtend.feature_selection import SequentialFeatureSelector as SFS


class RandomForest:

    def __init__(self, X_train, X_test, y_train, y_test, target):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.target = target

    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):

        model = Pipeline(steps=[
            ("featureselection", SFS(estimator=RandomForestClassifier())),
            ("randomforest", RandomForestClassifier()),
        ])

        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'f1_micro': make_scorer(f1_score, average='micro'),
            'f1_macro': make_scorer(f1_score, average='macro'),
            'f1_weighted': make_scorer(f1_score, average='weighted'),
        }

        param = {
            'featureselection__k_features': [2, 5],
            'featureselection__forward': [True],
            'featureselection__floating': [False],
            'featureselection__scoring': ['accuracy'],
            'featureselection__cv': [5],
            'randomforest__n_estimators': [100, 200, 300, 400, 500, 600, 700],
            # 'randomforest__max_depth': [None, 10, 20, 30, 40, 50],
            # 'randomforest__min_samples_split': [2, 5, 10],
            # 'randomforest__min_samples_leaf': [1, 2, 4],
            # 'randomforest__max_features': ['auto', 'sqrt', 'log2'],
            # 'randomforest__bootstrap': [True, False],
            # 'randomforest__class_weight': [None, 'balanced'],
            # 'randomforest__random_state': [123]
        }

        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param,
            n_iter=1,
            cv=2,
            n_jobs=-1,
            random_state=123,
            scoring=scoring,
            refit='f1_macro',
            verbose=4
        )

        random_search.fit(self.X_train, self.y_train)

        r.write_to_report("features (random forest)", str(data.colums.toList()))
        r.write_to_report("best model (random forest)", str(random_search.best_estimator_))
        r.write_to_report("best parameters (random forest)", str(random_search.best_params_))
        r.write_to_report("accuracy  (random forest)", random_search.best_estimator_.score(self.X_test, self.y_test))