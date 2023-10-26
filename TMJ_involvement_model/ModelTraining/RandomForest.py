import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, make_scorer, accuracy_score, f1_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
import Report as r


class RandomForest:

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):

        model = Pipeline(steps=[
            ("RandomForest", RandomForestClassifier()),
        ])

        scoring = {
            'accuracy': make_scorer(accuracy_score),
            # 'f1': make_scorer(f1_score),
            # 'f1_macro': make_scorer(f1_score, average='macro'),
            # 'f1_micro': make_scorer(f1_score, average='micro')
        }

        param_grid = {
            'RandomForest__n_estimators': [100, 200, 300, 400, 500],
            # 'RandomForest__max_depth': [None, 10, 20, 30, 40, 50],
            # 'RandomForest__min_samples_split': [2, 5, 10],
            # 'RandomForest__min_samples_leaf': [1, 2, 4],
            # 'max_features': ['auto', 'sqrt', 'log2'],
            # 'bootstrap': [True, False],
            # 'class_weight': [None, 'balanced'],
            'RandomForest__random_state': [123]
        }

        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=10,
            cv=5,
            n_jobs=-1,
            random_state=123,
            scoring=scoring,
            refit='accuracy'
        )

        model = random_search.fit(self.X_train, self.y_train)

        best_model = random_search.best_estimator_
        best_parameters = random_search.best_params_
        accuracy = random_search.best_estimator_.score(self.X_test, self.y_test)

        print("Best Params: ", best_parameters)
        print("Accuracy: ", accuracy)

        r.write_to_report("best model", str(best_model))
        r.write_to_report("best parameters", str(best_parameters))
        r.write_to_report("accuracy", accuracy)

        return model