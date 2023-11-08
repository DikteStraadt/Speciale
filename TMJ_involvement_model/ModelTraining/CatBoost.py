import pandas as pd
from imblearn.pipeline import Pipeline
from matplotlib import pyplot
from sklearn.metrics import make_scorer, accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostClassifier
from FeatureEngineering import FeatureSelection as f
from Utils import Report as r
from ModelEvaluation import Evaluation as e

class CatBoost:

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

        data_fs = f.feature_selection(data, self.X_train, self.X_test, CatBoostClassifier(), self.target, self.config)

        self.X_train = data_fs[0]
        self.X_test = data_fs[1]

        non_categorical_columns = ['openingmm', 'opening', 'protrusionmm', 'protrusion', 'laterotrusionrightmm', 'laterotrusionleftmm', 'drug', 'asypupilline', 'asybasis', 'asyoccl', 'asymenton', 'profile', 'asyupmid', 'asylowmi', 'lowerface', 'sagittalrelationright', 'sagitalrelationleft']
        categorical_columns = [col for col in self.X_train.columns if col not in non_categorical_columns]

        model = Pipeline(steps=[
            ("catboost", CatBoostClassifier(cat_features=categorical_columns)),
        ])

        feature_names = self.X_train.columns.tolist()
        model.named_steps['catboost'].set_feature_names(feature_names)

        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'f1_micro': make_scorer(f1_score, average='micro'),
            'f1_macro': make_scorer(f1_score, average='macro'),
            'f1_weighted': make_scorer(f1_score, average='weighted'),
        }

        param = {
            'catboost__iterations': [100, 500],
            'catboost__learning_rate': [0.1, 0.2, 1],
            'catboost__depth': [3, 7, 10, None],
            'catboost__l2_leaf_reg': [3],
            'catboost__border_count': [254],
            'catboost__thread_count': [4],
            'catboost__one_hot_max_size': [20],
            'catboost__bagging_temperature': [1.0],
            'catboost__early_stopping_rounds': [50],
            'catboost__loss_function': ['MultiClass'],
            'catboost__verbose': [False]
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

        e.evaluation("catboost", random_search_model, self.X_train, self.X_test, self.y_test)

        return data