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
        self.X_train = self.X_train.drop(columns=['sex'])
        self.X_train = self.X_train.drop(columns=['ageatvisitation'])
        self.X_train = self.X_train.drop(columns=['difftdate'])
        self.X_test = self.X_test.drop(columns=['ID'])
        self.X_test = self.X_test.drop(columns=['sex'])
        self.X_test = self.X_test.drop(columns=['ageatvisitation'])
        self.X_test = self.X_test.drop(columns=['difftdate'])

        self.X_train = self.X_train.loc[:, self.X_train.nunique() > 1]

        self.X_train.to_excel(f"Data/before_feature_selection_x_train_cat.xlsx")

        data_fs = f.feature_selection(self.X_train, self.y_train, self.X_test, CatBoostClassifier(iterations=self.config['catboost_SFS_iterations'], allow_const_label=True), self.config)

        self.X_train = data_fs[0]
        self.X_test = data_fs[1]

        self.X_train.to_excel(f"Data/after_feature_selection_x_train_cat.xlsx")


        r.write_to_report("(catboost) n_features", len(self.X_train.columns))
        r.write_to_report("(catboost) feature names", self.X_train.columns.tolist())

        non_categorical_columns = ['overjet', 'openbite', 'overbite', 'deepbite', 'openingmm', 'opening', 'protrusionmm', 'protrusion', 'laterotrusionrightmm', 'laterotrusionleftmm', 'asypupilline', 'asybasis', 'asyoccl', 'asymenton', 'profile', 'asyupmid', 'asylowmi', 'lowerface', 'sagittalrelation']
        categorical_columns = [col for col in self.X_train.columns if col not in non_categorical_columns]

        print(categorical_columns)

        model = Pipeline(steps=[
            ("catboost", CatBoostClassifier(cat_features=categorical_columns)),
        ])

        feature_names = self.X_train.columns.tolist()

        print(feature_names)

        model.named_steps['catboost'].set_feature_names(feature_names)

        if self.config['n_categories'] == 2:
            param = {
                'catboost__num_trees': [100, 300, 700, 1000],
                'catboost__learning_rate': [0.01, 0.1, 0.2, 0.3],
                'catboost__max_depth': [3, 5, 7, 10],
                'catboost__l2_leaf_reg': [1, 3, 5, 10],
                'catboost__border_count': [32, 64, 128],
                'catboost__thread_count': [4],
                'catboost__bagging_temperature': [1.0, 1.5, 2.0],
                'catboost__colsample_bylevel': [0.3, 0.5, 0.8, 1.0],
                'catboost__random_strength': [0.1, 0.5, 1.0],
                'catboost__grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide'],
                'catboost__min_data_in_leaf': [1, 5, 10, 20, 50],
                'catboost__class_weights': [[1, 2], [0.5, 2], [0.5, 4]],
                'catboost__early_stopping_rounds': [50],
                'catboost__random_seed': [42],
            }
        elif self.config['n_categories'] == 3:
            param = {
                'catboost__iterations': [100, 300, 700, 1000],
                'catboost__learning_rate': [0.01, 0.1, 0.2, 0.3],
                'catboost__max_depth': [3, 5, 7, 10],
                'catboost__l2_leaf_reg': [1, 3, 5, 10],
                'catboost__border_count': [32, 64, 128],
                'catboost__thread_count': [4],
                'catboost__bagging_temperature': [1.0, 1.5, 2.0],
                'catboost__colsample_bylevel': [0.3, 0.5, 0.8, 1.0],
                'catboost__random_strength': [0.1, 0.5, 1.0],
                'catboost__grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide'],
                'catboost__min_data_in_leaf': [1, 5, 10, 20, 50],
                'catboost__class_weights': [[1, 2, 0.5], [0.5, 2, 1], [0.5, 1, 2]],
                'catboost__early_stopping_rounds': [50],
                'catboost__loss_function': ['MultiClass'],
                'catboost__random_seed': [42],
            }

        catboost = RandomizedSearchCV(
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

        self.X_train.to_excel(f"Data/before_fit_x_train_cat.xlsx")

        catboost.fit(self.X_train, self.y_train)

        e.evaluation("catboost", catboost, self.X_test, self.y_test)

        return data
