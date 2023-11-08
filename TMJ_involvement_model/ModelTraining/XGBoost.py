from matplotlib import pyplot
from FeatureEngineering import FeatureSelection as f
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, make_scorer, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
from Utils import Report as r
import pandas as pd

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

        if self.config["feature_statistical"]:
            sfs_data = f.ForwardSubsetSelection(xgb.XGBClassifier(), self.target, self.config).transform(data)
            self.X_train = self.X_train.loc[:, sfs_data.columns]
            self.X_test = self.X_test.loc[:, sfs_data.columns]
        else:
            clinical_columns = ['drug', 'painmoveleft', 'painmoveright', 'asybasis', 'asyoccl', 'profile', 'lowerface',
                                'laterpalpright', 'laterpalpleft', 'translationright', 'translationleft', 'openingmm',
                                'opening', 'protrusionmm', 'protrusion', 'laterotrusionrightmm', 'laterotrusionleftmm',
                                'overjet', 'overbite', 'openbite', 'chewingfunction', 'retrognathism', 'deepbite',
                                'Krepitationright', 'Krepitationleft']
            X_train_fs = self.X_train.loc[:, clinical_columns]
            X_test_fs = self.X_test.loc[:, clinical_columns]

            extra = ['asypupilline_0.0', 'asypupilline_1.0', 'asypupilline_2.0', 'asypupilline_3.0', 'asypupilline_4.0']

            for column in extra:
                if column in self.X_train.columns:
                    X_train_fs = pd.concat([X_train_fs, self.X_train[column]], axis=1)
                    X_test_fs = pd.concat([X_test_fs, self.X_train[column]], axis=1)

            r.write_to_report("feature selection", "Clinical")
            r.write_to_report(f"n_features", len(clinical_columns))

        model = Pipeline(steps=[
            ("xgboost", xgb.XGBClassifier()),
        ])

        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'f1_micro': make_scorer(f1_score, average='micro'),
            'f1_macro': make_scorer(f1_score, average='macro'),
            'f1_weighted': make_scorer(f1_score, average='weighted'),
        }

        param = {
            'xgboost__enable_categorical': [True],
            'xgboost__max_depth': [3, 7, 10],
            'xgboost__eta': [0.01, 0.1, 0.2],
            'xgboost__objective': ['multi:softmax'],
            'xgboost__min_child_weight': [10, 15, 20, 25],
            'xgboost__colsample_bytree': [0.8, 0.9, 1],
            'xgboost__n_estimators': [300, 400, 500, 600],
            'xgboost__reg_alpha': [0.5, 0.2, 1],
            'xgboost__reg_lambda': [2, 3, 5],
            'xgboost__gamma': [1, 2, 3],
            'xgboost__random_state': [self.config["random_state"]]
        }

        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param,
            n_iter=self.config["iterations"],
            cv=self.config["cv"],
            n_jobs=-1,
            random_state=self.config["random_state"],
            scoring=scoring,
            refit='f1_weighted',
            verbose=self.config["verbose"]
        )

        random_search.fit(self.X_train, self.y_train)

        importance = random_search.best_estimator_.named_steps["xgboost"].feature_importances_
        category_names = self.X_train.columns

        pyplot.figure(figsize=(8, 10))
        pyplot.bar(category_names, importance)
        pyplot.xlabel('Features')
        pyplot.ylabel('Importance')
        pyplot.xticks(rotation=45, ha='right')
        pyplot.show()

        y_preds = random_search.predict(self.X_test)

        print("\nConfusion Matrix : ")
        print(confusion_matrix(self.y_test, y_preds))

        print("\nClassification Report : ")
        print(classification_report(self.y_test, y_preds))

        r.write_to_report("(XGBClassifier) confusion matrix", confusion_matrix(self.y_test, y_preds).tolist())
        r.write_to_report("(XGBClassifier) best model", str(random_search.best_estimator_))
        r.write_to_report("(XGBClassifier) best parameters", str(random_search.best_params_))
        r.write_to_report("(XGBClassifier) accuracy", random_search.best_estimator_.score(self.X_test, self.y_test))

        return data