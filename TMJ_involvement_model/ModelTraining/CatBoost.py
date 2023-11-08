import pandas as pd
from imblearn.pipeline import Pipeline
from matplotlib import pyplot
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostClassifier
from FeatureEngineering import FeatureSelection as f
from Utils import Report as r

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

        if self.config["feature_statistical"]:
            sfs_data = f.ForwardSubsetSelection(CatBoostClassifier(), self.target, self.config).transform(data)
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
            r.write_to_report(f"(Clinical) n_features", len(clinical_columns))

        model = Pipeline(steps=[
            ("catboost", CatBoostClassifier()),
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
            'catboost__iterations': [100, 200, 300, 500, 700, 1000, 100000],
            'catboost__learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3],
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
            random_state=self.config["random_state"],
            scoring=scoring,
            refit='f1_weighted',
            verbose=self.config["verbose"],
        )

        random_search.fit(self.X_train, self.y_train)

        importance = random_search.best_estimator_.named_steps["catboost"].feature_importances_
        category_names = self.X_train.columns

        pyplot.figure(figsize=(8, 10))
        pyplot.bar(category_names, importance)
        pyplot.xlabel('Features')
        pyplot.ylabel('Importance')
        pyplot.xticks(rotation=45, ha='right')
        pyplot.show()

        r.write_to_report("(CatBoostClassifier) best model", str(random_search.best_estimator_))
        r.write_to_report("(CatBoostClassifier) best parameters", str(random_search.best_params_))
        r.write_to_report("(CatBoostClassifier) accuracy", random_search.best_estimator_.score(self.X_test, self.y_test))

        return data