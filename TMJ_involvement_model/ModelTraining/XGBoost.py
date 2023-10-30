from matplotlib import pyplot
from sklearn.ensemble import RandomForestClassifier
from FeatureEngineering import FeatureSelection as f
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, make_scorer, f1_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import xgboost as xgb
import Report as r


class XGBoostClassifier:

    def __init__(self, X_train, X_test, y_train, y_test, target):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.target = target

    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):
        sfs_data = f.ForwardSubsetSelection(xgb.XGBClassifier(), self.target).transform(data)
        self.X_train = self.X_train.loc[:, sfs_data.columns]
        self.X_test = self.X_test.loc[:, sfs_data.columns]

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
            'xgboost__max_depth': [3, 7, 10],
            'xgboost__eta': [0.01, 0.1, 0.2],
            'xgboost__objective': ['multi:softmax'],
            'xgboost__min_child_weight': [10, 15, 20, 25],
            'xgboost__colsample_bytree': [0.8, 0.9, 1],
            'xgboost__n_estimators': [300, 400, 500, 600],
            'xgboost__reg_alpha': [0.5, 0.2, 1],
            'xgboost__reg_lambda': [2, 3, 5],
            'xgboost__gamma': [1, 2, 3],
            'xgboost__random_state': [123]
        }

        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param,
            n_iter=2,
            cv=2,
            n_jobs=-1,
            random_state=123,
            scoring=scoring,
            refit='f1_weighted',
            verbose=3
        )

        random_search.fit(self.X_train, self.y_train)

        importance = random_search.best_estimator_.named_steps["xgboost"].feature_importances_
        category_names = self.X_train.columns  # Replace with your actual feature names

        pyplot.figure(figsize=(10, 6))
        pyplot.bar(category_names, importance)
        pyplot.xlabel('Features')
        pyplot.ylabel('Importance')
        pyplot.xticks(rotation=45, ha='right')
        pyplot.show()

        r.write_to_report("(XGBClassifier) best model", str(random_search.best_estimator_))
        r.write_to_report("(XGBClassifier) best parameters", str(random_search.best_params_))
        r.write_to_report("(XGBClassifier) accuracy", random_search.best_estimator_.score(self.X_test, self.y_test))

        return data