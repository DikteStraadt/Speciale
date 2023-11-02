from matplotlib import pyplot
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from FeatureEngineering import FeatureSelection as f
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

        if self.config["feature_statistical"]:
            sfs_data = f.ForwardSubsetSelection(RandomForestClassifier(), self.target, self.config).transform(data)
            self.X_train = self.X_train.loc[:, sfs_data.columns]
            self.X_test = self.X_test.loc[:, sfs_data.columns]
        else:
            # Det sæt af features klinikerne kommer med
            r.write_to_report("feature selection", "Clinical")

        model = Pipeline(steps=[
            ("randomforest", RandomForestClassifier()),
        ])

        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'f1_micro': make_scorer(f1_score, average='micro'),
            'f1_macro': make_scorer(f1_score, average='macro'),
            'f1_weighted': make_scorer(f1_score, average='weighted'),
        }

        param = {
            'randomforest__n_estimators': [100, 200, 300, 400, 500, 600, 700],
            'randomforest__max_depth': [None, 10, 20, 30, 40, 50],
            'randomforest__min_samples_split': [2, 5, 10],
            'randomforest__min_samples_leaf': [1, 2, 4],
            'randomforest__max_features': ['sqrt', 'log2'],
            'randomforest__bootstrap': [True, False],
            'randomforest__class_weight': [None, 'balanced'],
            'randomforest__random_state': [123],
        }

        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param,
            n_iter=self.config["iterations"],
            cv=self.config["cv"],
            n_jobs=-1,
            random_state=self.config["random_state"],
            scoring=scoring,
            refit='accuracy',
            verbose=self.config["verbose"]
        )

        random_search.fit(self.X_train, self.y_train)

        importance = random_search.best_estimator_.named_steps["randomforest"].feature_importances_
        category_names = self.X_train.columns

        pyplot.figure(figsize=(8, 10))
        pyplot.bar(category_names, importance)
        pyplot.xlabel('Features')
        pyplot.ylabel('Importance')
        pyplot.xticks(rotation=45, ha='right')
        pyplot.show()

        r.write_to_report("(RandomForestClassifier) best model", str(random_search.best_estimator_))
        r.write_to_report("(RandomForestClassifier) best parameters", str(random_search.best_params_))
        r.write_to_report("(RandomForestClassifier) accuracy", random_search.best_estimator_.score(self.X_test, self.y_test))

        return data