import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split


class RandomForest:

    def __init__(self, n_estimators, criterion, random_state):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.random_state = random_state

    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):

        columns_to_exclude = ['sex', 'type', 'studyid', 'involvementstatus', 'Unnamed: 0', 'visitationdate']
        target = data['involvementstatus']
        data = data.drop(columns=columns_to_exclude)

        X_train, X_test, Y_train, Y_test = train_test_split(data, target, train_size=0.9, stratify=target, random_state=123)
        print("Train/Test Sizes : ", X_train.shape, X_test.shape, Y_train.shape, Y_test.shape, "\n")

        model_params = {
            'n_estimators': self.n_estimators,
            'criterion': self.criterion,
            'random_state': self.random_state
        }

        random_forest_classifier = RandomForestClassifier(**model_params)
        parameters = random_forest_classifier.get_params()
        random_forest_classifier.fit(X_train, Y_train)
        Y_preds = random_forest_classifier.predict(X_test)

        print('Training Coefficient of R^2 : %.3f' % random_forest_classifier.score(X_train, Y_train))
        print('Test Coefficient of R^2 : %.3f' % random_forest_classifier.score(X_test, Y_test))

        print("\nConfusion Matrix : ")
        print(confusion_matrix(Y_test, Y_preds))

        print("\nClassification Report : ")
        print(classification_report(Y_test, Y_preds))

        print("Random forest model fitted")

        return data