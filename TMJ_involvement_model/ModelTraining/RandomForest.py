import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split


class RandomForest:

    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):

        columns_to_exclude = ['sex', 'type', 'studyid', 'involvementstatus', 'Unnamed: 0', 'visitationdate']
        target = data['involvementstatus']
        data = data.drop(columns=columns_to_exclude)

        X_train, X_test, Y_train, Y_test = train_test_split(data, target, train_size=0.9, stratify=target, random_state=123)

        random_forest_classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=42)
        random_forest_classifier.fit(X_train, Y_train)
        Y_preds = random_forest_classifier.predict(X_test)

        print('Training Coefficient of R^2 : %.3f' % random_forest_classifier.score(X_train, Y_train))
        print('Test Coefficient of R^2 : %.3f' % random_forest_classifier.score(X_test, Y_test))

        print("\nConfusion Matrix : ")
        print(confusion_matrix(Y_test, Y_preds))

        print("\nClassification Report : ")
        print(classification_report(Y_test, Y_preds))

        print("Done!")