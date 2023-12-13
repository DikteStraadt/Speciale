from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from Utils import Report as r
from ModelEvaluation import Evaluation as e

class myDummyClassifier:

    def __init__(self, X_train, X_test, y_train, y_test, config):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.config = config

    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):

        dummy_classifier = DummyClassifier(strategy='most_frequent')
        dummy_classifier.fit(self.X_train, self.y_train)
        y_pred = dummy_classifier.predict(self.X_test)

        print(confusion_matrix(self.y_test, y_pred))
        print(classification_report(self.y_test, y_pred))
        print(f1_score(self.y_test, y_pred, average='micro'))

        r.write_to_report(f"(dummy) confusion matrix", confusion_matrix(self.y_test, y_pred).tolist())
        r.write_to_report(f"(dummy) classification report", classification_report(self.y_test, y_pred))
        r.write_to_report(f"(dummy) accuracy", f1_score(self.y_test, y_pred, average='micro'))

        return data
