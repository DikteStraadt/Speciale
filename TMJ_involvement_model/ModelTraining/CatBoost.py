from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from sklearn.datasets import load_iris

class CatBoost:

    def __init__(self, iterations, depth, learning_rate):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate

    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):

        columns_to_exclude = ['sex', 'type', 'studyid', 'involvementstatus', 'Unnamed: 0', 'visitationdate']
        target = data['involvementstatus']
        data = data.drop(columns=columns_to_exclude)

        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.1, random_state=123)

        model_params = {
            'iterations': self.iterations,
            'depth': self.depth,
            'learning_rate': self.learning_rate,
            'loss_function': 'MultiClass',
            'verbose': False
        }

        model = CatBoostClassifier(**model_params)
        parameters = model.get_params()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print('Training Coefficient of R^2 : %.3f' % model.score(X_train, y_train))
        print('Test Coefficient of R^2 : %.3f' % model.score(X_test, y_test))

        print("\nConfusion Matrix : ")
        print(confusion_matrix(y_test, y_pred))

        print("\nClassification Report : ")
        print(classification_report(y_test, y_pred))

        print("CatBoost model fitted")