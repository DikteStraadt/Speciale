from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import json

class XGBoost:

    def __init__(self, max_depth, eta, num_class):
        self.max_depth = max_depth
        self.eta = eta
        self.num_class = num_class

    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):

        columns_to_exclude = ['sex', 'type', 'studyid', 'involvementstatus', 'Unnamed: 0', 'visitationdate']
        target = data['involvementstatus']
        data = data.drop(columns=columns_to_exclude)

        X_train, X_test, Y_train, Y_test = train_test_split(data, target, train_size=0.9, stratify=target, random_state=123)

        dmat_train = xgb.DMatrix(X_train, Y_train)
        dmat_test = xgb.DMatrix(X_test, Y_test)

        model_params = {
            'max_depth': self.max_depth,
            'eta': self.eta,
            'objective': 'multi:softmax',
            'num_class': self.num_class
        }

        booster = xgb.train(model_params, dmat_train, evals=[(dmat_train, "train"), (dmat_test, "test")])
        parameters = json.loads(booster.save_config())

        print("\nTrain RMSE : ", booster.eval(dmat_train))
        print("Test  RMSE : ", booster.eval(dmat_test))

        print("Train Accuracy : %.2f" % accuracy_score(Y_train, booster.predict(data=dmat_train)))
        print("\nTest  Accuracy : %.2f" % accuracy_score(Y_test, booster.predict(data=dmat_test)))

        print("\nConfusion Matrix : ")
        print(confusion_matrix(Y_test, booster.predict(data=dmat_test)))

        print("\nClassification Report : ")
        print(classification_report(Y_test, booster.predict(data=dmat_test)))

        print("XGBoost model fitted")