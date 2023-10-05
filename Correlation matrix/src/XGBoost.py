from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import xgboost as xgb

def xgboost(data, target):

    X_train, X_test, Y_train, Y_test = train_test_split(data, target, train_size=0.90, stratify=target, random_state=42)

    print("Train/Test Sizes : ", X_train.shape, X_test.shape, Y_train.shape, Y_test.shape, "\n")

    dmat_train = xgb.DMatrix(X_train, Y_train)
    dmat_test = xgb.DMatrix(X_test, Y_test)

    booster = xgb.train({'max_depth': 5, 'eta': 1, 'objective': 'multi:softmax', 'num_class': 8}, dmat_train, evals=[(dmat_train, "train"), (dmat_test, "test")])

    print("\nTrain RMSE : ", booster.eval(dmat_train))
    print("Test  RMSE : ", booster.eval(dmat_test))

    #train_preds = [1 if pred > 0.5 else 0 for pred in booster.predict(data=dmat_train)]
    #test_preds = [1 if pred > 0.5 else 0 for pred in booster.predict(data=dmat_test)]

    print("\nTest  Accuracy : %.2f" % accuracy_score(Y_test, booster.predict(data=dmat_test)))
    print("Train Accuracy : %.2f" % accuracy_score(Y_train, booster.predict(data=dmat_train)))

    print("\nConfusion Matrix : ")
    print(confusion_matrix(Y_test, booster.predict(data=dmat_test)))

    print("\nClassification Report : ")
    print(classification_report(Y_test, booster.predict(data=dmat_test)))

    print("Done!")
