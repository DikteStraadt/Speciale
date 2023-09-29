from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def XGBoostBinaryClassification():
    breast_cancer = load_breast_cancer()

    for line in breast_cancer.DESCR.split("\n")[5:31]:
        print(line)

    breast_cancer_df = pd.DataFrame(data=breast_cancer.data, columns=breast_cancer.feature_names)
    breast_cancer_df["TumorType"] = breast_cancer.target
    breast_cancer_df.head()
    breast_cancer_featurenames = []
    for x in range(len(breast_cancer.feature_names)):
        breast_cancer_featurenames.append(breast_cancer.feature_names[x])

    X_train, X_test, Y_train, Y_test = train_test_split(breast_cancer.data, breast_cancer.target, train_size=0.90,
                                                        stratify=breast_cancer.target, random_state=42)

    print("Train/Test Sizes : ", X_train.shape, X_test.shape, Y_train.shape, Y_test.shape, "\n")

    dmat_train = xgb.DMatrix(X_train, Y_train, feature_names=breast_cancer_featurenames)
    dmat_test = xgb.DMatrix(X_test, Y_test, feature_names=breast_cancer_featurenames)

    booster = xgb.train({'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic'},
                        dmat_train,
                        evals=[(dmat_train, "train"), (dmat_test, "test")])

    print("\nTrain RMSE : ", booster.eval(dmat_train))
    print("Test  RMSE : ", booster.eval(dmat_test))

    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    train_preds = [1 if pred > 0.5 else 0 for pred in booster.predict(data=dmat_train)]
    test_preds = [1 if pred > 0.5 else 0 for pred in booster.predict(data=dmat_test)]

    print("\nTest  Accuracy : %.2f" % accuracy_score(Y_test, test_preds))
    print("Train Accuracy : %.2f" % accuracy_score(Y_train, train_preds))

    print("\nConfusion Matrix : ")
    print(confusion_matrix(Y_test, test_preds))

    print("\nClassification Report : ")
    print(classification_report(Y_test, test_preds))

