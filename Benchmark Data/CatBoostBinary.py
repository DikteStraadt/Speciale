import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from catboost import CatBoost
from catboost.utils import eval_metric

def catboostBinaryClassification():
    breast_cancer = load_breast_cancer()

    for line in breast_cancer.DESCR.split("\n")[5:31]:
        print(line)

    breast_cancer_df = pd.DataFrame(data=breast_cancer.data, columns=breast_cancer.feature_names)
    breast_cancer_df["TumorType"] = breast_cancer.target
    breast_cancer_df.head()

    X_train, X_test, Y_train, Y_test = train_test_split(breast_cancer.data, breast_cancer.target, train_size=0.9,
                                                        stratify=breast_cancer.target, random_state=123)

    booster = CatBoost(params={'iterations': 100, 'verbose': 10, 'loss_function': 'Logloss'})

    booster.fit(X_train, Y_train, eval_set=(X_test, Y_test))
    booster.set_feature_names(breast_cancer.feature_names)

    test_preds = booster.predict(X_test, prediction_type="Class")
    train_preds = booster.predict(X_train, prediction_type="Class")

    print("\nTest Accuracy: %.2f" % eval_metric(Y_test, test_preds, "Accuracy")[0])
    print("Train Accuracy : %2f" % eval_metric(Y_train, train_preds, "Accuracy")[0])