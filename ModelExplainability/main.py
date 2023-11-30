import pandas as pd
import numpy as np
import shap
from Utils import LoadFile as lf
from sklearn.model_selection import train_test_split
from Utils import ImportExportData as d
import imblearn
import catboost
from ShapMultiClassifier import multiclass_forceplot, waterfallplot, aggregatedplot
from ShapBinaryClassifier import forceplot_binary, waterfallplotbinary, aggregatedplotbinary


def getModel(path, model):
    clf_model = lf.load_model(path)
    clf_est = clf_model.best_estimator_
    clf_model = clf_est.named_steps[model]

    if model == "catboost":
        feature_names = clf_model.feature_names_

    else:
        feature_names = clf_model.feature_names_in_

    clf_explainer = shap.TreeExplainer(clf_model)

    return (clf_explainer, clf_est, feature_names)


def prepareTrainTestSplit(path, feature_names, columns_to_encode, catBoostFlag):
    data = d.import_data(path, "Sheet1")

    y = data['involvementstatus']
    X = data.drop('involvementstatus', axis=1)

    X = X[feature_names]
    if catBoostFlag == True:
        for feat in columns_to_encode:
            if feat in X.columns:
                X[feat] = X[feat].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return (X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    columns_to_encode = ['drug', 'asypupilline', 'asybasis', 'asyoccl', 'profile',
                         'lowerface']


    """ Code for generating SHAP plots for 2 categories"""
    clf_explainer2, clf_est2, feature_names2 = getModel("TestData/cbTest2.pkl", "catboost")
    X_train2, X_test2, y_train2, y_test2 = prepareTrainTestSplit("TestData/cbTestdata2.xlsx", feature_names2, columns_to_encode, True)
    catb_shap_values = clf_explainer2.shap_values(X_train2)
    print(len(catb_shap_values))

    forceplot_binary(clf_est2, 'catboost', 7, X_train2, y_train2, clf_explainer2, catb_shap_values)
    waterfallplotbinary(clf_explainer2, X_train2,  7)
    aggregatedplotbinary(clf_explainer2,X_train2 , 'beeswarm')

    """ Code for generating SHAP plots for 3 categories """
    # clf_explainer, clf_est, feature_names = getModel("TestData/cbTest3.pkl", "catboost")
    # X_train, X_test, y_train, y_test = prepareTrainTestSplit("TestData/cbTestdata3.xlsx", feature_names, columns_to_encode, True)
    #
    #
    # catb_shap_values = clf_explainer.shap_values(X_train)
    # print(len(catb_shap_values))
    #
    # waterfallplot(clf_explainer, X_train, "TMJ involvement", 7)
    # multiclass_forceplot(clf_est, 'catboost', 7, X_train, y_train, clf_explainer, catb_shap_values,
    #                     classes='pred')
    # aggregatedplot(X_train, clf_explainer, 'beeswarm', 'TMJ Involvement')
