import pandas as pd
import numpy as np
import shap
from Utils import LoadFile as lf
from sklearn.model_selection import train_test_split
from Utils import ImportExportData as d
from Utils import Configuration as c
import imblearn
import catboost
from ShapMultiClassifier import multiclass_forceplot, waterfallplot, aggregatedplot
from ShapBinaryClassifier import forceplot_binary
from Utils import FeatureMerging as fm
from Utils import OneHotEncoding as ohe


"""
Looping through feature names to get encoded features
"""
def getEncodedFeatureNames(featureList):
    encodingList = []
    for feature in featureList:
        if '_' in feature:
            feature = feature[ 0 : feature.index("_")]
            if feature not in encodingList:
                encodingList.append(feature)

    return encodingList

def getOneHotDataFrame(data, encoding_columns):
    X_enc = pd.get_dummies(data[encoding_columns].astype(str), prefix=encoding_columns)
    X = pd.concat([data, X_enc], ignore_index=True)
    #X.columns = X.columns.removesuffix(".0")
    X.columns = X.columns.map(lambda x: x.removesuffix('.0'))

    return X

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


def prepareTrainTestSplit(feature_names, columns_to_encode, catBoostFlag):
    #configurations = c.get_configurations()
    #columns_to_exclude = ['sex', 'type', 'studyid', 'Unnamed: 0', 'visitationdate']
    columns_to_exclude = ['Unnamed: 0']

    data = d.import_data("TestData/processed_data.xlsx", "Sheet1")
    data = data.drop(columns=columns_to_exclude)
    #data = fm.mergeFeatures(data)

    y = data['involvementstatus']
    X = data.drop('involvementstatus', axis=1)

    # if configurations[0]['encoding'] == 'one hot':
    #    X = ohe.doOneHotEncoding(columns_to_encode, X)
    #
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

    clf_explainer, clf_est, feature_names = getModel("TestData/testmodel.pkl", "catboost")
    X_train, X_test, y_train, y_test = prepareTrainTestSplit(feature_names, columns_to_encode, True)


    catb_shap_values = clf_explainer.shap_values(X_train)
    print(len(catb_shap_values))

    #waterfallplot(catb_explainer, X_train, "TMJ involvement", 7)
    #multiclass_forceplot(catb_est, 'catboost', 7, X_train, y_train, catb_explainer, catb_shap_values,
    #                     classes='pred')
    aggregatedplot(X_train, clf_explainer, 'beeswarm', 'TMJ Involvement')
