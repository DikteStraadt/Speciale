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

def getOneHotDataFrame():
    X_enc = pd.get_dummies(data[encoding_columns].astype(str), prefix=encoding_columns)
    # X = pd.concat([X, X_enc], ignore_index=True)
    # X.columns = X.columns.str.removesuffix(".0")


if __name__ == '__main__':
    configurations = c.get_configurations()
    columns_to_exclude = ['sex', 'type', 'studyid', 'Unnamed: 0', 'visitationdate']
    columns_to_encode = ['drug', 'asypupilline', 'asybasis', 'asyoccl', 'profile',
                         'lowerface']
    data = d.import_data(f"output_{configurations[0]['n_categories']}_cat.xlsx", "Sheet1")
    data = data.drop(columns=columns_to_exclude)
    data = fm.mergeFeatures(data)
    #featurename_cols = ['painmoveright', 'openingfunction']


    #X_train_df = pd.DataFrame(X_train, columns=featurename_cols)

    # Calculate SHAP values for multi class model
    #catb_model = lf.load_model("0HH3_model.pkl")
    catb_model = lf.load_model("P1RH.pkl")
    catb_est = catb_model.best_estimator_
    catb_model = catb_est.named_steps['catboost']
    feature_names = catb_model.feature_names_
    catb_explainer = shap.TreeExplainer(catb_model)

    y = data['involvementstatus']
    X = data.drop('involvementstatus', axis=1)
    X = X.astype(int)

    encoding_columns = getEncodedFeatureNames(feature_names)


    X = X[feature_names]
    X[columns_to_encode] = X[columns_to_encode].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    catb_shap_values = catb_explainer.shap_values(X_train)
    print(len(catb_shap_values))

    #waterfallplot(catb_explainer, X_train, "TMJ involvement", 7)
    #multiclass_forceplot(catb_est, 'catboost', 7, X_train, y_train, catb_explainer, catb_shap_values,
    #                     classes='pred')
    aggregatedplot(X_train, catb_explainer, 'bar', 'TMJ Involvement')
