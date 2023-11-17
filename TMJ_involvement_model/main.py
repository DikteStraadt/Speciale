import pickle
import string
import warnings
import random
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import make_scorer, f1_score

from Utils import Configuration as c, Report as r, SaveLoadModel as sl

from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from DataCleaning.RawData import ImportExportData as d
from FeatureEngineering import Normalization as n
from FeatureEngineering import Encoding as e
from FeatureEngineering import Sampling as s
from ModelTraining import RandomForest as rf
from ModelTraining import XGBoost as xg
from ModelTraining import CatBoost as cat
from sklearn.pipeline import Pipeline
from FeatureEngineering import TypeConverter as tc
from DataCleaning import PreprocessData as p
from FeatureEngineering import TransformFeatures as fm
from ModelEvaluation import Evaluation as ev
from ModelEvaluation import ConformancePrediction as cp
from Utils.SaveLoadModel import save_model
from ModelEvaluation import CatBoostWrapper as cbw

warnings.filterwarnings('ignore')

if __name__ == '__main__':

    sl.remove_models()

    ##################### IMPORT CONFIGS #####################

    configurations = c.get_configurations()

    imported_data_3_cat = d.import_data("Data/processed_data.xlsx", "Sheet1")
    print("Data with three categories is imported")

    columns_to_exclude = ['Unnamed: 0']

    data = imported_data_3_cat.drop(columns=columns_to_exclude)

    ##################### SPLIT DATA #####################
    columns_to_encode = ['drug', 'asypupilline', 'asybasis', 'asyoccl', 'asymenton', 'profile', 'asyupmid', 'asylowmi',
                         'lowerface', 'sagittalrelation']
    encoding_method = e.OneHotEncode(columns_to_encode)

    target = data['involvementstatus']
    data = data.drop('involvementstatus', axis=1)

    X_train, X_rem, y_train, y_rem = train_test_split(data, target, train_size=0.7, random_state=42, shuffle=True)
    X_valid, X_rem_2, y_valid, y_rem_2 = train_test_split(X_rem, y_rem, train_size=(1/3), random_state=42, shuffle=True)
    X_uncertainty, X_test, y_uncertainty, y_test = train_test_split(X_rem_2, y_rem_2, train_size=0.5, random_state=42, shuffle=True)


    ##################### PERFORM CONFORMANCE PREDICTION #####################

    ## cp.conformancePrediction(wrapper_model.model, X_uncertainty, y_uncertainty, X_test, y_test)


    test_model = sl.load_model("Tester/best_model.pkl")
    test_est = test_model.best_estimator_
    test_model = test_est.named_steps['catboost'] # here needs to be name of best_model
    featurenames = test_model.feature_names_
    # wrapper model only needed for CatBoost
    wrapper_model = cbw.CatBoostWrapper(test_model, feature_names_=test_model.feature_names_, classes_=test_model.classes_)
    cp.conformancePrediction(wrapper_model, featurenames, X_uncertainty, y_uncertainty, X_test, y_test)

    print("Done!")
