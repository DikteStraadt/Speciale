import string
import warnings
import random
import pandas as pd
from sklearn.preprocessing import StandardScaler

from Utils import Configuration as c, Report as r, SaveLoadModel as sl

from datetime import datetime
from sklearn.model_selection import train_test_split
from DataCleaning.RawData import ImportExportData as d
from FeatureEngineering import Normalization as n
from FeatureEngineering import Encoding as e
from FeatureEngineering import Sampling as s
from ModelTraining import RandomForest as rf
from ModelTraining import XGBoost as xg
from ModelTraining import CatBoost as cat
from ModelTraining import DummyClassifier as dc
from sklearn.pipeline import Pipeline
from FeatureEngineering import TypeConverter as tc
from DataCleaning import PreprocessData as p
from FeatureEngineering import RightLeftTransformation as fm
from ModelEvaluation import Evaluation as ev
from ModelEvaluation import ConformalPrediction as cp
from FeatureEngineering import DrugTransformation as dt
from FeatureEngineering import mmTransformation as mm
from ModelEvaluation import CatBoostWrapper as cbw
from FeatureEngineering import InverseEmbeddingTransformation as ie

warnings.filterwarnings('ignore')

if __name__ == '__main__':

    sl.clean_temp_folder()

    ##################### IMPORT CONFIGS #####################

    configurations = c.get_configurations()

    ##################### PREPROCESS AND SAVE DATA #####################

    if configurations[0]['preprocess_data']:
        p.preprocess_data(configurations[0]['lag_features'], configurations[0]['n_lag_features'], configurations[0]['lag_feature_list'])
        print("Data is preprocessed and saved")

    ##################### IMPORT DATA #####################

    print("Starting data import")
    imported_data = d.import_data("Data/cleaned_data.xlsx", "Sheet1")
    print("Data with two categories is imported")


    for config in configurations:

        r.create_empty_report()
        columns_to_exclude = ['type', 'studyid', 'Unnamed: 0', 'visitationdate']
        data = imported_data.drop(columns=columns_to_exclude)
        id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))

        r.write_to_report("id", id)
        r.write_to_report("timestamp start", datetime.now().strftime('%d-%m-%Y %H-%M-%S'))
        r.write_to_report("timestamp end", "")  # placeholder
        r.write_to_report("original data size", f"{data.shape}")
        r.write_to_report("lag features", config['lag_features'])

        ##################### PROCESS DATA #####################

        scaler = StandardScaler()
        columns_to_encode = ['asypupilline', 'asybasis', 'asyoccl', 'asymenton', 'profile', 'asyupmid', 'asylowmi', 'lowerface', 'sagittalrelation']

        feature_engineering_pipeline = Pipeline(steps=[
            ("Opening transformer", mm.OpeningTransformer()),
            ("Protrusion transformer", mm.ProtrusionTransformer()),
            ("New drug categories", dt.DrugTransformer()),
            ("Convert type", tc.ConvertToCategories(config)),
            ("Merging features", fm.MergeFeatures()),
            ("Encoding", e.EntityEmbeddingTransformer('involvementstatus', columns_to_encode, config)),
            ("Normalization", n.NormalizeData(config, scaler, True)),
        ])

        data = feature_engineering_pipeline.transform(data)

        ##################### SPLIT DATA #####################

        target = data['involvementstatus']
        d.export_data(data, f"Temp/{id} transformed data.xlsx")
        data = data.drop('involvementstatus', axis=1)

        X_train, X_rem, y_train, y_rem = train_test_split(data, target, train_size=0.8, random_state=42, shuffle=True)
        X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, train_size=0.5, random_state=42, shuffle=True)

        r.write_to_report("train size", f"{X_train.shape} {y_train.shape}")
        r.write_to_report("test size", f"{X_test.shape} {y_test.shape}")
        r.write_to_report("validation size", f"{X_valid.shape} {y_valid.shape}")

        ##################### UPSAMPLE DATA #####################

        smote_pipeline = Pipeline(steps=[
            ("dummy classifier", dc.myDummyClassifier(X_train, X_test, y_train, y_test, config)),
            ("Sampling", s.SMOTE(config))
        ])

        data_train = pd.concat([y_train, X_train], axis=1)
        #data_train = data_train.drop('index', axis=1)
        data_train = smote_pipeline.transform(data_train)

        y_train = data_train['involvementstatus']
        X_train = data_train.drop(columns=['involvementstatus'], axis=1)

        ##################### PERFORM FEATURE SELECTION AND TRAIN MODEL #####################

        ml_pipeline = Pipeline(steps=[
            ("catboost", cat.CatBoost(X_train, X_test, y_train, y_test, config)),
            ("randomforest", rf.RandomForest(X_train, X_test, y_train, y_test, config)),
            ("xgboost", xg.XGBoostClassifier(X_train, X_test, y_train, y_test, config))
        ])

        ml_pipeline.transform(data)

        ##################### INVERSE TRANSFORM FEATURES #####################

        inverse_transform_pipeline = Pipeline(steps=[
            ("inverse encoding", ie.ReverseEmbeddingTransformer(columns_to_encode)),
            ("inverse normalization", n.NormalizeData(config, scaler, False)),
        ])

        data = inverse_transform_pipeline.transform(data)
        data_for_export = pd.concat([data, target], axis=1)
        d.export_data(data_for_export, f"Temp/{id} inverse transformed data.xlsx")

        ##################### UTILS AND FIND BEST MODEL #####################

        r.write_to_report("timestamp end", datetime.now().strftime('%d-%m-%Y %H-%M-%S'))
        report = r.read_report()
        best_model = ev.find_best_model()
        best_model_name = sl.rename_model(best_model, report)

        ##################### PERFORM CONFORMANCE PREDICTION #####################

        test_model = sl.load_model(best_model_name)
        test_est = test_model.best_estimator_
        test_model = test_est.named_steps[best_model]

        if best_model == "random forest":
            feature_names = test_model.feature_names_in_
            cp.conformalPrediction(test_model, feature_names, X_valid, y_valid, X_test, y_test)
        elif best_model == "xgboost":
            feature_names = test_model.feature_names_in_
            cp.conformalPrediction(test_model, feature_names, X_valid, y_valid, X_test, y_test)
        elif best_model == "catboost":
            feature_names = test_model.feature_names_
            wrapper_model = cbw.CatBoostWrapper(test_model, feature_names_=feature_names, classes_=test_model.classes_)
            cp.conformalPrediction(wrapper_model, feature_names, X_valid, y_valid, X_test, y_test)

        ##################### CLEAN WORKSPACE #####################

        r.rename_report_file()
        sl.clean_temp_folder()

    print("Done!")