import string
import warnings
import random
import pandas as pd

from Utils import Configuration as c, Report as r, SaveLoadModel as sl

from datetime import datetime
from sklearn.model_selection import train_test_split
from DataCleaning.RawData import ImportExportData as d, TimeSliceData as f
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

warnings.filterwarnings('ignore')

if __name__ == '__main__':

    sl.remove_models()

    ##################### IMPORT CONFIGS #####################

    configurations = c.get_configurations()
    configurations = [combo for combo in configurations if not (combo["feature_selection"] == "SFS" and combo["encoding_embedding"] == False)]
    if len(configurations) == 0:
        exit()

    ##################### PREPROCESS AND SAVE DATA #####################

    if configurations[0]['preprocess_data_2_cat']:
        p.preprocess_data(2)  # Two categories
        print("Data with 2 categories is preprocessed and saved")

    if configurations[0]['preprocess_data_3_cat']:
        p.preprocess_data(3)  # Three categories
        print("Data with 3 categories is preprocessed and saved")

    ##################### IMPORT DATA #####################

    print("Starting data import")
    if any(obj['n_categories'] == 2 for obj in configurations):
        imported_data_2_cat = d.import_data("Data/cleaned_data_2_cat.xlsx", "Sheet1")
        print("Data with two categories is imported")

    if any(obj['n_categories'] == 3 for obj in configurations):
        imported_data_3_cat = d.import_data("Data/cleaned_data_3_cat.xlsx", "Sheet1")
        print("Data with three categories is imported")

    for config in configurations:

        columns_to_exclude = ['sex', 'type', 'studyid', 'Unnamed: 0', 'visitationdate']

        if config['n_categories'] == 2:
            data = imported_data_2_cat.drop(columns=columns_to_exclude)
            data = f.filter_visitations(data, config['time_slice'])
        elif config['n_categories'] == 3:
            data = imported_data_3_cat.drop(columns=columns_to_exclude)
            data = f.filter_visitations(data, config['time_slice'])

        r.create_empty_report()

        r.write_to_report("id", ''.join(random.choices(string.ascii_uppercase + string.digits, k=4)))
        r.write_to_report("timestamp start", datetime.now().strftime('%d-%m-%Y %H-%M-%S'))
        r.write_to_report("timestamp end", "")  # placeholder
        r.write_to_report("n_categories", config['n_categories'])
        r.write_to_report("timeliness", config['timeliness'])
        r.write_to_report("time_slice", config['time_slice'])
        r.write_to_report("original data size", f"{data.shape}")

        ##################### PROCESS DATA #####################

        columns_to_encode = ['drug', 'asypupilline', 'asybasis', 'asyoccl', 'asymenton', 'profile', 'asyupmid', 'asylowmi', 'lowerface', 'sagittalrelation']

        if config['encoding_embedding']:
            encoding_method = e.EntityEmbeddingTransformer('involvementstatus', columns_to_encode, config)
        else:
            encoding_method = e.OneHotEncode(columns_to_encode)

        feature_engineering_pipeline = Pipeline(steps=[
            ("Convert type", tc.ConvertToCategories()),
            ("Merging features", fm.MergeFeatures()),
            ("Encoding", encoding_method),
            ("Normalization", n.NormalizeData(config)),
        ])

        data = feature_engineering_pipeline.transform(data)

        ##################### SPLIT DATA #####################

        target = data['involvementstatus']
        previous_status = data[['previousstatus', 'previousinvolvementstatusvisitation0', 'previousinvolvementstatusvisitation1',
                          'previousinvolvementstatusvisitation2', 'previousinvolvementstatusvisitation3',
                          'previousinvolvementstatusvisitation4', 'previousinvolvementstatusvisitation5',
                          'previousinvolvementstatusvisitation6', 'previousinvolvementstatusvisitation7',
                          'previousinvolvementstatusvisitation8', 'previousinvolvementstatusvisitation9',
                          'previousinvolvementstatusvisitation10', 'previousinvolvementstatusvisitation11',
                          'previousinvolvementstatusvisitation12', 'previousinvolvementstatusvisitation13',
                          'previousinvolvementstatusvisitation14', 'previousinvolvementstatusvisitation15']]

        data = data.drop('involvementstatus', axis=1)
        data = data.drop(previous_status.columns, axis=1)

        X_train, X_rem, y_train, y_rem = train_test_split(data, target, train_size=0.8, random_state=42, shuffle=True)
        X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, train_size=0.5, random_state=42, shuffle=True)

        r.write_to_report("train size", f"{X_train.shape} {y_train.shape}")
        r.write_to_report("test size", f"{X_test.shape} {y_test.shape}")
        r.write_to_report("validation size", f"{X_valid.shape} {y_valid.shape}")

        ##################### UPSAMPLE DATA #####################

        smote_pipeline = Pipeline(steps=[
            ("Sampling", s.SMOTE(config))
        ])

        data_train = pd.concat([y_train, X_train], axis=1)
        data_train = data_train.drop('index', axis=1)
        data_train = smote_pipeline.transform(data_train)
        d.export_data(data_train, f"Data/processed_data.xlsx")

        y_train = data_train['involvementstatus']
        X_train = data_train.drop(columns=['involvementstatus'], axis=1)

        ##################### PERFORM FEATURE SELECTION AND TRAIN MODEL #####################

        pipeline = Pipeline(steps=[
            ("randomforest", rf.RandomForest(X_train, X_test, y_train, y_test, config)),
            ("xgboost", xg.XGBoostClassifier(X_train, X_test, y_train, y_test, config)),
            ("catboost", cat.CatBoost(X_train, X_test, y_train, y_test, config))
        ])

        pipeline.transform(data)

        r.write_to_report("timestamp end", datetime.now().strftime('%d-%m-%Y %H-%M-%S'))

        report = r.read_report()
        best_model = ev.find_best_model()

        ##################### PERFORM CONFORMANCE PREDICTION #####################
        test_model = sl.load_model("Tester/best_model.pkl")
        test_est = test_model.best_estimator_
        test_model = test_est.named_steps['catboost'] # here needs to be name of best_model
        cp.conformancePrediction(test_model, X_valid, y_valid, X_test, y_test)

        sl.rename_model(best_model, report)
        sl.remove_models()
        r.rename_report_file()

    print("Done!")