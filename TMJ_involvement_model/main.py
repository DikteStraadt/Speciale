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
from FeatureEngineering import RightLeftTransformation as fm
from ModelEvaluation import Evaluation as ev
from ModelEvaluation import ConformalPrediction as cp
from FeatureEngineering import DrugTransformation as dt
from FeatureEngineering import mmTransformation as mm
from ModelEvaluation import CatBoostWrapper as cbw
from CorrelationMatrix import CorrelationMatrix as cor

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
        p.preprocess_data(2, configurations[0]['previous_two_involvement_status'])  # Two categories
        print("Data with 2 categories is preprocessed and saved")

    if configurations[0]['preprocess_data_3_cat']:
        p.preprocess_data(3, configurations[0]['previous_two_involvement_status'])  # Three categories
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

        r.create_empty_report()
        columns_to_exclude = ['type', 'studyid', 'Unnamed: 0', 'visitationdate']

        if config['n_categories'] == 2:
            data = imported_data_2_cat.drop(columns=columns_to_exclude)
            data = f.filter_visitations(data, config['time_slice_2_cat'])
            r.write_to_report("time slice", f"{config['time_slice_2_cat'][0]} from {config['time_slice_2_cat'][1]} to {config['time_slice_2_cat'][2]}")
        elif config['n_categories'] == 3:
            data = imported_data_3_cat.drop(columns=columns_to_exclude)
            data = f.filter_visitations(data, config['time_slice_3_cat'])
            r.write_to_report("time slice", f"{config['time_slice_3_cat'][0]} from {config['time_slice_3_cat'][1]} to {config['time_slice_3_cat'][2]}")

        id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))

        r.write_to_report("id", id)
        r.write_to_report("timestamp start", datetime.now().strftime('%d-%m-%Y %H-%M-%S'))
        r.write_to_report("timestamp end", "")  # placeholder
        r.write_to_report("n_categories", config['n_categories'])
        r.write_to_report("original data size", f"{data.shape}")

        if config['previous_two_involvement_status']:
            r.write_to_report("previous values", "y-2")
        else:
            r.write_to_report("previous values", "y-15")

        ##################### PROCESS DATA #####################

        columns_to_encode = ['asypupilline', 'asybasis', 'asyoccl', 'asymenton', 'profile', 'asyupmid', 'asylowmi', 'lowerface', 'sagittalrelation']

        if config['encoding_embedding']:
            encoding_method = e.EntityEmbeddingTransformer('involvementstatus', columns_to_encode, config)
        else:
            encoding_method = e.OneHotEncode(columns_to_encode)

        feature_engineering_pipeline = Pipeline(steps=[
            ("Opening transformer", mm.OpeningTransformer()),
            ("Protrusion transformer", mm.ProtrusionTransformer()),
            ("New drug categories", dt.DrugTransformer()),
            ("Convert type", tc.ConvertToCategories(config)),
            ("Merging features", fm.MergeFeatures()),
            ("Encoding", encoding_method),
            ("Normalization", n.NormalizeData(config)),
        ])

        data = feature_engineering_pipeline.transform(data)

        ##################### SPLIT DATA #####################

        target = data['involvementstatus']

        if config["previous_two_involvement_status"]:
            previous_status = data[['previousstatus', 'previousinvolvementstatusvisitation_y-1', 'previousinvolvementstatusvisitation_y-2']]
        else:
            previous_status = data[
                ['previousstatus', 'previousinvolvementstatusvisitation0', 'previousinvolvementstatusvisitation1',
                 'previousinvolvementstatusvisitation2', 'previousinvolvementstatusvisitation3',
                 'previousinvolvementstatusvisitation4', 'previousinvolvementstatusvisitation5',
                 'previousinvolvementstatusvisitation6', 'previousinvolvementstatusvisitation7',
                 'previousinvolvementstatusvisitation8', 'previousinvolvementstatusvisitation9',
                 'previousinvolvementstatusvisitation10', 'previousinvolvementstatusvisitation11',
                 'previousinvolvementstatusvisitation12', 'previousinvolvementstatusvisitation13',
                 'previousinvolvementstatusvisitation14', 'previousinvolvementstatusvisitation15']]

        data = data.drop('involvementstatus', axis=1)

        previous_status[['index', 'ID']] = data[['index', 'ID']]

            ##################### CORRELATION MATRIX FOR PREVIOUS STATUS' ####################
            # cor.make_previous_status_correlation_matrix(previous_status)

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
        d.export_data(data_train, f"Temp/{id} data.xlsx")

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
        best_model_name = sl.rename_model(best_model, report)

        ##################### PERFORM CONFORMANCE PREDICTION #####################

        if config["do_conformance_prediction"]:

            test_model = sl.load_model(best_model_name)
            test_est = test_model.best_estimator_
            test_model = test_est.named_steps[best_model]
            n_categories = 2 if config['n_categories'] == 2 else 3

            if best_model == "random forest":
                feature_names = test_model.feature_names_in_
                cp.conformalPrediction(test_model, feature_names, X_valid, y_valid, X_test, y_test, n_categories)
            elif best_model == "xgboost":
                feature_names = test_model.feature_names_in_
                cp.conformalPrediction(test_model, feature_names, X_valid, y_valid, X_test, y_test, n_categories)
            elif best_model == "catboost":
                feature_names = test_model.feature_names_
                wrapper_model = cbw.CatBoostWrapper(test_model, feature_names_=feature_names, classes_=test_model.classes_)
                cp.conformalPrediction(wrapper_model, feature_names, X_valid, y_valid, X_test, y_test, n_categories)

        ##################### CLEAN WORKSPACE #####################

        r.rename_report_file()
        sl.remove_models()

    print("Done!")