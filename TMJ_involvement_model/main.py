import warnings
import Report as r

from datetime import datetime
from sklearn.model_selection import train_test_split
from DataCleaning.RawData import ImportExportData as d
from FeatureEngineering import Normalization as n
from FeatureEngineering import Sampling as s
from FeatureEngineering import Encoding as e
from ModelTraining import RandomForest as rf
from ModelTraining import XGBoost as xg
from ModelTraining import CatBoost as cat
from sklearn.pipeline import Pipeline
from itertools import product

warnings.filterwarnings('ignore')

N_CATEGORIES = [3]  # [2, 3, 5, 8]
TIMELINESS = [False]  # True, False
FEATURES_STATISTICAL = [True]  # True, False
ENCODING_EMBEDDING = [True]  # True, False
configs = list(product(N_CATEGORIES, TIMELINESS, FEATURES_STATISTICAL, ENCODING_EMBEDDING))

if __name__ == '__main__':

    ##################### PREPROCESS AND SAVE DATA #####################

    # Import, preprocess and export data to file
    # data = p.preprocess_data(N_CATEGORIES)
    # print("Data is preprocessed")

    ##################### IMPORT DATA #####################

    # Import formatted visitation data
    imported_data = d.import_data("output.xlsx", "Sheet1")
    print("Data is imported")

    for c in configs:

        config = {
            'N_CATEGORIES': c[0],
            'TIMELINESS': c[1],
            'FEATURES_STATISTICAL': c[2],
            'ENCODING_EMBEDDING': c[3]
        }

        columns_to_exclude = ['sex', 'type', 'studyid', 'tractionright', 'tractionleft', 'Unnamed: 0', 'visitationdate']
        data = imported_data.drop(columns=columns_to_exclude)
        target = data['involvementstatus']

        r.create_empty_report()
        r.write_to_report("timestamp", datetime.now().strftime('%d-%m-%Y %H-%M-%S'))
        r.write_to_report("N_categories", config['N_CATEGORIES'])
        r.write_to_report("timeliness", config['TIMELINESS'])
        r.write_to_report("original data size", f"{data.shape}")

        ##################### PROCESS DATA #####################

        #columns_to_encode = ['drug', 'asypupilline', 'asybasis', 'asymenton', 'asyoccl', 'asyupmid', 'asylowmi',
        #                     'profile', 'lowerface', 'spacerelationship', 'sagittalrelationright',
        #                     'sagitalrelationleft', 'transversal']
        columns_to_encode = ['asypupilline', 'headache']


        if config['ENCODING_EMBEDDING']:
            encoding_method = e.EntityEmbeddingTransformer('involvementstatus', columns_to_encode)
        else:
            encoding_method = e.OneHotEncode(columns_to_encode)

        feature_engineering_pipeline = Pipeline(steps=[
            # ("Sampling", s.SMOTE()),
            ("Encoding", encoding_method),
            ("Normalization", n.NormalizeData()),
        ])

        data = feature_engineering_pipeline.fit_transform(data)

        ##################### SPLIT DATA #####################
        data = data.drop('involvementstatus', axis=1)

        X_train, X_rem, y_train, y_rem = train_test_split(data, target, train_size=0.8, random_state=123, shuffle=True)
        X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=123, shuffle=True)

        r.write_to_report("train size", f"{X_train.shape} {y_train.shape}")
        r.write_to_report("test size", f"{X_test.shape} {y_test.shape}")
        r.write_to_report("validation size", f"{X_valid.shape} {y_valid.shape}")

        ##################### PERFORM FEATURE SELECTION AND TRAIN MODEL #####################

        pipeline = Pipeline(steps=[
            ("randomforest", rf.RandomForest(X_train, X_test, y_train, y_test, target, config)),
            ("xgboost", xg.XGBoostClassifier(X_train, X_test, y_train, y_test, target, config)),
            ("catboost", cat.CatBoost(X_train, X_test, y_train, y_test, target, config))
        ])

        pipeline.transform(data)

        r.rename_report_file()

    print("Done!")