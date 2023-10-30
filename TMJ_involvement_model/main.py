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
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

configs = [
    {'N_CATEGORIES': 3, 'TIMELINESS': False, 'FEATURES_STATISTICAL': False, 'ENCODING_EMBEDDING': False},
    {'N_CATEGORIES': 3, 'TIMELINESS': False, 'FEATURES_STATISTICAL': False, 'ENCODING_EMBEDDING': True},
    {'N_CATEGORIES': 3, 'TIMELINESS': False, 'FEATURES_STATISTICAL': True, 'ENCODING_EMBEDDING': False},
    {'N_CATEGORIES': 3, 'TIMELINESS': False, 'FEATURES_STATISTICAL': True, 'ENCODING_EMBEDDING': True}
]

if __name__ == '__main__':

    ##################### PREPROCESS AND SAVE DATA #####################

    # Import, preprocess and export data to file
    # data = p.preprocess_data(N_CATEGORIES)
    # print("Data is preprocessed")

    for config in configs:

        ##################### IMPORT DATA #####################

        # Import formatted visitation data
        data = d.import_data("C:/Users/User/Downloads/output.xlsx", "Sheet1")
        print("Data is imported")

        r.create_empty_report()
        r.write_to_report("timestamp", datetime.now().strftime('%d-%m-%Y %H-%M-%S'))
        r.write_to_report("N_categories", config['N_CATEGORIES'])
        r.write_to_report("timeliness", config['TIMELINESS'])
        r.write_to_report("original data size", f"{data.shape}")

        ##################### PROCESS DATA #####################

        if config['ENCODING_EMBEDDING']:
            encoding_method = e.EntityEmbeddingEncoding()
        else:
            encoding_method = e.OneHotEncode()

        feature_engineering_pipeline = Pipeline(steps=[
            ("Sampling", s.SMOTE()),
            ("Encoding", encoding_method),
            ("Normalization", n.NormalizeData()),
        ])

        data = feature_engineering_pipeline.fit_transform(data)

        ##################### SPLIT DATA #####################

        columns_to_exclude = ['sex', 'type', 'studyid', 'involvementstatus', 'Unnamed: 0', 'visitationdate']
        target = data['involvementstatus']
        data = data.drop(columns=columns_to_exclude)

        X_train, X_rem, y_train, y_rem = train_test_split(data, target, train_size=0.8, random_state=123, shuffle=True)
        X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=123, shuffle=True)

        r.write_to_report("train size", f"{X_train.shape} {y_train.shape}")
        r.write_to_report("test size", f"{X_test.shape} {y_test.shape}")
        r.write_to_report("validation size", f"{X_valid.shape} {y_valid.shape}")

        ##################### PERFORM FEATURE SELECTION AND TRAIN MODEL #####################

        pipeline = Pipeline(steps=[
            ("randomforest", rf.RandomForest(X_train, X_test, y_train, y_test, target, config)),
            ("xgboost", xg.XGBoostClassifier(X_train, X_test, y_train, y_test, target, config)),
        ])

        pipeline.transform(data)

        r.rename_report_file()

    print("Done!")