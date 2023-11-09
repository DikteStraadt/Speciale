import warnings

from Utils import Configuration as c, Report as r

from datetime import datetime
from sklearn.model_selection import train_test_split
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
from FeatureEngineering import FeatureMerging as fm

warnings.filterwarnings('ignore')

if __name__ == '__main__':

    ##################### IMPORT CONFIGS #####################

    configurations = c.get_configurations()

    ##################### PREPROCESS AND SAVE DATA #####################

    # n_categories = configurations[0]['n_categories']
    # data = p.preprocess_data(n_categories)
    # print("Data is preprocessed and saved")

    ##################### IMPORT DATA #####################

    imported_data = d.import_data(f"output_{configurations[0]['n_categories']}_cat.xlsx", "Sheet1")
    print("Data is imported")

    for config in configurations:

        columns_to_exclude = ['sex', 'type', 'studyid', 'Unnamed: 0', 'visitationdate']
        data = imported_data.drop(columns=columns_to_exclude)

        r.create_empty_report()
        r.write_to_report("timestamp start", datetime.now().strftime('%d-%m-%Y %H-%M-%S'))
        r.write_to_report("timestamp end", "")  # placeholder
        r.write_to_report("n_categories", config['n_categories'])
        r.write_to_report("timeliness", config['timeliness'])
        r.write_to_report("original data size", f"{data.shape}")

        ##################### PROCESS DATA #####################

        columns_to_encode = ['drug', 'asypupilline', 'asybasis', 'asyoccl', 'asymenton', 'profile', 'asyupmid', 'asylowmi', 'lowerface', 'sagittalrelation']

        if config['encoding_embedding']:
            encoding_method = e.EntityEmbeddingTransformer('involvementstatus', columns_to_encode)
        else:
            encoding_method = e.OneHotEncode(columns_to_encode)

        feature_engineering_pipeline = Pipeline(steps=[
            ("Convert type", tc.ConvertToCategories()),
            ("Merging features", fm.MergeFeatures()),
            ("Sampling", s.SMOTE(config)),
            ("Encoding", encoding_method),
            ("Normalization", n.NormalizeData(config)),
        ])

        data = feature_engineering_pipeline.fit_transform(data)

        ##################### SPLIT DATA #####################

        target = data['involvementstatus']
        data = data.drop('involvementstatus', axis=1)

        X_train, X_rem, y_train, y_rem = train_test_split(data, target, train_size=0.8, random_state=42, shuffle=True)
        X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=42, shuffle=True)

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

        r.write_to_report("timestamp end", datetime.now().strftime('%d-%m-%Y %H-%M-%S'))

        r.rename_report_file()

    print("Done!")