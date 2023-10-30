import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, precision_score, recall_score, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV

import Report as r
from DataCleaning import PreprocessData as p
from DataCleaning.RawData import ImportExportData as d
from FeatureEngineering import Normalization as n
from FeatureEngineering import Sampling as s
from FeatureEngineering import Encoding as e
from FeatureEngineering import FeatureSelection as f
from ModelTraining import RandomForest as rf
from sklearn.pipeline import Pipeline
from FeatureEngineering import EntityEmbedding
from collections import Counter

warnings.filterwarnings('ignore')
N_CATEGORIES = 3  # 3, 4, 5, 8
TIMELINESS = "false"  # true, false

if __name__ == '__main__':
    # Create empty report file
    r.create_empty_report()
    r.write_to_report("timestamp", datetime.now().strftime('%d-%m-%Y %H-%M-%S'))
    r.write_to_report("N_categories", N_CATEGORIES)
    r.write_to_report("timeliness", TIMELINESS)

    ##################### IMPORT DATA #####################

    # Import, preprocess and export data to file
    # data = p.preprocess_data(N_CATEGORIES)
    # print("Data is preprocessed")

    # OR

    # Import formatted visitation data
    data = d.import_data("C:/Users/User/Downloads/output.xlsx", "Sheet1")
    print("Data is imported")

    ##################### PROCESS DATA #####################

    # Build and run pipeline to perform feature engineering
    feature_engineering_pipeline = Pipeline(steps=[
        ("Upsampling", s.UpsampleData(2500, 500)),
        ("Downsampling", s.DownsampleData(2500)),
        ("Encoding", e.OneHotEncode()),
        ("Normalization", n.NormalizeData()),
        # ("Feature selection", f.PCATransformer(50)),
    ])

    data = feature_engineering_pipeline.fit_transform(data)

    # Split data
    columns_to_exclude = ['sex', 'type', 'studyid', 'involvementstatus', 'Unnamed: 0', 'visitationdate']
    target = data['involvementstatus']
    data = data.drop(columns=columns_to_exclude)

    X_train, X_rem, y_train, y_rem = train_test_split(data, target, train_size=0.8, random_state=123, shuffle=True)
    X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=123, shuffle=True)

    r.write_to_report("train size", f"{X_train.shape} {y_train.shape}")
    r.write_to_report("test size", f"{X_test.shape} {y_test.shape}")
    r.write_to_report("validation size", f"{X_valid.shape} {y_valid.shape}")

    # Run ML algoritm transformer and perform hyper parameter tuning
    rf_pipeline = Pipeline(steps=[
        ("Random forest", rf.RandomForest(X_train, X_test, y_train, y_test)),
    ])

    rf_pipeline.fit_transform(data)

    r.rename_report_file()

    print("Done!")