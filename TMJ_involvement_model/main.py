import sys
import warnings
import xgboost as xgb
from datetime import datetime
from catboost import CatBoostClassifier, Pool

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, precision_score, recall_score, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from xgboost import XGBClassifier

import Report as r
from DataCleaning import PreprocessData as p
from DataCleaning.RawData import ImportExportData as d
from FeatureEngineering import Normalization as n
from FeatureEngineering import Sampling as s
from FeatureEngineering import Encoding as e
from FeatureEngineering import FeatureSelection as f
from ModelTraining import RandomForest as rf
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')
N_CATEGORIES = 3  # 2, 3, 4, 5, 8
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

    r.write_to_report("original data size", f"{data.shape}")

    ##################### PROCESS DATA #####################

    feature_engineering_pipeline = Pipeline(steps=[
        ("Upsampling", s.UpsampleData(2500, 500)),
        ("Downsampling", s.DownsampleData(2500)),
        ("Encoding", e.OneHotEncode()),
        ("Normalization", n.NormalizeData()),
    ])

    # data = feature_engineering_pipeline.fit_transform(data)

    columns_to_exclude = ['sex', 'type', 'studyid', 'involvementstatus', 'Unnamed: 0', 'visitationdate']
    target = data['involvementstatus']
    data = data.drop(columns=columns_to_exclude)

    rf_data = f.ForwardSubsetSelection(RandomForestClassifier(), target).transform(data)
    xg_data = f.ForwardSubsetSelection(XGBClassifier(), target).transform(data)
    # cat_data = f.ForwardSubsetSelection()

    X_train, X_rem, y_train, y_rem = train_test_split(data, target, train_size=0.8, random_state=123, shuffle=True)
    X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=123, shuffle=True)

    r.write_to_report("train size", f"{X_train.shape} {y_train.shape}")
    r.write_to_report("test size", f"{X_test.shape} {y_test.shape}")
    r.write_to_report("validation size", f"{X_valid.shape} {y_valid.shape}")

    rf_model = Pipeline(steps=[
        ("randomforest", RandomForestClassifier()),
    ])

    xg_model = Pipeline(steps=[
       ("xgboost", XGBClassifier()),
    ])

    # y_train = pd.Series(y_train, name='target')
    # catboost = CatBoostClassifier()
    # feature_names = X_train.columns.tolist()
    # X_train.columns = [''] * len(X_train.columns)
    # train_pool = Pool(data=X_train, label=y_train, feature_names=feature_names)
    # catboost.fit(train_pool)

    # cat_model = Pipeline(steps=[
    #     # ("Feature selection", f.SubsetSelection),
    #     ("catboost", catboost),
    # ])

    # catboost.set_feature_names(X_train.columns.tolist())

    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'f1_micro': make_scorer(f1_score, average='micro'),
        'f1_macro': make_scorer(f1_score, average='macro'),
        'f1_weighted': make_scorer(f1_score, average='weighted'),
    }

    rf_param_grid = {
        'randomforest__n_estimators': [100, 200, 300, 400, 500, 600, 700],
        'randomforest__max_depth': [None, 10, 20, 30, 40, 50],
        'randomforest__min_samples_split': [2, 5, 10],
        'randomforest__min_samples_leaf': [1, 2, 4],
        'randomforest__max_features': ['auto', 'sqrt', 'log2'],
        'randomforest__bootstrap': [True, False],
        'randomforest__class_weight': [None, 'balanced'],
        'randomforest__random_state': [123],
    }

    xg_param_grid = {
        'xgboost__max_depth': [3, 7, 10],
        'xgboost__eta': [0.01, 0.1, 0.2],
        'xgboost__objective': ['multi:softmax'],
        'xgboost__min_child_weight': [10, 15, 20, 25],
        'xgboost__colsample_bytree': [0.8, 0.9, 1],
        'xgboost__n_estimators': [300, 400, 500, 600],
        'xgboost__reg_alpha': [0.5, 0.2, 1],
        'xgboost__reg_lambda': [2, 3, 5],
        'xgboost__gamma': [1, 2, 3],
        'xgboost__random_state': [123]
    }

    # cat_param_grid = {
    #     'catboost__iterations': [100, 500],
    #     'catboost__learning_rate': [0.1, 0.2, 1],
    #     'catboost__depth': [3, 6, 10],
    #     'catboost__l2_leaf_reg': [3],
    #     'catboost__border_count': [254],
    #     'catboost__thread_count': [4],
    #     'catboost__cat_features': [0, 1, 2],
    #     'catboost__one_hot_max_size': [20],
    #     'catboost__bagging_temperature': [1.0],
    #     'catboost__early_stopping_rounds': [50],
    #     'catboost__loss_function': ['Logloss', 'MultiClass'],
    #     'catboost__verbose': [False]
    # }

    rf_random_search = RandomizedSearchCV(
        estimator=rf_model,
        param_distributions=rf_param_grid,
        n_iter=2,
        cv=2,
        n_jobs=-1,
        random_state=123,
        scoring=scoring,
        refit='f1_weighted',
        verbose=3
    )

    xg_random_search = RandomizedSearchCV(
        estimator=xg_model,
        param_distributions=xg_param_grid,
        n_iter=2,
        cv=2,
        n_jobs=-1,
        random_state=123,
        scoring=scoring,
        refit='f1_weighted',
        verbose=3
    )

    # cat_random_search = RandomizedSearchCV(
    #     estimator=cat_model,
    #     param_distributions=cat_param_grid,
    #     n_iter=10,
    #     cv=5,
    #     n_jobs=-1,
    #     random_state=123,
    #     scoring=scoring,
    #     refit='f1_weighted',
    #     verbose=3
    # )

    rf_random_search.fit(X_train, y_train)
    xg_random_search.fit(X_train, y_train)

    # cat_random_search.fit(X_train, y_train)

    print("Accuracy (Random forest): ", rf_random_search.best_estimator_.score(X_test, y_test))
    print("Accuracy (XGBoost): ", xg_random_search.best_estimator_.score(X_test, y_test))

    r.write_to_report("best model (random forest)", str(rf_random_search.best_estimator_))
    r.write_to_report("best parameters (random forest)", str(rf_random_search.best_params_))
    r.write_to_report("accuracy  (random forest)", rf_random_search.best_estimator_.score(X_test, y_test))

    r.write_to_report("best model (xgboost)", str(xg_random_search.best_estimator_))
    r.write_to_report("best parameters (xgboost)", str(xg_random_search.best_params_))
    r.write_to_report("accuracy (xgboost)", xg_random_search.best_estimator_.score(X_test, y_test))

    #
    # print("Accuracy (CatBoost): ", cat_random_search.best_estimator_.score(X_test, y_test))
    # r.write_to_report("best model (catboost)", str(cat_random_search.best_estimator_))
    # r.write_to_report("best parameters (catboost)", str(cat_random_search.best_params_))
    # r.write_to_report("accuracy (catboost)", cat_random_search.best_estimator_.score(X_test, y_test))

    r.rename_report_file()

    print("Done!")