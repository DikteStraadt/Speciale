import warnings

import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split

import Report as report
from DataCleaning import ImportExportData as d
from FeatureEngineering import Encoding as e
from FeatureEngineering import Normalization as n
from FeatureEngineering import Sampling as s
from FeatureEngineering import FeatureSelection as f
from ModelTraining import RandomForest as r
from ModelTraining import XGBoost as x
from ModelTraining import CatBoost as c
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')
N_CATEGORIES = 3  # 3, 4, 5, 8

if __name__ == '__main__':

    # Import raw data
    # data = d.import_data("C:/Users/User/Downloads/Master_Excel_Sep4.xlsx", "Sheet1")
    # print("Data is imported")

    # Build and run pipeline to preprocess data (no time)
    # preprocessing_pipeline = Pipeline(steps=[
    #     ("Clean columns", cd.CleanColumns()),
    #     ("Edit misregistered data", cd.EditData()),
    #     ("Remove Patients", cd.RemovePatients()),
    #     ("Convert timestamps", cd.ConvertTimestamps()),
    #     ("Read visitations", v.ReadVisitations()),
    #     ("Remove visitations", cv.RemoveVisitations()),
    #     ("Convert visitation status", cv.ConvertVisitationStatus(N_CATEGORIES)),
    #     ("Insert zeros", cv.InsertZeros()),
    #     ("Combine to single DataFrame", v.CombineToDataFrame())
    # ])

    # data = preprocessing_pipeline.fit_transform(data)

    # Save visitations to file
    # d.export_data(data, "C:/Users/User/Downloads/output.xlsx")
    # print("Data exported to file")

    ##################################################################

    # Import formatted visitation data
    data = d.import_data("C:/Users/User/Downloads/output.xlsx", "Sheet1")
    print("Data is imported")

    # Create empty report file
    report.create_empty_report()

    # Build and run pipeline to perform feature engineering
    feature_engineering_pipeline = Pipeline(steps=[
        ("Upsampling", s.UpsampleData(2500, 500)),
        ("Downsampling", s.DownsampleData(2500)),
        # ("Feature selection", f.SubsetSelection()),
        # ("Encoding", e.OneHotEncode()),
        ("Normalization", n.NormalizeData()),
        ("Random forest", r.RandomForest(10, 'entropy', 42)),
        # ("XGBoost", x.XGBoost(10, 1, 8))
        # ("CatBoost", c.CatBoost(100, 10, 0.1))
    ])

    # Rename report file
    report.rename_report_file()

    print("Done!")








