import warnings
from DataCleaning import ImportExportData as d
from FeatureEngineering import Encoding as e
from FeatureEngineering import Normalization as n
from FeatureEngineering import Sampling as s
from ModelTraining import RandomForest as r
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')
N_CATEGORIES = 3

if __name__ == '__main__':

    # Import raw data
    # data = d.import_data("C:/Users/User/Downloads/Master_Excel_Sep4.xlsx", "Sheet1")
    # print("Data is imported")

    # Run Pipeline with data without time
    # pipeline = Pipeline(steps=[
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

    # data = pipeline.fit_transform(data)

    # Save visitations to file
    # d.export_data(data, "C:/Users/User/Downloads/output.xlsx")
    # print("Data exported to file")

    ##################################################################

    # Import formatted visitation data
    data = d.import_data("C:/Users/User/Downloads/output.xlsx", "Sheet1")
    print("Data is imported")

    pipeline = Pipeline(steps=[
        ("Up sampling", s.UpsampleData()),
        ("Down sampling", s.DownsampleData()),
        # ("Feature selection", X)
        # ("Encoding", e.OneHotEncode()),
        # ("Normalization", n.NormalizeData()),
        ("Random forest", r.RandomForest())
    ])

    data = pipeline.fit_transform(data)

    print("Done!")








