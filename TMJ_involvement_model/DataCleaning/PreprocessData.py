from sklearn.pipeline import Pipeline

from DataCleaning.RawData import CleanData as cd, ImportExportData as d
from DataCleaning.Visitations import CleanVisitations as cv
from DataCleaning.Visitations import ReadWriteVisitations as v

def preprocess_data(n_categories):

    # Import raw data
    data = d.import_data("C:/Users/User/Downloads/Master_Excel_Sep4.xlsx", "Sheet1")
    print("Data is imported")

    # Build and run pipeline to preprocess data (no time)
    preprocessing_pipeline = Pipeline(steps=[
        ("Clean columns", cd.CleanColumns()),
        ("Edit misregistered data", cd.EditData()),
        ("Remove Patients", cd.RemovePatients()),
        ("Convert timestamps", cd.ConvertTimestamps()),
        ("Read visitations", v.ReadVisitations()),
        ("Remove visitations", cv.RemoveVisitations()),
        ("Convert visitation status", cv.ConvertVisitationStatus(n_categories)),
        ("Insert zeros", cv.InsertZeros()),
        ("Combine to single DataFrame", v.CombineToDataFrame())
    ])

    data = preprocessing_pipeline.fit_transform(data)

    # Save visitations to file
    d.export_data(data, "C:/Users/User/Downloads/output.xlsx")
    print("Data exported to file")

    return data