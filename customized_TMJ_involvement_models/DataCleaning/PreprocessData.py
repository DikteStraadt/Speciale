from sklearn.pipeline import Pipeline
from DataCleaning.RawData import CleanData as cd, ImportExportData as d, CalculatePatientAge as ca
from DataCleaning.Visitations import ReadWriteVisitations as v, CleanVisitations as cv, LagFeaturesForVisitations as l

def preprocess_data(lag_features):

    # Import raw data
    data = d.import_data("Data/Master_Excel_Sep4.xlsx", "Sheet1")
    print("Data is imported")

    # Build and run pipeline to preprocess data
    preprocessing_pipeline = Pipeline(steps=[
        ("Clean columns", cd.CleanColumns()),
        ("Edit misregistered data", cd.EditData()),
        ("Remove Patients", cd.RemovePatients()),
        ("Convert timestamps", cd.ConvertTimestamps()),
        ("Calculate age for visitation dates", ca.CalculateAge()),
        ("Read visitations", v.ReadVisitations()),
        ("Remove visitations", cv.RemoveVisitations()),
        ("Convert visitation status", cv.ConvertVisitationStatus()),
        ("Add previous values", l.LagFeaturesForVisitations(lag_features)),
        ("Insert zeros", cv.InsertZeros()),
        ("Combine to single DataFrame", v.CombineToDataFrame(lag_features))
    ])

    data = preprocessing_pipeline.transform(data)

    # Save visitations to file
    d.export_data(data, f"Data/cleaned_data.xlsx")
    print("Data exported to file")

    return data