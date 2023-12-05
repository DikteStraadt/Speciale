from sklearn.pipeline import Pipeline
from DataCleaning.RawData import CleanData as cd, ImportExportData as d, TimeSliceData as f
from DataCleaning.Visitations import ReadWriteVisitations as v, CleanVisitations as cv, SlidingTimeWindowForVisitations as s

def preprocess_data(n_categories, previous_values):

    # Import raw data
    data = d.import_data("Data/Master_Excel_Sep4.xlsx", "Sheet1")
    print("Data is imported")

    # Build and run pipeline to preprocess data
    preprocessing_pipeline = Pipeline(steps=[
        ("Clean columns", cd.CleanColumns()),
        ("Edit misregistered data", cd.EditData()),
        ("Remove Patients", cd.RemovePatients()),
        ("Convert timestamps", cd.ConvertTimestamps()),
        ("Calculate age and diff for visitation dates", f.TimeSliceData()),
        ("Read visitations", v.ReadVisitations()),
        ("Remove visitations", cv.RemoveVisitations()),
        ("Convert visitation status", cv.ConvertVisitationStatus(n_categories)),
        ("Add previous values", s.SlidingTimeWindowForVisitations(previous_values)),
        ("Insert zeros", cv.InsertZeros()),
        ("Combine to single DataFrame", v.CombineToDataFrame(previous_values))
    ])

    data = preprocessing_pipeline.transform(data)

    # Save visitations to file
    d.export_data(data, f"Data/cleaned_data_{n_categories}_cat.xlsx")
    print("Data exported to file")

    return data