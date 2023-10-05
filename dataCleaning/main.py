import ImportExportData as d
import ReadWriteVisitations as v
import ReadWritePatients as p
import CleanData as cd
import CleanVisitations as cv

def import_clean_data(path, sheet):

    # Import data from Excel sheet
    data = d.import_data(path, sheet)
    print("Data imported and cleaned")

    # Rename columns with does not follow naming convention
    data = cd.rename_columns(data)
    print("Columns renamed")

    # Remove old patients
    data = cd.remove_old_patients(data)
    print("Old patients removed")

    return data

def read_visitations(path, sheet):

    # Import and clean raw data
    data = import_clean_data(path, sheet)
    print("Data imported and cleaned")

    # Change format of visitations
    visitations = v.read_visitations(data)
    print("Format converted")

    # Remove visitations if they do not have involvement score
    visitations = cv.remove_visitations(visitations)
    print("Empty visitations removed")

    # Insert zero in all empty entries
    visitations = cv.insert_zeros(visitations)
    print("Zeroes inserted")

    # Add all visitations to one dataframe
    visitations_list = v.combine_to_dataframe(visitations)
    print("Visitations combined to one data frame")

    # Save visitations to file
    d.export_data(visitations_list, "C:/Users/User/Downloads/output.xlsx")
    print("Data saved to file")

    return visitations_list

def read_patients(path, sheet):

    # Import and clean raw data
    data = import_clean_data(path, sheet)

    # Change format of patients
    patients = p.read_patients(data)
    print("Format converted")

    return patients

if __name__ == '__main__':

    read_visitations("C:/Users/User/Downloads/Master_Excel_Sep4.xlsx", "Sheet1")
