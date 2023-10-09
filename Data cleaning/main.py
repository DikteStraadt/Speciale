import ImportExportData as d
import ReadWriteVisitations as v
import ReadWritePatients as p
import CleanData as cd
import CleanVisitations as cv
import CleanPatients as cp

if __name__ == '__main__':

    # Import data from Excel sheet
    data = d.import_data("C:/Users/User/Downloads/Master_Excel_Sep4.xlsx", "Sheet1")
    print("Data imported and cleaned")

    # Rename columns with does not follow naming convention
    data = cd.clean_columns(data)
    print("Columns renamed")

    # Changes to data according to Stratos'
    data = cd.edit_misregistered_data(data)
    print("Misregistered data edited")

    # Remove old patients
    data = cd.remove_old_patients(data)
    print("Old patients removed")

    # Convert time stamps
    data = cd.convert_time_stamps(data)
    print("Time stamps converted")

########################## READ PATIENTS ##########################

    # Change format of visitations
    patients = p.read_patients(data)
    print("Format converted")

    # Remove visitations if they do not have involvement status
    patients = cp.remove_visitations(patients)
    print("Empty visitations removed")

    # Convert involvement status
    patients = cp.convert_visitation_status(patients)
    print("Involvement status converted")

    # Insert zero in all empty entries
    patients = cp.insert_zeros(patients)
    print("Zeroes inserted")

########################## READ VISITATIONS ##########################

    # Change format of visitations
    # visitations = v.read_visitations(data)
    # print("Format converted")

    # Remove visitations if they do not have involvement status
    # visitations = cv.remove_visitations(visitations)
    # print("Empty visitations removed")

    # Convert involvement status
    # visitations = cv.convert_visitation_status(visitations)
    # print("Involvement status converted")

    # Insert zero in all empty entries
    # visitations = cv.insert_zeros(visitations)
    # print("Zeroes inserted")

    # Add all visitations to one dataframe
    # data = v.combine_to_dataframe(visitations)
    # print("Visitations combined to one data frame")

    # Save visitations to file
    # d.export_data(data, "C:/Users/User/Downloads/output.xlsx")
    # print("Data saved to file")

########################## DONE ##########################

    print("Done!")
