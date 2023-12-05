from Tradjectory import ImportExportData as d
from Tradjectory import ReadWritePatients as p
from Tradjectory import CleanData as cd
from Tradjectory import CleanPatients as cp
from Tradjectory import Plot as plot

if __name__ == '__main__':

    # Import data from Excel sheet
    data = d.import_data("C:/Users/User/Downloads/Master_Excel_Sep4.xlsx", "Sheet1")
    indexes_to_remove = [0, 4, 146, 150, 178, 269, 297, 421, 532, 557, 558, 621, 646, 654, 685, 852, 901, 917, 971]
    data = data.drop(indexes_to_remove).reset_index(drop=True)
    print("Data imported and cleaned")

    # Rename columns with does not follow naming convention
    data = cd.clean_columns(data)
    print("Columns renamed")

    # Convert time stamps
    data = cd.convert_time_stamps(data)
    print("Time stamps converted")

    # Change format of visitations
    patients = p.read_patients(data)
    print("Format converted")

    # Remove visitations if they do not have involvement status or visitation date
    patients = cp.remove_visitations(patients)
    print("Empty visitations removed")

    # Plot tradjectories
    plot.trajectory_plot(patients)
    print("Trajectory plot plotted")

    print("Done!")
