import ImportData as d
import ReadVisitations as r
import CleanData as c

if __name__ == '__main__':

    # Import data from Excel sheet
    data1 = d.import_data_sheet1()

    # Remove old patients
    data1 = c.remove_old_patients(data1)

    # Number of patients in reduced data set
    n1 = len(data1)

    # Read visitations from sheet 1
    visitations = r.read_visitations(data1)

    print("Break!")

    # Remove visitations if they do not have visitation date or involvement score
    visitations = c.remove_visitations(visitations)

    print("Done!")

# For each visitation
## for each patient (row)
### Check that they have date, otherwise remove
### Check that they have visitation score, otherwise remove
### For each column
#### If empty, insert 0 or None
## Add visitation to one list of visitations
