import pandas as pd

# Import data
data = pd.read_excel("C:/Users/User/Downloads/Master_Excel_Sep4.xlsx")

# List of list of visitations
vistation_headers = []
list_list_visitations = []

headers_visitation_6 = ('seventh_US', 'transversal6')

vistation_headers.extend([headers_visitation_6])

for visitation in vistation_headers:
    start_index = data.columns.get_loc(visitation[0])
    end_index = data.columns.get_loc(visitation[1])
    list_visitations = data.iloc[:, start_index : end_index + 1]
    list_list_visitations.append(list_visitations)

# Column 1
