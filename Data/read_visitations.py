import pandas as pd

# Import data
data = pd.read_excel("C:/Users/User/Downloads/Master_Excel_Sep4.xlsx")
n = len(data)

# List of list of visitations
vistation_headers = []
list_list_visitations = []

headers_patient_info = ('ID', 'overall TMJ involvement')
headers_visitation_0 = ('first_visitation', 'Alder_ved_afslut')
headers_visitation_1 = ('second_US', 'transversal1')
headers_visitation_2 = ('third_US', 'transversal2')
headers_visitation_3 = ('fourth_US', 'transversal3')
headers_visitation_4 = ('fifth_US', 'transversal4')
headers_visitation_5 = ('sixth_US', 'transversal5')
headers_visitation_6 = ('seventh_US', 'transversal6')
headers_visitation_7 = ('eigth_US', 'transversal7')
headers_visitation_8 = ('nineth_US', 'transversal8')
headers_visitation_9 = ('tenth_US', 'transversal9')
headers_visitation_10 = ('eleventh_US', 'transversal10')
headers_visitation_11 = ('twelth_US', 'transversal11')
headers_visitation_12 = ('thirteenth_US', 'transversal12')
headers_visitation_13 = ('fourteenth_US', 'transversal13')
headers_visitation_14 = ('fifteenth_US', 'transversal14')
headers_visitation_15 = ('sixteenth_US', 'transversal15')
headers_visitation_16 = ('seventeenth_US', 'transversal16')

vistation_headers.extend([headers_patient_info, headers_visitation_0, headers_visitation_1, headers_visitation_2, headers_visitation_3, headers_visitation_4, headers_visitation_5, headers_visitation_6, headers_visitation_7, headers_visitation_8, headers_visitation_9, headers_visitation_10, headers_visitation_11, headers_visitation_12, headers_visitation_13, headers_visitation_14, headers_visitation_15, headers_visitation_16])

for visitation in vistation_headers:
    start_index = data.columns.get_loc(visitation[0])
    end_index = data.columns.get_loc(visitation[1])
    list_visitations = data.iloc[:, start_index : end_index + 1]
    list_list_visitations.append(list_visitations)

print("Done")
