import numpy as np
def remove_visitations(visitations_3D):

    visitation_number = 1

    # Visitation 0
    visitation_0 = visitations_3D[0]
    column_2 = visitation_0.iloc[:, 2]
    indexes_to_be_removed = []

    for i, value in enumerate(column_2):
        if np.isnan(value):
            indexes_to_be_removed.append(i)

    for j in sorted(indexes_to_be_removed, reverse=True):
        visitation_0 = visitation_0.drop(j)

    visitations_3D[0] = visitation_0.reset_index(drop=True)

    # Visitation 1-16
    for visitation_2D in visitations_3D[1:17]:

        column_1 = visitation_2D.iloc[:, 1]
        indexes_to_be_removed = []

        for i, value in enumerate(column_1):
            if np.isnan(value):
                indexes_to_be_removed.append(i)

        for j in sorted(indexes_to_be_removed, reverse=True):
            visitation_2D = visitation_2D.drop(j)

        visitations_3D[visitation_number] = visitation_2D.reset_index(drop=True)
        visitation_number = visitation_number + 1

    return visitations_3D

def insert_zeros(visitations_3D):

    prefixes_to_exclude = ['Alder_ved_afslut', 'columns_to_include', 'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', ' eighths', 'nineth', 'tenths', 'eleventh', 'twelfth', 'thirteenth', 'fourteenth', 'fifteenth', 'sixteenth']

    for visitation_2D in visitations_3D[0:17]:

        columns_to_include = [col for col in visitation_2D.columns if not any(col.startswith(prefix) for prefix in prefixes_to_exclude)]
        visitation_2D_selected = visitation_2D[columns_to_include]

        for i, row in visitation_2D_selected.iterrows():

            for column_name, cell_value in row.items():

                if type(cell_value) == str:
                    if cell_value == "nan":
                        visitation_2D.at[i, column_name] = "0"
                elif type(cell_value) == float:
                    if np.isnan(cell_value):
                        visitation_2D.at[i, column_name] = 0
                else:
                    print(f'Error other data type, row: {i}, column: {column_name}, value: {cell_value}')

    return visitations_3D