import numpy as np
import DataCleaning.Utils.Utils as u

class RemoveVisitations:

    def fit(self, visitations_3D, y=None):
        return self

    def transform(self, visitations_3D, y=None):

        visitation_number = 1

        # Visitation 0
        visitation_0 = visitations_3D[0]
        # column_2 = visitation_0.iloc[:, 2]
        involvement_status_column = visitation_0['involvement_status_0']
        indexes_to_be_removed = []

        for i, value in enumerate(involvement_status_column):
            if np.isnan(value):
                indexes_to_be_removed.append(i)

        for j in sorted(indexes_to_be_removed, reverse=True):
            visitation_0 = visitation_0.drop(j)

        visitations_3D[0] = visitation_0.reset_index(drop=True)

        # Visitation 1-16
        for visitation_2D in visitations_3D[1:17]:

            involvement_status_column_name = next(
                (col for col in visitation_2D.columns if col.startswith("involvement_status_")), None)
            indexes_to_be_removed = []

            for i, value in enumerate(visitation_2D[involvement_status_column_name]):
                if np.isnan(value):
                    indexes_to_be_removed.append(i)

            for j in sorted(indexes_to_be_removed, reverse=True):
                visitation_2D = visitation_2D.drop(j)

            visitations_3D[visitation_number] = visitation_2D.reset_index(drop=True)
            visitation_number = visitation_number + 1

        print("Visitations with no visitation status removed")

        return visitations_3D

class ConvertVisitationStatus:

    def transform(self, visitations_3D, y=None):

        visitations_number = 1

        # Visitation 0
        for i, value in enumerate(visitations_3D[0]['involvement_status_0']):
            visitations_3D[0].at[i, 'involvement_status_0'] = u.transform_visitation_status(value)

        # Visitation 1-16
        for visitation_2D in visitations_3D[1:17]:

            involvement_status_column_name = next(
                (col for col in visitation_2D.columns if col.startswith("involvement_status_")), None)

            for i, value in enumerate(visitation_2D[involvement_status_column_name]):
                visitations_3D[visitations_number].at[i, involvement_status_column_name] = u.transform_visitation_status(value)

            visitations_number = visitations_number + 1

        print("Visitation status converted")

        return visitations_3D

class InsertZeros:

    def fit(self, visitations_3D, y=None):
        return self

    def transform(self, visitations_3D, y=None):

        prefixes_to_exclude = ['study_id', 'type', 'sex', 'columns_to_include', 'first', 'second', 'third', 'fourth',
                               'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth', 'eleventh', 'twelfth',
                               'thirteenth', 'fourteenth', 'fifteenth', 'sixteenth', 'seventeenth']

        for visitation_2D in visitations_3D[0:17]:

            columns_to_include = [col for col in visitation_2D.columns if
                                  not any(col.startswith(prefix) for prefix in prefixes_to_exclude)]
            visitation_2D_selected = visitation_2D[columns_to_include]

            for i, row in visitation_2D_selected.iterrows():

                for column_name, cell_value in row.items():

                    if type(cell_value) == str:
                        if cell_value == "nan":
                            visitation_2D.at[i, column_name] = "0"
                    elif type(cell_value) == float:
                        if np.isnan(cell_value):
                            visitation_2D.at[i, column_name] = 0
                    #else:
                    #    print(f'Other data type than string or float, row: {i}, column: {column_name}, value: {cell_value}')

        print("Zeros inserted")

        return visitations_3D