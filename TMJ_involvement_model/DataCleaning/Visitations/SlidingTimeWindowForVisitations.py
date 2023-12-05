import numpy as np
import pandas as pd

def sliding_time_window_y_15_visitations(visitations_3D):
    modified_dataframes = []

    for i, dataframe in enumerate(visitations_3D):
        current_dataframe = dataframe.copy()

        if i > 0:

            previous_status_values = []

            for index, row in current_dataframe.iterrows():
                current_id = row['ID']
                current_id_base = current_id.split("_")[0]
                previous_status_value = []

                for prev_i in range(i):
                    prev_dataframe = visitations_3D[prev_i]
                    previous_status_column_name = f"involvement_status_{prev_i}"

                    if any(prev_dataframe['ID'].str.startswith(current_id_base)):
                        matching_patients = prev_dataframe[prev_dataframe['ID'].str.startswith(current_id_base)]
                        previous_status_value.extend(matching_patients[previous_status_column_name].tolist())

                previous_status_values.append(previous_status_value)

            current_dataframe['previous_status'] = previous_status_values

        modified_dataframes.append(current_dataframe)

    columns_to_add = [f'previous_involvement_status_visitation_{j}' for j in range(16)]

    for i, dataframe in enumerate(modified_dataframes):

        dataframe[columns_to_add] = pd.DataFrame([], columns=columns_to_add)

        if 'previous_status' not in dataframe.columns:
            # print(f"Skipping dataframe {i} as it doesn't have previous_status column")
            continue

        for j in range(i):
            column_name = f'previous_involvement_status_visitation_{j}'

            dataframe[column_name] = dataframe['previous_status'].apply(
                lambda x: x[j] if isinstance(x, list) and j < len(x) else np.nan
            )

    print("Previous involvement status' converted from list to columns (all previous status')")

    return modified_dataframes

def sliding_time_window_y_2_visitations(visitations_3D):
    modified_dataframes = []

    for i, dataframe in enumerate(visitations_3D):
        current_dataframe = dataframe.copy()

        if i > 1:
            previous_status_values = []

            for index, row in current_dataframe.iterrows():
                current_id = row['ID']
                current_id_base = current_id.split("_")[0]
                previous_status_value = []

                prev_i = i-1
                while len(previous_status_value) < 2 and prev_i >= 0:
                    prev_dataframe = visitations_3D[prev_i]
                    previous_status_column_name = f"involvement_status_{prev_i}"

                    if any(prev_dataframe['ID'].str.startswith(current_id_base)):
                        matching_patients = prev_dataframe[prev_dataframe['ID'].str.startswith(current_id_base)]
                        previous_status_value.extend(matching_patients[previous_status_column_name].tolist())

                    prev_i -= 1

                previous_status_values.append(previous_status_value)

            current_dataframe['previous_status'] = previous_status_values
            current_dataframe = current_dataframe[current_dataframe['previous_status'].apply(lambda x: len(str(x)) >= 9)]
            modified_dataframes.append(current_dataframe)

    columns_to_add = ['previous_involvement_status_visitation_y-1', 'previous_involvement_status_visitation_y-2']
    for dataframe in enumerate(modified_dataframes):

        dataframe[1][columns_to_add] = pd.DataFrame([], columns=columns_to_add)

        for j in range(2):

            column_name = f'previous_involvement_status_visitation_y-{j+1}'

            dataframe[1][column_name] = dataframe[1]['previous_status'].apply(
                lambda x: x[j] if isinstance(x, list) and j < len(x) else np.nan
            )

        dataframe = dataframe[1]

    print("Previous involvement status' converted from list to columns (two previous status)")

    return modified_dataframes

class SlidingTimeWindowForVisitations:

    def __init__(self, previous_two_values):
        self.previous_two_values = previous_two_values

    def fit(self, visitations_3D, y=None):
        return self

    def transform(self, visitations_3D, y=None):

        if self.previous_two_values == "y-2":
            transformed_data = sliding_time_window_y_2_visitations(visitations_3D)
        elif self.previous_two_values == "y-15":
            transformed_data = sliding_time_window_y_15_visitations(visitations_3D)
        else:
            transformed_data = visitations_3D

        return transformed_data