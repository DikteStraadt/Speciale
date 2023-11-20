import numpy as np
import pandas as pd
from pandas import CategoricalDtype


class SlidingTimeWindowForVisitations:

    def fit(self, visitations_3D, y=None):
        return self

    def transform(self, visitations_3D, y=None):

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

        print("Previous involvement status' added to following visitations")

        columns_to_add = [f'previous_involvement_status_visitation_{j}' for j in range(16)]

        for i, dataframe in enumerate(modified_dataframes):

            dataframe[columns_to_add] = pd.DataFrame([], columns=columns_to_add)

            if 'previous_status'not in dataframe.columns:
                # print(f"Skipping dataframe {i} as it doesn't have previous_status column")
                continue

            #max_index = max(dataframe['previous_status'].apply(lambda x: len(x) if isinstance(x, list) else 0))

            for j in range(i):

                column_name = f'previous_involvement_status_visitation_{j}'

                dataframe[column_name] = dataframe['previous_status'].apply(
                    lambda x: x[j] if isinstance(x, list) and j < len(x) else np.nan
                )

                # dataframe[column_name] = dataframe[column_name].astype('category').cat.codes

            dataframe = dataframe.drop(columns=['previous_status'])

        print("Previous involvement status' converted from list to columns")

        return modified_dataframes
