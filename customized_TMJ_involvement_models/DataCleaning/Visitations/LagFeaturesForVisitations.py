import numpy as np
import pandas as pd
import math

# LAG FEATURES (see GitHub)

def add_lag_features(visitations_3D, N, lag_feature_list):

    modified_dataframes = []
    first_feature = True

    for feature in lag_feature_list:

        temp_dataframes = []

        if first_feature:
            dataframe_3D = visitations_3D
            first_feature = False
        else:
            dataframe_3D = modified_dataframes
            for i in range(N-1, -1, -1):  # Add the first N visitations again
                dataframe_3D.insert(0, visitations_3D[i])

        for i, dataframe in enumerate(dataframe_3D):

            current_dataframe = dataframe.copy()

            if i > N-1:  # Udelad de N første eksaminationer
                lag_values = []

                for index, row in current_dataframe.iterrows():
                    current_id = row['ID']
                    current_id_base = current_id.split("_")[0]
                    lag_feature_value = []

                    prev_i = i-1
                    while len(lag_feature_value) < N and prev_i >= 0:
                        prev_dataframe = visitations_3D[prev_i]

                        if prev_i == 0:
                            previous_lag_feature_column_name = f"{feature}"
                        else:
                            previous_lag_feature_column_name = f"{feature}{prev_i}"

                        if any(prev_dataframe['ID'].str.startswith(current_id_base)):
                            matching_patients = prev_dataframe[prev_dataframe['ID'].str.startswith(current_id_base)]

                            if math.isnan(matching_patients[previous_lag_feature_column_name]):
                                lag_feature_value.append(0.0)
                            else:
                                lag_feature_value.extend(matching_patients[previous_lag_feature_column_name].tolist())

                        prev_i -= 1

                    lag_values.append(lag_feature_value)

                current_dataframe[f'{feature}-{N}'] = lag_values
                new_lag_feature = current_dataframe[f'{feature}-{N}']

                if N == 1:
                    current_dataframe = current_dataframe[new_lag_feature.apply(lambda x: len(str(x)) >= 4)]
                    columns_to_add = [f'{feature}-1']
                elif N == 2:
                    current_dataframe = current_dataframe[new_lag_feature.apply(lambda x: len(str(x)) >= 9)]
                    columns_to_add = [f'{feature}-1', f'{feature}-2']

                current_dataframe[columns_to_add] = pd.DataFrame(current_dataframe[f'{feature}-{N}'].tolist(), index=current_dataframe.index)

                temp_dataframes.append(current_dataframe)

        modified_dataframes = temp_dataframes

    print("Lag features added to feature space")

    return modified_dataframes

class AddLagFeatures:

    def __init__(self, lag_features_boolean, n_lag_features, lag_feature_list):
        self.lag_features_boolean = lag_features_boolean
        self.n_lag_features = n_lag_features
        self.lag_feature_list = lag_feature_list

    def fit(self, data_without_lag_features, y=None):
        return self

    def transform(self, data_without_lag_features, y=None):

        if self.lag_features_boolean:
            data_with_lag_features = add_lag_features(data_without_lag_features, self.n_lag_features, self.lag_feature_list)
            return data_with_lag_features
        else:
            return data_without_lag_features