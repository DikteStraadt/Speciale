import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


class NormalizeData:

    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):
        columns_to_normalize = data

        scaler = RobustScaler()  # StandardScaler(), MinMaxScaler()

        normalized_columns = pd.DataFrame(scaler.fit_transform(columns_to_normalize), columns=columns_to_normalize.columns)

        # Get ranges for each column
        summary = normalized_columns.describe()
        column_ranges = summary.loc[['min', 'max']]

        # print(column_ranges)

        return normalized_columns


