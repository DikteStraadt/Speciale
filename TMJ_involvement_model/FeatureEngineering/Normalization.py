import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


class NormalizeData:

    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):

        excluded_columns = ['Unnamed: 0', 'studyid', 'visitationdate', 'involvementstatus']
        columns_to_normalize = data.drop(columns=excluded_columns)

        scaler = RobustScaler()  # StandardScaler(), MinMaxScaler()

        normalized_columns = pd.DataFrame(scaler.fit_transform(columns_to_normalize), columns=columns_to_normalize.columns)
        normalized_df = pd.concat([data[excluded_columns], normalized_columns], axis=1)

        # Get ranges for each column
        summary = normalized_df.describe()
        column_ranges = summary.loc[['min', 'max']]

        print(normalized_df)


