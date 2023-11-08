import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


class NormalizeData:

    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):

        columns_to_normalize = ['openingmm', 'opening', 'protrusionmm', 'protrusion', 'laterotrusionrightmm', 'laterotrusionleftmm']
        scaler = StandardScaler()  # StandardScaler(), MinMaxScaler(), RobustScaler()
        data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])

        # Get ranges for each column
        # column_ranges = data.describe().loc[['min', 'max']]

        return data


