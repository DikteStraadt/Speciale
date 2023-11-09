import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


class NormalizeData:

    def __init__(self, config):
        self.config = config

    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):

        if self.config['encoding_embedding']:
            columns_to_normalize = ['openingmm', 'opening', 'protrusionmm', 'protrusion', 'laterotrusionrightmm', 'laterotrusionleftmm', 'drug', 'asypupilline', 'asybasis', 'asyoccl', 'asymenton', 'profile', 'asyupmid', 'asylowmi', 'lowerface', 'sagittalrelation']
        elif not self.config['encoding_embedding']:
            columns_to_normalize = ['openingmm', 'opening', 'protrusionmm', 'protrusion', 'laterotrusionrightmm', 'laterotrusionleftmm']

        scaler = StandardScaler()  # StandardScaler(), MinMaxScaler(), RobustScaler()
        data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])

        # Get ranges for each column
        # column_ranges = data.describe().loc[['min', 'max']]

        return data


