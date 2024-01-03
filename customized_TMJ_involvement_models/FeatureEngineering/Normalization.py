import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


class NormalizeData:

    def __init__(self, config, scaler, transform_bool):
        self.config = config
        self.scaler = scaler
        self.transform_bool = transform_bool

    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):

        columns_to_normalize = ['overjet', 'openbite', 'overbite', 'deepbite', 'openingmm', 'opening', 'protrusionmm', 'protrusion', 'laterotrusionrightmm', 'laterotrusionleftmm']

        if self.transform_bool:
            data[columns_to_normalize] = self.scaler.fit_transform(data[columns_to_normalize])
            print("Data normalized")
        else:
            data[columns_to_normalize] = self.scaler.inverse_transform(data[columns_to_normalize])
            print("Data inverse normalized")

        return data


