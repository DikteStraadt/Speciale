# https://elitedatascience.com/imbalanced-classes
import pandas as pd
from sklearn.utils import resample
import Report as r

class UpsampleData:

    def __init__(self, n_1, n_2):
        self.n_1 = n_1
        self.n_2 = n_2

    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):

        print(data['involvementstatus'].value_counts())

        df_0 = data[data.involvementstatus == 0]
        df_1 = data[data.involvementstatus == 1]
        df_2 = data[data.involvementstatus == 2]

        df_1_upsampled = resample(df_1, replace=True, n_samples=self.n_1, random_state=123)
        df_2_upsampled = resample(df_2, replace=True, n_samples=self.n_2, random_state=123)
        data_upsampled = pd.concat([df_0, df_1_upsampled, df_2_upsampled])

        print(data_upsampled['involvementstatus'].value_counts())

        r.write_to_report("upsampling (cat1)", self.n_1)
        r.write_to_report("upsampling (cat2)", self.n_2)
        return data_upsampled.reset_index(drop=True)

class DownsampleData:

    def __init__(self, n_0):
        self.n_0 = n_0

    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):
        
        print(data['involvementstatus'].value_counts())

        df_0 = data[data.involvementstatus == 0]
        df_1 = data[data.involvementstatus == 1]
        df_2 = data[data.involvementstatus == 2]

        df_0_downsampled = resample(df_0, replace=False, n_samples=self.n_0, random_state=123)
        data_downsampled = pd.concat([df_0_downsampled, df_1, df_2])

        print(data_downsampled['involvementstatus'].value_counts())

        r.write_to_report("downsampling (cat0)", self.n_0)
        return data_downsampled.reset_index(drop=True)