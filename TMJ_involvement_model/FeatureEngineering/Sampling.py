# https://elitedatascience.com/imbalanced-classes
import pandas as pd
from sklearn.utils import resample


class UpsampleData:

    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):

        print(data['involvementstatus'].value_counts())

        df_0 = data[data.involvementstatus == 0]
        df_1 = data[data.involvementstatus == 1]
        df_2 = data[data.involvementstatus == 2]

        df_1_upsampled = resample(df_1, replace=True, n_samples=2500, random_state=123)
        df_2_upsampled = resample(df_2, replace=True, n_samples=500, random_state=123)
        data_upsampled = pd.concat([df_0, df_1_upsampled, df_2_upsampled])

        print(data_upsampled['involvementstatus'].value_counts())

        return data_upsampled

class DownsampleData:

    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):
        
        print(data['involvementstatus'].value_counts())

        df_0 = data[data.involvementstatus == 0]
        df_1 = data[data.involvementstatus == 1]
        df_2 = data[data.involvementstatus == 2]

        df_0_downsampled = resample(df_0, replace=False, n_samples=2500, random_state=123)
        data_downsampled = pd.concat([df_0_downsampled, df_1, df_2])

        print(data_downsampled['involvementstatus'].value_counts())
        
        return data_downsampled