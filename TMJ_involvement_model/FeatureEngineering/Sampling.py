# https://elitedatascience.com/imbalanced-classes
import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.utils import resample
from Utils import Report as r
from imblearn.combine import SMOTEENN
from collections import Counter

class SMOTE:

    def __init__(self, config):
        self.config = config

    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):

        y = data["involvementstatus"]
        X = data.drop(columns=["involvementstatus"])
        print("Before SMOTE: ", Counter(y))

        non_categorical_columns = ['openingmm', 'opening', 'protrusionmm', 'protrusion', 'laterotrusionrightmm', 'laterotrusionleftmm']
        categorical_columns = [col for col in X.columns if col not in non_categorical_columns]

        smote = SMOTENC(categorical_features=categorical_columns, random_state=self.config["random_state"], sampling_strategy={1: 2300, 2: 2000})
        X_res, y_res = smote.fit_resample(X, y)

        final_df = pd.concat([y_res.reset_index(drop=True), X_res.reset_index(drop=True)], axis=1)
        print("After SMOTE: ", Counter(y_res))

        r.write_to_report("smote size", f"{final_df.shape}")

        return final_df

class UpsampleData:

    def __init__(self, n_1, n_2, config):
        self.n_1 = n_1
        self.n_2 = n_2
        self.config = config

    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):

        print(data['involvementstatus'].value_counts())

        df_0 = data[data.involvementstatus == 0]
        df_1 = data[data.involvementstatus == 1]
        df_2 = data[data.involvementstatus == 2]

        df_1_upsampled = resample(df_1, replace=True, n_samples=self.n_1, random_state=self.config["random_state"])
        df_2_upsampled = resample(df_2, replace=True, n_samples=self.n_2, random_state=self.config["random_state"])
        data_upsampled = pd.concat([df_0, df_1_upsampled, df_2_upsampled])

        print(data_upsampled['involvementstatus'].value_counts())

        r.write_to_report("upsampling (cat1)", self.n_1)
        r.write_to_report("upsampling (cat2)", self.n_2)
        return data_upsampled.reset_index(drop=True)

class DownsampleData:

    def __init__(self, n_0, config):
        self.n_0 = n_0
        self.config = config

    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):
        
        print(data['involvementstatus'].value_counts())

        df_0 = data[data.involvementstatus == 0]
        df_1 = data[data.involvementstatus == 1]
        df_2 = data[data.involvementstatus == 2]

        df_0_downsampled = resample(df_0, replace=False, n_samples=self.n_0, random_state=self.config["random_state"])
        data_downsampled = pd.concat([df_0_downsampled, df_1, df_2])

        print(data_downsampled['involvementstatus'].value_counts())

        r.write_to_report("downsampling (cat0)", self.n_0)
        return data_downsampled.reset_index(drop=True)