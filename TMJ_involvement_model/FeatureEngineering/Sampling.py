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

        id_column = data["ID"]
        sex_column = data["sex"]
        age_column = data["ageatvisitation"]
        diff_column = data["difftdate"]
        y = data["involvementstatus"]
        X = data.drop(columns=["sex", "involvementstatus", "ID", "ageatvisitation", "difftdate"])
        print("Before SMOTE: ", Counter(y))

        non_categorical_columns = ['overjet', 'openbite', 'overbite', 'deepbite', 'openingmm', 'opening', 'protrusionmm', 'protrusion', 'laterotrusionrightmm', 'laterotrusionleftmm']
        categorical_columns = [col for col in X.columns if col not in non_categorical_columns]

        if self.config['n_categories'] == 2:
            sampling_strategy = {1: self.config['time_slice_2_cat'][3]}
            r.write_to_report("smote values", self.config['time_slice_2_cat'][3])
        elif self.config['n_categories'] == 3:
            sampling_strategy = {1: self.config['time_slice_3_cat'][3], 2: self.config['time_slice_3_cat'][4]}
            r.write_to_report("smote values", [self.config['time_slice_3_cat'][3], self.config['time_slice_3_cat'][4]])
        else:
            sampling_strategy = {}

        smote = SMOTENC(categorical_features=categorical_columns, random_state=42, sampling_strategy=sampling_strategy)
        X_res, y_res = smote.fit_resample(X, y)

        final_df = pd.concat([id_column.reset_index(drop=True),sex_column.reset_index(drop=True), age_column.reset_index(drop=True), diff_column.reset_index(drop=True), y_res.reset_index(drop=True), X_res.reset_index(drop=True)], axis=1)  #".reset_index(drop=True)
        print("After SMOTE: ", Counter(y_res))

        r.write_to_report("smote data size", f"{final_df.shape}")

        return final_df

class UpsampleData:

    # DEPRECATED

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

        df_1_upsampled = resample(df_1, replace=True, n_samples=self.n_1, random_state=42)
        df_2_upsampled = resample(df_2, replace=True, n_samples=self.n_2, random_state=42)
        data_upsampled = pd.concat([df_0, df_1_upsampled, df_2_upsampled])

        print(data_upsampled['involvementstatus'].value_counts())

        r.write_to_report("upsampling (cat1)", self.n_1)
        r.write_to_report("upsampling (cat2)", self.n_2)
        return data_upsampled.reset_index(drop=True)

class DownsampleData:

    # DEPRECATED

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

        df_0_downsampled = resample(df_0, replace=False, n_samples=self.n_0, random_state=42)
        data_downsampled = pd.concat([df_0_downsampled, df_1, df_2])

        print(data_downsampled['involvementstatus'].value_counts())

        r.write_to_report("downsampling (cat0)", self.n_0)
        return data_downsampled.reset_index(drop=True)