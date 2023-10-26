import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import Report as r

class PCATransformer:

    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):

        columns_to_exclude = ['sex', 'type', 'studyid', 'involvementstatus', 'Unnamed: 0', 'visitationdate']
        data_no_pca = data.drop(columns=columns_to_exclude)

        # fig, ax = plt.subplots(figsize=(32, 24))
        # sns.heatmap(data.corr())
        # plt.tight_layout()
        # plt.savefig("withoutPCA", dpi=300)

        pca = PCA(n_components=self.n_components)
        pca.fit(data_no_pca)
        data_pca = pca.transform(data_no_pca)
        data_pca = pd.DataFrame(data_pca)

        data = pd.concat([data[columns_to_exclude], data_pca], axis=1)

        #fig, ax = plt.subplots(figsize=(32, 24))
        #sns.heatmap(data_pca.corr())
        #plt.tight_layout()
        #plt.savefig("withPCA", dpi=300)

        r.write_to_report("feature selection", "PCA")
        print("PCA performed")
        return data

class SubsetSelection:

    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):

        r.write_to_report("feature selection", "subset selection")
        print("Subset selection performed")
        return data