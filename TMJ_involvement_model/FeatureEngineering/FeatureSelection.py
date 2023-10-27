import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
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

        fig, ax = plt.subplots(figsize=(32, 24))
        sns.heatmap(data_no_pca.corr())
        plt.tight_layout()
        plt.savefig("withoutPCA", dpi=300)

        pca = PCA(n_components=self.n_components)
        pca.fit(data_no_pca)
        data_pca = pca.transform(data_no_pca)
        data_pca = pd.DataFrame(data_pca)

        fig, ax = plt.subplots(figsize=(32, 24))
        sns.heatmap(data_pca.corr())
        plt.tight_layout()
        plt.savefig("withPCA", dpi=300)

        data = pd.concat([data[columns_to_exclude], data_pca], axis=1)

        r.write_to_report("feature selection", "PCA")
        r.write_to_report("PCA components", self.n_components)

        print("PCA performed")
        return data

class ForwardSubsetSelection:
    def __init__(self, estimator, target ,sfs_feature, sfs_cv, sfs_scoring):
        """
        :param estimator: The type of classifier (XGBoost, CatBoost, Random Forest)
        :param target: The target column of the dataset
        :param sfs_feature: Maximum number of features
        :param sfs_cv: Number of folds for cross-validation
        :param sfs_scoring: Scoring metrics to be used
        """
        self.estimator = estimator
        self.target = target
        self.sfs_feature = sfs_feature
        self.sfs_cv = sfs_cv
        self.sfs_scoring = sfs_scoring


    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):
        cv = StratifiedKFold(n_splits=self.sfs_cv, random_state=101, shuffle=True)

        sfs1 = SFS(estimator=self.estimator,
                   k_features=(1, self.sfs_feature),
                   forward=True,
                   floating=False,
                   scoring=self.sfs_scoring,
                   cv=self.sfs_cv)

        sfs1.fit(data, self.target)

        print('Best combination (ACC: %.3f): %s\n ' % (sfs1.k_score_, sfs1.k_feature_idx_))
        print(pd.DataFrame.from_dict(sfs1.get_metric_dict()).T)
        plot_sfs(sfs1.get_metric_dict(), kind='std_err')
        plt.grid()
        plt.show()

        features = []
        sfs_df = pd.DataFrame(columns=['Feature'])

        for x in sfs1.k_feature_idx_:
            print(data.columns[x])
            features.append(data.columns[x])


        sfs_df['Feature'] = features
        sfs_features = sfs_df['Feature'].tolist()

        data = data.loc[:, sfs_features].copy()

        print("~~Feature selection done!!~~")

        return data