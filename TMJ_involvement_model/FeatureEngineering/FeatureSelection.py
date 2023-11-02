import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

from Utils import Report as r

class ForwardSubsetSelection:
    def __init__(self, estimator, target, config):
        self.estimator = estimator
        self.target = target
        self.config = config

    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):

        data = data.iloc[:, 10:20]

        sfs = SFS(estimator=self.estimator,
                   k_features=self.config["SFS_n_features"],
                   forward=True,
                   floating=False,
                   scoring='accuracy',
                   cv=self.config["cv"],
                   verbose=self.config["verbose"])

        sfs.fit(data, self.target)

        print('Best combination (ACC: %.3f): %s\n ' % (sfs.k_score_, sfs.k_feature_idx_))
        print(pd.DataFrame.from_dict(sfs.get_metric_dict()).T)
        plot_sfs(sfs.get_metric_dict(), kind='std_err')
        plt.grid()
        plt.show()

        features = []
        sfs_df = pd.DataFrame(columns=['Feature'])

        for x in sfs.k_feature_idx_:
            print(data.columns[x])
            features.append(data.columns[x])

        sfs_df['Feature'] = features
        sfs_features = sfs_df['Feature'].tolist()

        data = data.loc[:, sfs_features].copy()

        if "catboost" in str(self.estimator):
            self.estimator = "CatBoostClassifier"

        r.write_to_report("feature selection", "SFS")
        r.write_to_report(f"({str(self.estimator).split('(')[0]}) SFS n_features", len(features))
        r.write_to_report(f"({str(self.estimator).split('(')[0]}) SFS features names ({str(self.estimator).split('(')[0]})", features)
        r.write_to_report(f"({str(self.estimator).split('(')[0]}) SFS accuracy ({str(self.estimator).split('(')[0]})", sfs.k_score_)

        return data

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