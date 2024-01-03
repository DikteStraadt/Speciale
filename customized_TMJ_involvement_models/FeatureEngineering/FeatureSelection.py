import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

from Utils import Report as r

def feature_selection(X_train, y_train, X_test, estimator, config):

    if config["feature_selection"] == "SFS":

        sfs_data = ForwardSubsetSelection(estimator, y_train, config).transform(X_train)

        X_train_fs = X_train.loc[:, sfs_data.columns]
        X_test_fs = X_test.loc[:, sfs_data.columns]

        return X_train_fs, X_test_fs

    elif config["feature_selection"] == "ultra_short_feature_list":
        if config["lag_features"]:
            print("Feature selection lag feature Do Something")
        else:
            ultra_short_features = ['painmove', 'laterpalp', 'translation', 'openingmm',
                                'opening', 'protrusionmm', 'protrusion', 'laterotrusionrightmm', 'laterotrusionleftmm',
                                'overjet', 'overbite', 'openbite', 'chewingfunction', 'retrognathism', 'deepbite',
                                'Krepitation']

        X_train_fs = X_train.loc[:, ultra_short_features]
        X_test_fs = X_test.loc[:, ultra_short_features]

        extra = ['asybasis', 'asybasis_0', 'asybasis_1', 'asybasis_2', 'asybasis_3', 'asybasis_4',
                 'asypupilline', 'asypupilline_0', 'asypupilline_1', 'asypupilline_2', 'asypupilline_3', 'asypupilline_4',
                 'drug', 'Biologics', 'Conventional', 'Eye medicine', 'NSAID', 'Corticosteroids',
                 'asyoccl', 'asyoccl_0', 'asyoccl_1', 'asyoccl_2', 'asyoccl_3', 'asyoccl_4',
                 'profile', 'profile_0', 'profile_1', 'profile_2', 'profile_3',
                 'lowerface', 'lowerface_0', 'lowerface_1', 'lowerface_2', 'lowerface_3']

        for column in extra:
            if column in X_train.columns:
                X_train_fs = pd.concat([X_train_fs, X_train[column]], axis=1)
                X_test_fs = pd.concat([X_test_fs, X_test[column]], axis=1)

        r.write_to_report("feature selection", "ultra short feature list")

        return X_train_fs, X_test_fs

class ForwardSubsetSelection:
    def __init__(self, estimator, target, config):
        self.estimator = estimator
        self.target = target
        self.config = config

    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):

        sfs = SFS(estimator=self.estimator,
                   k_features=(1, self.config["SFS_n_features"]),
                   forward=True,
                   floating=False,
                   scoring='f1_macro',
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
        r.write_to_report(f"({str(self.estimator).split('(')[0]}) SFS cv score ({str(self.estimator).split('(')[0]})", sfs.k_score_)

        return data

class PCATransformer:

    # DEPRECATED

    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):
        data_no_pca = data

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