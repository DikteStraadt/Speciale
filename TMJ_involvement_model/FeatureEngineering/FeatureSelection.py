import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

from Utils import Report as r

def feature_selection(data, X_train, X_test, estimator, target, config):

    if config["feature_selection"] == "SFS":

        sfs_data = ForwardSubsetSelection(estimator, target, config).transform(data)

        X_train_fs = X_train.loc[:, sfs_data.columns]
        X_test_fs = X_test.loc[:, sfs_data.columns]

        return X_train_fs, X_test_fs

    elif config["feature_selection"] == "clinical":

        clinical_columns = ['painmoveleft', 'painmoveright', 'laterpalpright', 'laterpalpleft', 'translationright', 'translationleft', 'openingmm',
                            'opening', 'protrusionmm', 'protrusion', 'laterotrusionrightmm', 'laterotrusionleftmm',
                            'overjet', 'overbite', 'openbite', 'chewingfunction', 'retrognathism', 'deepbite',
                            'Krepitationright', 'Krepitationleft']

        X_train_fs = X_train.loc[:, clinical_columns]
        X_test_fs = X_test.loc[:, clinical_columns]

        extra = ['asybasis', 'asybasis_0', 'asybasis_1', 'asybasis_2', 'asybasis_3', 'asybasis_4',
                 'asypupilline', 'asypupilline_0', 'asypupilline_1', 'asypupilline_2', 'asypupilline_3', 'asypupilline_4',
                 'drug', 'drug_1', 'drug_2', 'drug_3', 'drug_4', 'drug_5', 'drug_6', 'drug_7', 'drug_8', 'drug_9', 'drug_10', 'drug_11', 'drug_12', 'drug_13', 'drug_14', 'drug_15', 'drug_16', 'drug_17', 'drug_18', 'drug_19', 'drug_20', 'drug_21', 'drug_22', 'drug_23', 'drug_24', 'drug_25', 'drug_26', 'drug_27', 'drug_28', 'drug_29', 'drug_30', 'drug_31',
                 'asyoccl', 'asyoccl_0', 'asyoccl_1', 'asyoccl_2', 'asyoccl_3', 'asyoccl_4',
                 'profile', 'profile_0', 'profile_1', 'profile_2', 'profile_3',
                 'lowerface', 'lowerface_0', 'lowerface_1', 'lowerface_2', 'lowerface_3']

        for column in extra:
            if column in X_train.columns:
                X_train_fs = pd.concat([X_train_fs, X_train[column]], axis=1)
                X_test_fs = pd.concat([X_test_fs, X_test[column]], axis=1)

        r.write_to_report("feature selection", "clinical")
        r.write_to_report("n_features", len(X_train_fs.columns))

        return X_train_fs, X_test_fs

    elif config["feature_selection"] == "short":

        clinical_columns = ['painright', 'painleft', 'painmoveright', 'morningstiffness', 'muscularpainright', 'muscularpainleft', 'headache', 'openingfunction', 'neckpain', 'neckpalpation', 'laterpalpright', 'laterpalpleft', 'postpalpright', 'postpalpleft', 'translationright', 'translationleft', 'masseterright', 'masseterleft', 'temporalisright', 'temporalisleft', 'ptextright', 'ptextleft', 'ptintright', 'ptintleft', 'tempsenright', 'tempsenleft', 'sternoright', 'sternoleft', 'stylomandibularligamentright', 'stylomandibularligamentleft', 'asymmetrymasseterright', 'asymmetrymasseterleft', 'openingmm', 'opening', 'protrusionmm', 'protrusion', 'laterotrusionrightmm', 'laterotrusionleftmm', 'overjet', 'overbite', 'openbite']

        X_train_fs = X_train.loc[:, clinical_columns]
        X_test_fs = X_test.loc[:, clinical_columns]

        extra = ['asybasis', 'asybasis_0', 'asybasis_1', 'asybasis_2', 'asybasis_3', 'asybasis_4',
                 'asypupilline', 'asypupilline_0', 'asypupilline_1', 'asypupilline_2', 'asypupilline_3',
                 'asypupilline_4',
                 'drug', 'drug_1', 'drug_2', 'drug_3', 'drug_4', 'drug_5', 'drug_6', 'drug_7', 'drug_8', 'drug_9',
                 'drug_10', 'drug_11', 'drug_12', 'drug_13', 'drug_14', 'drug_15', 'drug_16', 'drug_17', 'drug_18',
                 'drug_19', 'drug_20', 'drug_21', 'drug_22', 'drug_23', 'drug_24', 'drug_25', 'drug_26', 'drug_27',
                 'drug_28', 'drug_29', 'drug_30', 'drug_31',
                 'asyoccl', 'asyoccl_0', 'asyoccl_1', 'asyoccl_2', 'asyoccl_3', 'asyoccl_4',
                 'profile', 'profile_0', 'profile_1', 'profile_2', 'profile_3',
                 'lowerface', 'lowerface_0', 'lowerface_1', 'lowerface_2', 'lowerface_3',
                 'asymenton', 'asyupmid', 'asylowmi', 'sagittalrelationright', 'sagitalrelationleft']

        for column in extra:
            if column in X_train.columns:
                X_train_fs = pd.concat([X_train_fs, X_train[column]], axis=1)
                X_test_fs = pd.concat([X_test_fs, X_test[column]], axis=1)

        r.write_to_report("feature selection", "short")
        r.write_to_report("n_features", len(X_train_fs.columns))

        return X_train_fs, X_test_fs

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
                   k_features= (1, self.config["SFS_n_features"]),
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