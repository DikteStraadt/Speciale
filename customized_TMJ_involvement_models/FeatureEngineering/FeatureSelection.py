import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

from Utils import Report as r

def feature_selection(X_train, y_train, X_test, estimator, config):

    if config["SFS_feature_selection"]:

        sfs_data = ForwardSubsetSelection(estimator, y_train, config).transform(X_train)

        X_train_fs = X_train.loc[:, sfs_data.columns]
        X_test_fs = X_test.loc[:, sfs_data.columns]

        return X_train_fs, X_test_fs

    else:

        if config["lag_features"]:
            feature_list = [col for col in X_train.columns if any(col.startswith(prefix) for prefix in config['feature_list'])]
        else:
            feature_list_with_new_drugs = config['feature_list'] + ['Biologics', 'Conventional', 'Eye medicine', 'NSAID', 'Corticosteroids']
            feature_list_with_new_drugs.remove('drug')
            feature_list = config['feature_list'] if 'drug' not in config['feature_list'] else feature_list_with_new_drugs

        X_train_fs = X_train.loc[:, feature_list]
        X_test_fs = X_test.loc[:, feature_list]

        r.write_to_report("feature selection", "list")

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