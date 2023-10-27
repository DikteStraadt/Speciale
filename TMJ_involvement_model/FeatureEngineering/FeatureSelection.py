import Report as r

class PCA:

    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):
        columns_to_exclude = ['sex', 'type', 'studyid', 'involvementstatus', 'Unnamed: 0', 'visitationdate']

        scalar = StandardScaler()
        scaled_data = pd.DataFrame(scalar.fit_transform(data))  # scaling the data

        # fig, ax = plt.subplots(figsize=(32, 24))
        # sns.heatmap(scaled_data.corr())
        # plt.tight_layout()
        # plt.savefig("withoutPCA", dpi=300)

        # Applying PCA
        # Taking no. of Principal Components as 3

        pca = PCA(n_components=50)
        pca.fit(scaled_data)
        data_pca = pca.transform(scaled_data)
        data_pca = pd.DataFrame(data_pca)
        # data_pca.head()

        fig, ax = plt.subplots(figsize=(32, 24))
        sns.heatmap(data_pca.corr())
        plt.tight_layout()
        plt.savefig("withPCA", dpi=300)

        print("Done!")

        r.write_to_report("feature selection", "PCA")
        return data

class SubsetSelection:

    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):

        r.write_to_report("feature selection", "Subset selection")
        return data