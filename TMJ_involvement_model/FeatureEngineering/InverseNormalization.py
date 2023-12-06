from sklearn.preprocessing import StandardScaler


class InverseNormalizeData:

    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):

        columns_to_inverse = ['overjet', 'openbite', 'overbite', 'deepbite', 'openingmm', 'opening', 'protrusionmm',
                                'protrusion', 'laterotrusionrightmm', 'laterotrusionleftmm']



        scaler = StandardScaler()
        data[columns_to_inverse] = scaler.inverse_transform(data[columns_to_inverse])
