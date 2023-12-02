from sklearn.preprocessing import StandardScaler


class InverseNormalizeData:

    def __init__(self, config):
        self.config = config

    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):
        if self.config['encoding_embedding']:
            columns_to_inverse = ['overjet', 'openbite', 'overbite', 'deepbite', 'openingmm', 'opening',
                                    'protrusionmm', 'protrusion', 'laterotrusionrightmm', 'laterotrusionleftmm',
                                    'asypupilline', 'asybasis', 'asyoccl', 'asymenton', 'profile', 'asyupmid',
                                    'asylowmi', 'lowerface', 'sagittalrelation']

        elif not self.config['encoding_embedding']:
            columns_to_inverse = ['overjet', 'openbite', 'overbite', 'deepbite', 'openingmm', 'opening',
                                    'protrusionmm', 'protrusion', 'laterotrusionrightmm', 'laterotrusionleftmm']


        scaler = StandardScaler()
        data[columns_to_inverse] = scaler.inverse_transform(data[columns_to_inverse])
