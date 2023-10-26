import pandas as pd
import Report as r

class OneHotEncode:

    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):

        new_df = data
        columns_to_encode = ['drug', 'asypupilline', 'asybasis', 'asymenton', 'asyoccl', 'asyupmid', 'asylowmi',
                             'profile', 'lowerface', 'spacerelationship', 'sagittalrelationright',
                             'sagitalrelationleft', 'transversal']

        for column_name in columns_to_encode:
            nominal_encoded_column = pd.get_dummies(data[column_name].astype(str), prefix=column_name)
            new_df.drop(column_name, axis=1, inplace=True)
            new_df = new_df.join(nominal_encoded_column)

        r.write_to_report("encoding", "one hot")
        print("Data encoded")

        return new_df

class EntityEmbeddingEncoding:

    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):

        r.write_to_report("encoding", "entity embedding")
        return data