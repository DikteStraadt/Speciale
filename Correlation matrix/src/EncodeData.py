import pandas as pd

def one_hot_encode(data):

    new_df = data
    columns_to_encode = ['type', 'drug', 'asypupilline', 'asybasis', 'asymenton', 'asyoccl', 'asyupmid', 'asylowmi', 'profile', 'lowerface', 'spacerelationship',  'sagittalrelationright', 'sagitalrelationleft', 'transversal']

    for column_name in columns_to_encode:
        nominal_encoded_column = pd.get_dummies(data[column_name], prefix=column_name)
        new_df.drop(column_name, axis=1, inplace=True)
        new_df = new_df.join(nominal_encoded_column)

    return new_df