import pandas as pd

def doOneHotEncoding(columns_to_encode, data):
    new_df = data
    for column_name in columns_to_encode:
        nominal_encoded_column = pd.get_dummies(data[column_name].astype(str), prefix=column_name)
        new_df.drop(column_name, axis=1, inplace=True)
        new_df = new_df.join(nominal_encoded_column)

    return new_df