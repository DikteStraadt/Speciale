import pandas as pd
from FeatureEngineering import drugMapping as dm

class DrugTransformer:
    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):
        new_data = transform_drug_categories(data)
        return new_data

def transform_drug_categories(data):
    data["drug"] = data["drug"].astype(int)
    for new_category, original_category in dm.DrugMapper.drug_dict.items():
        data[new_category] = data["drug"].apply(
            lambda x: 1 if x in original_category else 0
        )

    # drop the original column representing drug combinations if needed
    data.drop("drug", axis=1, inplace=True)

    return data
