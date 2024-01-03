import pandas as pd

class ReverseEmbeddingTransformer:
    def __init__(self, embeddingList):
        self.embeddingList = embeddingList

    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):
        for idx, x in enumerate(self.embeddingList):
            dict = {row[0]: row[2] for _, row in pd.read_csv(f"Embeddings/{x}.csv").iterrows()}

            # Merging data
            for key, value in dict.items():
                data[x].replace(value, key, inplace=True)

        print('Data inverse transformed back from embedded to original format!')

        return data