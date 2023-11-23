import pandas as pd
from functools import lru_cache

@lru_cache(maxsize=4)
def import_data(path, sheet):
    return pd.read_excel(path, sheet)

def export_data(data, path):
    data.to_excel(path)