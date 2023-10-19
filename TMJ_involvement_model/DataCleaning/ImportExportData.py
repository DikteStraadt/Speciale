import pandas as pd

def import_data(path, sheet):
    return pd.read_excel(path, sheet)

def export_data(data, path):
    data.to_excel(path)