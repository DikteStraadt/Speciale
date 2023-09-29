import pandas as pd

def import_data_sheet1():
    return pd.read_excel("C:/Users/User/Downloads/Master_Excel_Sep4.xlsx")

def import_data_sheet2():
    return pd.read_excel("C:/Users/User/Downloads/Master_Excel_Sep4.xlsx", "Sheet2")
