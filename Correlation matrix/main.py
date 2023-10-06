import pandas as pd
import CorrelationMatrix as c
import RandomForest as r
import XGBoost as x
import EncodeData as e

if __name__ == '__main__':

    data = pd.read_excel("C:/Users/User/Downloads/output.xlsx")
    print("Data imported")

    # Encode data
    data = e.one_hot_encode(data)
    print("Data encoded")

    # Correlation matrix
    columns_to_exclude_ = ['studyid', 'Unnamed: 0', 'visitationdate', 'Aldervedafslut', 'scheme']
    data = data.drop(columns=columns_to_exclude_)
    c.correlation_matrix(data, "jia_corr_matrix.png")
    print("Correlation matrix plotted")

    #columns_to_exclude_3 = ['type', 'studyid', 'involvementstatus', 'Unnamed: 0', 'visitationdate', 'Aldervedafslut', 'scheme']
    #data3 = data.drop(columns=columns_to_exclude_3)
    #target = data['involvementstatus']
    #r.random_forest(data3, target)
    #x.xgboost(data3, target)

    print("Done!")

