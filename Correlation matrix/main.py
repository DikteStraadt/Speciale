import pandas as pd
import CorrelationMatrix as c
import RandomForest as r
import XGBoost as x

if __name__ == '__main__':

    data = pd.read_excel("C:/Users/User/Downloads/output.xlsx")
    print("Data imported")

    # Correlation matrix
    # columns_to_exclude_1 = ['Unnamed: 0', 'visitationdate', 'Aldervedafslut']
    # data1 = data.drop(columns=columns_to_exclude_1)
    # c.correlation_matrix(data1, "jia_corr_matrix_1.png")
    # print("Correlation matrix 1 plotted")

    # columns_to_exclude_2 = ['Unnamed: 0', 'visitationdate', 'Aldervedafslut', 'retrognathism', 'clicklaterotrusionright', 'clicklaterotrusionleft', 'clicklaterorightright', 'clicklaterorightleft', 'clicklateroleftright', 'clicklateroleftleft', 'micrognathism', 'scheme']
    # data2 = data.drop(columns=columns_to_exclude_2)
    # c.correlation_matrix(data2, "jia_corr_matrix_2.png")
    # print("Correlation matrix 2 plotted")

    # XGBoost
    columns_to_exclude_3 = ['involvementstatus', 'Unnamed: 0', 'visitationdate', 'Aldervedafslut', 'retrognathism', 'clicklaterotrusionright', 'clicklaterotrusionleft', 'clicklaterorightright', 'clicklaterorightleft', 'clicklateroleftright', 'clicklateroleftleft', 'micrognathism', 'scheme']
    data3 = data.drop(columns=columns_to_exclude_3)
    target = data['involvementstatus']
    #r.random_forest(data3, target)
    x.xgboost(data3, target)
    print("XGBoost completed")

    print("Done!")

