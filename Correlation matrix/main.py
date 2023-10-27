import pandas as pd
import CorrelationMatrix as c
import RandomForest as r
import XGBoost as x
import CatBoost as cat
import EncodeData as e
import PCA as p
import LowVarianceThreshold as l
import FeatureClustering as f
import NonLinearCorMatrix as n

if __name__ == '__main__':

    data = pd.read_excel("C:/Users/User/Downloads/output.xlsx")
    # data = pd.read_csv("C:/Users/User/Desktop/CorMatrix/Book1.csv", sep=';')
    print("Data imported")

    # n.non_linear_cor_matrix(data)

    # f.feature_clustering()
    # print("Feature clustering")

    # l.low_variance_threshold(data)
    # print("Low variance threshold")

    p.PCA_tmj(data)
    #print("PCA")

    # Encode data
    # data = e.one_hot_encode(data)
    # print("Data encoded")

    # Correlation matrix
    # columns_to_exclude_ = ['studyid', 'Unnamed: 0', 'visitationdate', 'Aldervedafslut', 'scheme']
    # data = data.drop(columns=columns_to_exclude_)
    # c.correlation_matrix(data, "jia_corr_matrix.png")
    # print("Correlation matrix plotted")

    #columns_to_exclude_3 = ['type', 'studyid', 'involvementstatus', 'Unnamed: 0', 'visitationdate', 'Aldervedafslut', 'scheme']
    #data3 = data.drop(columns=columns_to_exclude_3)
    #target = data['involvementstatus']
    #r.random_forest(data3, target)
    #x.xgboost(data3, target)

    #cat.catBoost()



    print("Done!")

