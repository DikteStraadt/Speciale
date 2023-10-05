import pandas as pd
from sklearn.datasets import load_wine
from HelperFunction import data_prep, train_test, sfs_feature_selection, xgboostClassification, makeClassification
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings("ignore")


wine = load_wine()

fields_to_drop = []

string_columns = []
numeric_columns = ['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']

# Train and Test
WOE_encoding = False
target = 'target'
test = 0.3
shuffle_flag = True
stratify_flag = True


if __name__ == '__main__':
    wine_df=pd.DataFrame(data=np.c_[wine['data'],wine['target']],columns=wine['feature_names']+['target'])
    print(wine_df.head())

    # Create heatmap  - Seems like total_phenols & flavanoids could be correlated (0.86)
    plt.subplots(figsize=(15, 9))
    cor = wine_df.drop(['target'], axis=1).corr()
    sns.heatmap(cor, annot=True, linewidths=.5)
    plt.show()

    # Create Train and Test Data
    X_train, X_test, Y_train, Y_test = train_test_split(wine.data, wine.target, train_size=0.80, stratify=wine.target, random_state=101)
    print("*********** First fitting without feature selection ***************")
    makeClassification(wine_df, "XGBoost", X_train, X_test, Y_train, Y_test, False)

    # Perform feature selection - sfsFlag indicates that feature selection should be done before fitting model
    makeClassification(wine_df, "XGBoost", X_train, X_test, Y_train, Y_test, True)








