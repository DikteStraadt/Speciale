import pandas as pd
import numpy as np
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.pipeline import make_pipeline
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
import category_encoders as ce
from sklearn.model_selection import StratifiedKFold

import xgboost as xgb
from xgboost import XGBClassifier
from catboost import CatBoost
from catboost.utils import eval_metric
import seaborn as sns
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Sequential Feature Selector
sfs_feature = 10
sfs_cv = 5
sfs_scoring = 'f1_micro'

def data_prep(data, fields_to_drop, string_columns, numeric_columns):
    try:
        data_v2 = data.copy()
        # Drop unwanted fields
        if len(fields_to_drop)>0:
            data_v2 = data_v2.drop(fields_to_drop, axis = 1)
            print('Dropped {} Fields'.format(len(fields_to_drop)))

        else:
            print('No Fields to Drop')

        # Convert String Columns
        if len(string_columns)>0:
            data_v2[string_columns] = data_v2[string_columns].astype(str)
            print('{} Fields converted to string'.format(len(string_columns)))

        else:
            print('No Fields to Convert to String')


        # Convert numeric columns
        if len(numeric_columns)>0:
            data_v2[numeric_columns] = data_v2[string_columns].astype(float)
            print('{} Fields converted to string'.format(len(numeric_columns)))

        else:
            print('No Fields to Convert to Numeric')

    except:
        print('Error: Reload data and check inputs')

    return data_v2


# Don't think we are going to use this type of encoding, but some is needed for RF/XGB
def WOE(train_features, test_features, train_target, test_target):
    WOE_columns = train_features.select_dtypes(exclude= [np.number]).columns

    woe_encoder = ce.WOEEncoder(cols = WOE_columns)

    WOE_encoded_train =woe_encoder.fit_transform(train_features[WOE_columns], train_target).add_suffix('_woe')
    train_features = pd.concat([train_features, WOE_encoded_train], axis=1)

    WOE_encoded_test = woe_encoder.transform(test_features[WOE_columns], test_target).add_suffix('_woe')
    test_features = pd.concat([test_features, WOE_encoded_test], axis=1)

    train_features_v2 = train_features.drop(WOE_columns, axis=1)
    test_features_v2 = test_features.drop(WOE_columns, axis=1)

    return train_features_v2, test_features_v2

def train_test(data, test, shuffle_flag, stratify_flag, WOE_encoding):
    #Define X and Y
    data.target = data.target.astype('int64').astype('category')
    Target = data.target
    features = data.drop(['target'], axis=1)
    print('Target Distribution')
    print(data['target'].value_counts())

    # Treating Categorial Features and Train/Test Split
    if WOE_encoding == True:
        # Train and Test Split
        if stratify_flag == True:
            train_features, test_features, train_target, test_target = train_test_split(features, Target,
                                                                                        test_size=test,
                                                                                        shuffle=shuffle_flag,
                                                                                        stratify=Target)
        else:
            train_features, test_features, train_target, test_target = train_test_split(features, Target, test_size=test,
                                                                                    shuffle=shuffle_flag)

        train_features, test_features = WOE(train_features, test_features, train_target, test_target)


    else:
        #features = pd.get_dummies(features, drop_first=True)

        # Train and Test Split

        if stratify_flag == True:
            train_features, test_features, train_target, test_target = train_test_split(features, Target, test_size = test, shuffle = shuffle_flag, stratify = Target)

        else:
            train_features, test_features, train_target, test_target = train_test_split(features, Target,
                                                                                        test_size=test,
                                                                                        shuffle=shuffle_flag)
    print('Training Features Shape:', train_features.shape)
    print('Training Labels Shape: ', train_target.shape)

    print('Testing Features Shape: ', test_features.shape)
    print('Testing Labels Shape: ', test_target.shape)

    train_features = train_features.astype(float)
    train_target = train_target.astype(float)
    test_features = test_features.astype(float)
    test_target = test_target.astype(float)

    return train_features, test_features, train_target, test_target, features, Target

# Sequential Feature Selector
def sfs_feature_selection(estimatorClass, data, train_features, train_target, sfs_feature, sfs_cv, sfs_scoring):

    # Input feature data
    # estimatorClass - the type of estimator (XGBoost, CatBoost, etc)
    # data - Input feature data
    # train_target - Target variable training data
    # sfs_feature - no. of features to select
    # sfs_direction - forward and backward selection
    # sfs_cv - cross-validation splitting strategy
    # sfs_scoring - CV performance scoring metric

    cv = StratifiedKFold(n_splits=sfs_cv, random_state=101, shuffle=True)

    sfs1 = SFS(estimator=estimatorClass,
               k_features=(3,10),
               forward=True,
               floating=False,
               scoring='f1_micro',
               cv=cv)
    pipe = make_pipeline(StandardScaler(), sfs1)
    pipe.fit(train_features, train_target)

    print('best combination (ACC: %.3f): %s\n ' % (sfs1.k_score_, sfs1.k_feature_idx_))
    #print('all subsets:\n', sfs1.subsets_)
    print(pd.DataFrame.from_dict(sfs1.get_metric_dict()).T)
    plot_sfs(sfs1.get_metric_dict(), kind='std_err')
    plt.grid()
    plt.show()

    features = []
    sfs_df = pd.DataFrame(columns=['Feature'])

    for x in sfs1.k_feature_idx_:
        print(data.columns[x])
        features.append(data.columns[x])

    # select features based on SFS
    sfs_df['Feature'] = features
    sfs_train_features = sfs_df['Feature'].tolist()


    return sfs_train_features , sfs1


def makeClassification(data, classification_estimator, X_train, X_test, Y_train, Y_test, sfsFlag):
    if classification_estimator == "XGBoost":
        if sfsFlag == True:
            print("Performing feature selection")
            xgb = XGBClassifier(n_jobs=-1, random_state=101)
            X_train_sfs, sfs1 = sfs_feature_selection(xgb,data, X_train, Y_train, sfs_feature, sfs_cv, sfs_scoring)
            newData = data.loc[:, X_train_sfs].copy()
            X_train2, X_test2, Y_train2, Y_test2 = train_test_split(newData, data['target'], train_size=0.80,
                                                                    stratify=data['target'],
                                                                    random_state=101)
            xgboostClassification(X_train2, X_test2, Y_train2, Y_test2)
            return
        # Link to the function parameters - https://xgboost.readthedocs.io/en/stable/parameter.html
        xgboostClassification(X_train, X_test, Y_train, Y_test)

    elif classification_estimator == "CatBoost":
        # call CatBoost classifier
        catboostClassification()


    if classification_estimator == "RandomForest":
        # call Random Forest classifier
        print()



def xgboostClassification(X_train, X_test, Y_train, Y_test):
    print("Train/Test Sizes : ", X_train.shape, X_test.shape, Y_train.shape, Y_test.shape, "\n")


    dmat_train = xgb.DMatrix(X_train, Y_train)
    dmat_test = xgb.DMatrix(X_test, Y_test)

    param = {
        'max_depth': 5,  # the maximum depth of each tree
        'eta': 1, # the training step for each iteration
        'objective': 'multi:softmax', # error evaluation for multiclass training
        'num_class': 3
    } # the number of classes that exists in this dataset

    booster = xgb.train(param, dmat_train, evals=[(dmat_train, "train"), (dmat_test, "test")])

    print("\nTrain RMSE : ", booster.eval(dmat_train))
    print("Test RMSE : ", booster.eval(dmat_test))

    print("\nTest Accuracy : %.2f" %accuracy_score(Y_test, booster.predict(data=dmat_test)))
    print("Train Accuracy : %.2f" %accuracy_score(Y_train, booster.predict(data=dmat_train)))

    print("\nConfusion Matrix : ")
    print(confusion_matrix(Y_test, booster.predict(data=dmat_test)))

    print("\nClassification Report : ")
    print(classification_report(Y_test, booster.predict(data=dmat_test)))


def catboostClassification(feature, X_train, X_test, Y_train, Y_test):
    print("Train/Test sizes : ", X_train.shape, X_test.shape, Y_train.shape, Y_test.shape, "\n")

    booster = CatBoost(params={'iterations': 100, 'verbose':10, 'loss_function': 'MultiClass', 'classes_count':3 })
    booster.fit(X_train, Y_train, eval_set=(X_test, Y_test))
    booster.set_feature_names(feature)

    test_preds = booster.predict(X_test, prediction_type="Class").flatten()
    train_preds = booster.predict(X_train, prediction_type="Class").flatten()

    print("\nTest Accuracy : %.2f"%eval_metric(Y_test, test_preds, "Accuracy")[0])
    print("Train Accuracy : %.2f"%eval_metric(Y_train, train_preds, "Accuracy")[0])

    booster.predict(X_test, prediction_type="Probability")[:5]


