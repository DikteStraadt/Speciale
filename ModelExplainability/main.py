import pandas as pd
import numpy as np
import shap
from Utils import LoadFile as lf
from sklearn.model_selection import train_test_split
from Utils import ImportExportData as d
from Utils import Configuration as c
from ShapMultiClassifier import multiclass_forceplot, waterfallplot
from ShapBinaryClassifier import forceplot_binary

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ##################### IMPORT CONFIGS #####################

    configurations = c.get_configurations()
    columns_to_exclude = ['sex', 'type', 'studyid', 'Unnamed: 0', 'visitationdate', 'tractionright', 'tractionleft']
    data = d.import_data(f"output_{configurations[0]['n_categories']}_cat.xlsx", "Sheet1")
    data = data.drop(columns=columns_to_exclude)
    featurename_cols = ['painmoveright', 'openingfunction']
    y = data['involvementstatus']
    X = data[featurename_cols]



    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_df = pd.DataFrame(X_train, columns=featurename_cols)


    #Calculate SHAP values for multi class model
    multi_model = lf.load_model("rf_test.pkl")
    multi_est = multi_model.best_estimator_
    multi_model = multi_est.named_steps['randomforest']  #randomforest
    multi_explainer = shap.TreeExplainer(multi_model)

    multi_shap_values = multi_explainer.shap_values(X_train)
    print(len(multi_shap_values))   # prints out '3'

    waterfallplot(multi_explainer, X_train, "TMJ involvement", 7)

    #multiclass_forceplot(multi_est, 'randomforest', 7, X_train, y_train, multi_explainer, multi_shap_values, classes='all')
    #forceplot_binary(multi_est, 'randomforest', 7, X_train, y_train, multi_explainer, multi_shap_values)



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
