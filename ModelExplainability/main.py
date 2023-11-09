import pandas as pd
import numpy as np
import shap
from Utils import LoadFile as lf
from sklearn.model_selection import train_test_split

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ##################### IMPORT CONFIGS #####################

    configurations = c.get_configurations()
    data = d.import_data(f"output_{configurations[0]['n_categories']}_cat.xlsx", "Sheet1")
    y = data['involvementstatus']
    X = data.drop('involvementstatus', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    #Calculate SHAP values for multi class model
    multi_model = lf.load_model("pathpathpath")
    multi_explainer = shap.TreeExplainer(multi_model)

    multi_shap_values = multi_explainer.shap_values(X_train)



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
