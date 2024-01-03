import numpy as np
import pandas as pd
from mapie.classification import MapieClassifier
from mapie.metrics import classification_coverage_score
from mapie.metrics import classification_mean_width_score
from Utils import Report as r

def conformalPrediction(model, featurenames, X_calib, y_calib, X_new, y_new):
    X_calib = X_calib[featurenames]
    X_new = X_new[featurenames]
    y_calib = y_calib.astype(int)
    y_new = y_new.astype(int)

    mapie_score = MapieClassifier(model, cv="prefit", method='score')
    mapie_score.fit(X_calib, y_calib)  # by default 20% of regularization data is used for regularization
    y_pred, y_set = mapie_score.predict(X_new, alpha=0.05, include_last_label="randomized")
    y_set = np.squeeze(y_set)
    cov = classification_coverage_score(y_new, y_set)
    setsize = classification_mean_width_score(y_set)
    print('Coverage: {:.2f}'.format(cov))
    r.write_to_report("Conformal converage", format(cov,'.2f'))
    print("Avg. set size: {:.2f}".format(setsize))
    r.write_to_report("Conformal Avg. set size", format(setsize,'.2f'))
    print(class_wise_performance(y_new, y_set, y_calib.unique()))
    r.write_to_report("Conformal classwise performance", class_wise_performance(y_new, y_set, y_calib.unique()).to_json())


"""
Function based on Introduction to Conformal Prediction With Python by Molnar
"""
def class_wise_performance(y_new, y_set, classes):
    df = pd.DataFrame()
    # Looping through all the classes
    for i in range(len(classes)):
        # Calculating the coverage and test size for the current class
        ynew = y_new.values[y_new.values == i]
        yscore = y_set[y_new.values == i]
        cov = classification_coverage_score(ynew, yscore)
        size = classification_mean_width_score(yscore)

        # Create new dataframe with calculated values
        temp_df = pd.DataFrame({
            "class": [classes[i]],
            "coverage": [cov],
            "avg. set size": [size]
        }, index=[i])

        # Concatenate the new dataframe with the existing one
        df = pd.concat([df, temp_df])

    return(df)