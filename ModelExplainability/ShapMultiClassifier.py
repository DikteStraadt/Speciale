import pandas as pd
from Tools.scripts.dutree import display
from shap import maskers
import numpy as np
import shap
shap.initjs()



def multiclass_forceplot(clf, clf_pipe_name, index, X_train, y_train, explainer, multi_shap_vals, classes):
    pred = int(clf.named_steps[clf_pipe_name].predict(X_train.iloc[index, :]))
    true_label = y_train.iloc[index]


    # Assessing accuracy of prediction
    if true_label == pred:
        accurate = 'Correct!'

    else:
        accurate = 'Incorrect'

    print(f'True label: {true_label}')
    print()
    print(f'Model prediction: [{pred}] -- {accurate}')
    print()
    print()

    # Determining which classes to show plots for
    # All classes
    if classes == 'all':
        ## Visualize the ith prediction's explanation for all classes
        display(shap.force_plot(explainer))




"""
As dealing w. multiple classes, f(x) is given in terms of softmax.
Softmax values are converted to probabilities w. this function. 
"""
def softmax(x):
    """Computing softmax values for each sets of scores"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
