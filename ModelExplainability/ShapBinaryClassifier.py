import pandas as pd
from shap import maskers
from IPython.display import display
import numpy as np
import IPython
import shap
import matplotlib.pyplot as plt
shap.initjs()

def forceplot_binary(clf, clf_pipe_name, index, X_train, y_train, explainer, shap_values):
    """
    Takes in a fitted classifier and the pipeline it comes from,
    the X training data, the train target, a shap explainer, and the
    shap values for the ground truth and predicted label
    :param clf: Pipeline where the fitted classifier is included
    :param clf_pipe_name: The string identifying the classifier step of pipeline
    :param index: The index of a specific observation
    :param X_train:  Train data from train-test split used to train classifier, where
                    column names correspond to feature names
    :param y_train: Subset of target used for training
    :param explainer: Fitted shap.TreeExplainer object
    :param shap_values: The array of shap values
    :return:
    """

    ## Store model prediction and ground truth
    pred = clf.named_steps[clf_pipe_name].predict([X_train.iloc[index, :]])
    true_label = y_train.iloc[index]


    ## Assessing accuracy of prediction
    if true_label == pred:
        accurate = 'Correct!'

    else:
        accurate = 'Incorrect'

    print('***'*12)
    print(f'Ground Truth Label: {true_label}')
    print()
    print(f'Model Prediction: {pred} -- {accurate}')
    print('***'*12)
    print()

    fig = shap.force_plot(explainer.expected_value, shap_values[index, :], X_train.iloc[index, :], matplotlib=True,)

    return fig
