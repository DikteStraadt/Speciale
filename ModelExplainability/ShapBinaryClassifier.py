import pandas as pd
from shap import maskers
from IPython.display import display
import numpy as np
import IPython
import shap
import matplotlib.pyplot as plt
shap.initjs()


# forceplot_binary(multi_est, 'randomforest', 7, X_train, y_train, multi_explainer, multi_shap_values)
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

    shap.force_plot(explainer.expected_value, shap_values[index, :], X_train.iloc[index, :], matplotlib=True, show=False, text_rotation=5)
    plt.tight_layout()
    plt.gcf().set_size_inches(20,7)
    fig = plt.gcf()
    plt.show()
    fig.savefig(f'plots/force_plot_binary_{index}.png')

def waterfallplotbinary(explainer, X_train,  observationIndex):
    shap_values = explainer(X_train)
    print(shap_values.shape)

    shap.plots.waterfall(shap_values[observationIndex], max_display=20, show=False)
    plt.tight_layout()
    fig = plt.gcf()
    plt.show()
    fig.savefig(f'plots/waterfallplotbinary_observation{observationIndex}.png')





def aggregatedplotbinary(explainer, X_train, plotType):

    shap_values = explainer(X_train)

    if plotType == 'beeswarm':
        shap.plots.beeswarm(shap_values, show=False, max_display=20)
        plt.gcf().set_size_inches(20, 13)
        plt.tight_layout()
        plt.show()


    if plotType == 'bar':
        shap.plots.bar(shap_values, show=False, max_display=20)
        plt.gcf().set_size_inches(20, 13)
        plt.tight_layout()
        plt.show()



