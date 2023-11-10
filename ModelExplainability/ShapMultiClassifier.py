import pandas as pd
from shap import maskers
import numpy as np
import IPython
import shap
import matplotlib.pyplot as plt
shap.initjs()



def multiclass_forceplot(clf, clf_pipe_name, index, X_train, y_train, explainer, multi_shap_vals, classes):
    pred = int(clf.named_steps[clf_pipe_name].predict([X_train.iloc[index, :]]))
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
        print('No TMJ involvement{0}')
        shap.force_plot(explainer.expected_value[0],
                                        multi_shap_vals[0][index],
                                        X_train.iloc[index,:], matplotlib=True, show=False)
        plt.savefig('force_plot_noTMJ.png')
        print()

        print('TMJ involvement{1}')
        shap.force_plot(explainer.expected_value[1],
                                        multi_shap_vals[1][index],
                                        X_train.iloc[index, :], matplotlib=True, show=False)
        plt.savefig('force_plot_TMJ.png')
        print()

        print('Obs category{2}')
        shap.force_plot(explainer.expected_value[2],
                                        multi_shap_vals[2][index],
                                        X_train.iloc[index, :],matplotlib=True, show=False)
        plt.savefig('force_plot_OBS.png')
    # only the class predicted by the model
    elif classes == 'pred':
        print(f'Predicted: Class {pred}')
        IPython.display(shap.force_plot(explainer.expected_value[pred],
                                multi_shap_vals[pred][index],
                                X_train.iloc[index, :]))

    # only the class
    elif classes == 'true':
        print(f'True: Class {pred}')
        IPython.display(shap.force_plot(explainer.expected_value[true_label],
                                        multi_shap_vals[true_label][index],
                                        X_train.iloc[index, :]))

    # Bot predicted and ground truth - if prediction is correct these plots will be identical
    elif classes == 'both':
        print(f'Predicted: Class {pred}')
        IPython.display(shap.force_plot(explainer.expected_value[pred],
                                        multi_shap_vals[pred][index],
                                        X_train.iloc[index, :]))
        print()

        print(f'True: Class {true_label}')
        IPython.display(shap.force_plot(explainer.expected_value[true_label],
                                        multi_shap_vals[true_label][index],
                                        X_train.iloc[index, :]))







"""
As dealing w. multiple classes, f(x) is given in terms of softmax.
Softmax values are converted to probabilities w. this function. 
"""
def softmax(x):
    """Computing softmax values for each sets of scores"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
