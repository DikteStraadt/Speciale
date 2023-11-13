import pandas as pd
from shap import maskers
from IPython.display import display
import numpy as np
import IPython
import shap
import matplotlib.pyplot as plt
shap.initjs()



def multiclass_forceplot(clf, clf_pipe_name, index, X_train, y_train, explainer, multi_shap_vals, classes):
    pred = int(clf.named_steps[clf_pipe_name].predict([X_train.iloc[index, :]]))
    true_label = int(y_train.iloc[index])

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
                                        X_train.iloc[index,:], show=False, matplotlib=True, text_rotation=5)
        plt.title('No TMJ involvement{0}')
        plt.tight_layout()
        plt.gcf().set_size_inches(20, 7)
        fig1 = plt.gcf()
        plt.show()
        fig1.savefig(f'plots/force_plot_noTMJ_observation{index}.png')
        print()

        print('TMJ involvement{1}')
        shap.force_plot(explainer.expected_value[1],
                                        multi_shap_vals[1][index],
                                        X_train.iloc[index, :], show=False, matplotlib=True, text_rotation=5)
        plt.title('TMJ involvement{1}')
        plt.tight_layout()
        plt.gcf().set_size_inches(20, 7)
        fig2 = plt.gcf()
        plt.show()
        fig2.savefig(f'plots/force_plot_TMJ_observation{index}.png')
        print()

        print('Obs category{2}')
        shap.force_plot(explainer.expected_value[2],
                                        multi_shap_vals[2][index],
                                        X_train.iloc[index, :], show=False, matplotlib=True, text_rotation=5)
        plt.title('Obs Category{2}')
        plt.tight_layout()
        plt.gcf().set_size_inches(20, 7)
        fig3 = plt.gcf()
        plt.show()
        fig3.savefig(f'plots/force_plot_Obscatergory_observation{index}.png')


    # only the class predicted by the model
    elif classes == 'pred':
        print(f'Predicted: Class {pred}')
        shap.force_plot(explainer.expected_value[pred],
                                multi_shap_vals[pred][index],
                                X_train.iloc[index, :], show=False, matplotlib=True, text_rotation=5)
        predText = classConversion(pred)
        plt.title(f'{predText}')
        plt.tight_layout()
        plt.gcf().set_size_inches(20, 7)
        fig = plt.gcf()
        plt.show()
        fig.savefig(f'plots/force_plot_{predText}_observation{index}.png')

    # only the class
    elif classes == 'true':
        print(f'True: Class {true_label}')
        shap.force_plot(explainer.expected_value[true_label],
                                        multi_shap_vals[true_label][index],
                                        X_train.iloc[index, :], show=False, matplotlib=True, text_rotation=5)
        trueText = classConversion(true_label)
        plt.title(f'{trueText}')
        plt.tight_layout()
        plt.gcf().set_size_inches(20, 7)
        fig = plt.gcf()
        plt.show()
        fig.savefig(f'plots/force_plot_{trueText}_observation{index}.png')

    # Bot predicted and ground truth - if prediction is correct these plots will be identical
    elif classes == 'both':
        print(f'Predicted: Class {pred}')
        shap.force_plot(explainer.expected_value[pred],
                                        multi_shap_vals[pred][index],
                                        X_train.iloc[index, :],show=False, matplotlib=True, text_rotation=5)
        predText = classConversion(pred)
        plt.title(f'{predText}')
        plt.tight_layout()
        plt.gcf().set_size_inches(20, 7)
        fig = plt.gcf()
        plt.show()
        fig.savefig(f'plots/force_plot_{predText}_observation{index}.png')
        print()

        print(f'True: Class {true_label}')
        shap.force_plot(explainer.expected_value[true_label],
                                        multi_shap_vals[true_label][index],
                                        X_train.iloc[index, :],show=False, matplotlib=True, text_rotation=5)
        trueText = classConversion(true_label)
        plt.title(f'{trueText}')
        plt.tight_layout()
        plt.gcf().set_size_inches(20, 7)
        fig = plt.gcf()
        plt.show()
        fig.savefig(f'plots/force_plot_{trueText}_observation{index}.png')




def waterfallplot(explainer, X_train, className, observationIndex):
    """
    :param explainer: Tree explainer based on model
    :param X_train: Training data
    :param className: The class we want to make a plot for. Possible values: {No TMJ involvement,TMJ involvement, Obs}
    :param observationIndex: The index for the observation
    :return:
    """
    shap_values = explainer(X_train)

    print(shap_values.shape)

    class_index_to_plot = classMapping(className)

    shap_values_to_plot = shap_values[observationIndex, : , class_index_to_plot]
    shap.plots.waterfall(shap_values_to_plot, max_display=20, show=False)
    plt.title(className)
    plt.tight_layout()
    plt.show()



def classMapping(classString):
    if classString == 'No TMJ involvement':
        return 0
    if classString == 'TMJ involvement':
        return 1
    else:
        return 2


"""
New set of shap values are created by looping over original 
SHAP values and selecting the set that corresponds to the prediction
for that instance. Now getting one set of SHAP values per instance
"""
def aggregatedplot(X, explainer, plotType, classString):
    """
    :param X: Dataset
    :param explainer: Tree explainer based on model
    :param plotType: Type of plot {bar, beeswarm}
    :param classString: The class to make a plot of. Possible values: {No TMJ involvement,TMJ involvement, Obs}
    :return:
    """
    classIndex = classMapping(classString)

    shap_values = explainer(X)
    if plotType == 'beeswarm':
        shap.plots.beeswarm(shap_values[:,:,classIndex], show=False, max_display=20)
        plt.title(classString)
        plt.gcf().set_size_inches(20, 13)
        plt.tight_layout()
        plt.show()

    elif plotType == 'bar':
        shap.plots.bar(shap_values[:,:,classIndex],show=False, max_display=20)
        plt.title(classString)
        plt.gcf().set_size_inches(20, 13)
        plt.tight_layout()
        plt.show()



def classConversion(classId):
    if classId == 0:
        return 'No TMJ involvement'
    if classId == 1:
        return 'TMJ involvement'
    else:
        return 'Obs category'


"""
As dealing w. multiple classes, f(x) is given in terms of softmax.
Softmax values are converted to probabilities w. this function. 
"""
def softmax(x):
    """Computing softmax values for each sets of scores"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
