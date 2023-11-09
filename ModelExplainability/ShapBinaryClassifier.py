from shap import maskers
import shap
shap.initjs()


def waterfallPlot(X_train, X_test, model):

    background = maskers.Independent(X_train)

    explainer = shap.TreeExplainer(model, background)
    shap_values_bin = explainer(X_test)

    print(shap_values_bin.shape)

    # might have to be 1 instead
    # Will show a log odds of a positive prediction.
    # The SHAP values give the difference between the predicted
    # log odds and the average predicted log odds. Positive SHAP values
    # will increase the log odds.
    shap.plots.waterfall(shap_values_bin[0])


