from matplotlib import pyplot
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from Utils import Report as r
from Utils import SaveLoadModel as s

def evaluation(ml_type, model, X_test, y_test):

    # importance = model.best_estimator_.named_steps[ml_type].feature_importances_
    # category_names = X_train.columns
    # pyplot.figure(figsize=(8, 10))
    # pyplot.bar(category_names, importance)
    # pyplot.xlabel('Features')
    # pyplot.ylabel('Importance')
    # pyplot.xticks(rotation=45, ha='right')
    # pyplot.show()

    y_preds = model.predict(X_test)

    print("\nConfusion Matrix: ")
    print(confusion_matrix(y_test, y_preds))

    print("\nClassification Report: ")
    print(classification_report(y_test, y_preds))

    f1_macro = f1_score(y_test, y_preds, average='macro')
    f1_micro = f1_score(y_test, y_preds, average='micro')
    f1_weighted = f1_score(y_test, y_preds, average='weighted')

    r.write_to_report(f"({ml_type}) confusion matrix", confusion_matrix(y_test, y_preds).tolist())
    r.write_to_report(f"({ml_type}) classification report", classification_report(y_test, y_preds))
    r.write_to_report(f"({ml_type}) best model", str(model.best_estimator_))
    r.write_to_report(f"({ml_type}) best parameters", str(model.best_params_))
    r.write_to_report(f"({ml_type}) f1 macro", f1_macro)

    s.save_model(model, ml_type, f1_macro)

def find_best_model():

    report = r.read_report()
    models = [report['(random forest) f1 macro'], report['(xgboost) f1 macro'], report['(catboost) f1 macro']]
    index = models.index(max(models))

    if index == 0:
        return "random forest"
    elif index == 1:
        return "xgboost"
    elif index == 2:
        return "catboost"
    else:
        return "ERROR"