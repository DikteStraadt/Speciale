#from sklearn.datasets import make_friedman1
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

def random_forest(data, target):

    data.head()

    X_train, X_test, Y_train, Y_test = train_test_split(data, target, train_size=0.9, stratify=target, random_state=123)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    rforest_classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
    rforest_classifier.fit(X_train, Y_train)
    Y_preds = rforest_classifier.predict(X_test)

    print('Training Coefficient of R^2 : %.3f' % rforest_classifier.score(X_train, Y_train))
    print('Test Coefficient of R^2 : %.3f' % rforest_classifier.score(X_test, Y_test))

    print("\nConfusion Matrix : ")
    print(confusion_matrix(Y_test, Y_preds))

    print("\nClassification Report : ")
    print(classification_report(Y_test, Y_preds))

    # estimator = SVR(kernel="linear")
    # selector = RFECV(estimator, step=1, cv=5)
    # selector = selector.fit(X_train, Y_train)