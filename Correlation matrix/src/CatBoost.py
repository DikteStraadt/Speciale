from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from sklearn.datasets import load_iris

def catBoost():

    # Load the Iris dataset
    iris_df = load_iris()
    X = iris_df.data
    y = iris_df.target

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.1, random_state=123)

    # Create and train the CatBoostClassifier
    model_params = {
        'iterations': 100,
        'depth': 6,
        'learning_rate': 0.1,
        'loss_function': 'MultiClass',
        # 'train_dir': 'crossentropy',
        # 'allow_writing_files': False,
        # 'random_seed': 123,
        'verbose': False
    }

    model = CatBoostClassifier(**model_params)

    model.fit(X_train, Y_train)

    # Make predictions on the test set
    Y_pred = model.predict(X_test)

    print('Training Coefficient of R^2 : %.3f' % model.score(X_train, Y_train))
    print('Test Coefficient of R^2 : %.3f' % model.score(X_test, Y_test))

    print("\nConfusion Matrix : ")
    print(confusion_matrix(Y_test, Y_pred))

    print("\nClassification Report : ")
    print(classification_report(Y_test, Y_pred))
