import pickle
from datetime import datetime


def save_model(model, type):

    path = f"model-{type}-{datetime.now().strftime('%d-%m-%Y %H-%M-%S')}.pkl"

    with open(path, 'wb') as files:
        pickle.dump(model, files)

    print("Model saved")

def load_model(path):

    with open(path, 'rb') as file:
        model = pickle.load(file)

    return model