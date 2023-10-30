import pickle

def save_model(model):

    with open('model_pkl', 'wb') as files:
        pickle.dump(model, files)

    print("Model saved")

def load_model():

    with open('model_pkl', 'rb') as file:
        lr = pickle.load(file)

    return lr