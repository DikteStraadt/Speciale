import os
import pickle
from datetime import datetime
from Utils import Report as r

def save_model(model, ml_type):

    path = f"Temp/model-{ml_type}-{datetime.now().strftime('%d-%m-%Y %H-%M-%S')}.pkl"

    with open(path, 'wb') as files:
        pickle.dump(model, files)

    print("Model saved")

def load_model(path):

    with open(path, 'rb') as file:
        model = pickle.load(file)

    return model

def remove_models():

    folder_path = 'Temp'
    file_list = os.listdir(folder_path)

    for model_name in file_list:

        file_path = os.path.join(folder_path, model_name)
        try:
            os.remove(file_path)
            print(f"Deleted: {model_name}")
        except Exception as e:
            print(f"Error deleting {model_name}: {e}")

def rename_model(model_name, report):

    path = "Temp"
    file_list = os.listdir(path)

    if model_name == "random forest":
        metric = report['(random forest) f1 macro']
    elif model_name == "xgboost":
        metric = report['(xgboost) f1 macro']
    elif model_name == "catboost":
        metric = report['(catboost) f1 macro']
    else:
        metric = "ERROR"

    new_model_name = f"Results/{report['id']} model (ml={model_name}, f1 macro={round(metric, 4)}, categories={report['n_categories']}, timeliness={report['timeliness']}, encoding={report['encoding']}, features={report['feature selection']} {report['timestamp end']}.pkl"

    for filename in file_list:
        if filename.startswith(f"model-{model_name}"):
            current_file_path = os.path.join(path, filename)
            try:
                os.rename(current_file_path, new_model_name)
                print(f"Renamed: {filename} to {new_model_name}")
            except Exception as e:
                print(f"Error renaming {filename}: {e}")

    return new_model_name