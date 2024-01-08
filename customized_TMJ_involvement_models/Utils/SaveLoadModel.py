import os
import pickle
from datetime import datetime
from Utils import Report as r

def save_model(model, ml_type, f1):

    model_name = f"model-{ml_type}-({round(f1, 4)})-{datetime.now().strftime('%d-%m-%Y %H-%M-%S')}"
    path = f"Temp/{model_name}.pkl"

    with open(path, 'wb') as files:
        pickle.dump(model, files)

    print(f"Model saved as {model_name}")

def load_model(path):

    with open(path, 'rb') as file:
        model = pickle.load(file)

    return model

def clean_temp_folder():

    folder_path = 'Temp'
    file_list = os.listdir(folder_path)

    for file in file_list:

        file_path = os.path.join(folder_path, file)
        try:
            os.remove(file_path)
            print(f"Deleted: {file}")
        except Exception as e:
            print(f"Error deleting {file}: {e}")

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

    new_model_name = f"Results/{report['id']} model (ml={model_name}, f1 macro={round(metric, 4)}, lag features={report['lag features']}, number of lag feature={report['number of lag features']}, features={report['feature selection']} {report['timestamp end']}.pkl"

    for filename in file_list:
        if filename.startswith(f"model-{model_name}"):
            current_file_path = os.path.join(path, filename)
            try:
                os.rename(current_file_path, new_model_name)
                print(f"Renamed: {filename} to {new_model_name}")
            except Exception as e:
                print(f"Error renaming {filename}: {e}")

    return new_model_name