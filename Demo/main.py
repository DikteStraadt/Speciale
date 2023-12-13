import pickle

import numpy as np
import pandas as pd

if __name__ == '__main__':

    # Load model
    with open("test_model.pkl", 'rb') as file:
        model = pickle.load(file)

    # Get feature names
    feature_names = model.feature_names_in_
    feature_names = np.concatenate((feature_names, ['ID']))

    # Load data
    data = pd.read_excel("data.xlsx", "Sheet1")

    # Select feature names columns
    data = data[feature_names]

    visitation_1 = data[data['ID'] == '6076t_4']
    visitation_1 = visitation_1.drop('ID', axis=1)

    visitation_2 = data[data['ID'] == '8400a_4']
    visitation_2 = visitation_2.drop('ID', axis=1)

    visitation_3 = data[data['ID'] == '8843z_4']
    visitation_3 = visitation_3.drop('ID', axis=1)

    # Predict visitation
    predicted_involvement_status_1 = model.predict(visitation_1)
    print(predicted_involvement_status_1)

    predicted_involvement_status_2 = model.predict(visitation_2)
    print(predicted_involvement_status_2)

    predicted_involvement_status_3 = model.predict(visitation_3)
    print(predicted_involvement_status_3)

