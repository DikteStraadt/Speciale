import pickle

import numpy as np
import pandas as pd

if __name__ == '__main__':

    # Define data
    feature_names = ['pain', 'painmove', 'morningstiffness', 'muscularpain', 'headache', 'openingfunction', 'neckpain', 'neckpalpation', 'laterpalp', 'postpalp', 'translation', 'masseter', 'temporalis', 'ptext', 'ptint', 'tempsen', 'sterno', 'stylomandibularligament', 'asymmetrymasseter', 'openingmm', 'opening', 'protrusionmm', 'protrusion', 'laterotrusionrightmm', 'laterotrusionleftmm', 'overjet', 'overbite', 'openbite', 'asybasis', 'asypupilline', 'Biologics', 'Conventional', 'Eye medicine', 'NSAID', 'Corticosteroids', 'profile', 'lowerface', 'asymenton', 'asyoccl', 'asylowmi', 'sagittalrelation']

    #data = np.array([[]])
    #data = np.array([[0, 0, 0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0.082637859,	-0.063785877, 1.190879592, -0.13850578,	-0.170677257, -0.448022499,	-0.597958091, -0.072312883,	-0.152292754, -0.002298575,	-0.01179835, 0,	0, 0, 1, 0,	0.022425484, -0.037518222, 0.03799998, 0.023771171,	-0.005597177, 0.010628852]])  # 0
    #data = np.array([[0, 0, 0, 1, 2, 1, 1, 0, 1, 0, 1, 2, 1, 0, 0, 0, 0, 0, 0, -0.921400051, -0.063785877, -1.815357723, -0.13850578, -2.151531469, -2.071441814, -1.622347427, -0.296327696, -0.152292754, -0.002298575, -0.01179835, 1, 1, 0, 1, 1, 0.022425484, -0.037518222, 0.03799998, 0.023771171, -0.005597177, 0.010628852]])  # 1
    #data = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.623273657, -0.063785877, 0.990931486, -0.13850578, 0.395281089, 0.363687158, 1.450820582, -0.072312883, -0.152292754, -0.002298575, -0.01179835, 1, 0, 0, 1, 0, 0.022425484, -0.037518222, 0.03799998, 0.023771171, -0.005597177, 0.010628852]])  # 0
    #data = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0.685060605, -0.063785877, -0.612161225, -0.13850578, 0.395281089, -0.177452614, -0.085763423, -0.072312883, -0.152292754, -0.002298575, -0.01179835, 0, 1, 0, 1, 0, 0.022425484, -0.037518222, 0.03799998, 0.023771171, -0.005597177, 0.010628852]]) # 1
    data = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, -2.697774814, -0.063785877, 0.366532137, -0.13850578, 0.112301916, -0.177452614, -1.110152759, -0.18432029, -0.152292754, -0.002298575, -0.01179835, 0, 0, 0, 1, 0, 0.014077809, -0.037518222, 0.03799998, 0.023771171, -0.005597177, 0.010628852]])  # 1

    visitation = pd.DataFrame(data, columns=feature_names)

    # Load model
    with open("test_model.pkl", 'rb') as file:
        model = pickle.load(file)

    # Predict visitation
    predicted_involvement_status = model.predict(visitation)
    print(predicted_involvement_status)

