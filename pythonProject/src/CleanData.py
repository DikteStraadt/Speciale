import numpy as np
from dateutil import relativedelta
from datetime import datetime
import operator

def remove_old_patients(data):

    indexes = []

    for i in range(len(data)):
        date_birth = datetime.strptime(data['birth'][i], "%d-%b-%y")
        date_visitation = datetime.strptime(data['first_visitation'][i], "%d-%b-%y")
        difference = relativedelta.relativedelta(date_visitation, date_birth)

        if difference.years < 0 or difference.years > 100:
            indexes.append(i)

    for i in sorted(indexes, reverse=True):
        data = data.drop(i)

    return data.reset_index(drop=True)

def remove_visitations(visitations_3D):

    visitation_number = 1

    # Visitation 0
    visitation_0 = visitations_3D[0]
    column_2 = visitation_0.iloc[:, 2]
    indexes_to_be_removed = []

    for i, value in enumerate(column_2):
        if np.isnan(value):
            indexes_to_be_removed.append(i)

    for j in sorted(indexes_to_be_removed, reverse=True):
        visitation_0 = visitation_0.drop(j)

    visitations_3D[0] = visitation_0.reset_index(drop=True)

    # Visitation 1-16
    for visitation_2D in visitations_3D[visitation_number:17]:

        column_1 = visitation_2D.iloc[:, 1]
        indexes_to_be_removed = []

        for i, value in enumerate(column_1):
            if np.isnan(value):
                indexes_to_be_removed.append(i)

        for j in sorted(indexes_to_be_removed, reverse=True):
            visitation_2D = visitation_2D.drop(j)

        visitations_3D[visitation_number] = visitation_2D.reset_index(drop=True)
        visitation_number = visitation_number + 1

    return visitations_3D