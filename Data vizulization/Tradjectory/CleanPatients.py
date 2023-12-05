import numpy as np
from Tradjectory import Utils as u

def remove_visitations(patients):

    p = 0

    for patient in patients:

        indexes_to_be_removed = []
        i = 0

        for visitation in patient:

            if visitation['visitationdate'] == "#NULL!" or np.isnan(visitation['involvementstatus']):
                indexes_to_be_removed.append(i)

            i = i + 1

        for j in sorted(indexes_to_be_removed, reverse=True):

            patients[p].pop(j)

        p = p + 1

    return patients