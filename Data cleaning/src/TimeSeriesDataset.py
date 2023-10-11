import pandas as pd
import numpy as np

def add_sliding_timewindow(patients):
    i = 0
    for patient in patients:
        j = 0
        for visitation in patient:
            if j > 0:
                p = j - 1
                value = {'involvementstatus_previous': patients[i][p]['involvementstatus']}
                previous_involvement_status = pd.Series(value)
                result_series = pd.concat([patients[i][j], previous_involvement_status])
                patients[i][j] = result_series

            j = j + 1

        i = i + 1

    return patients