import numpy as np
import Utils as u

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

def convert_visitation_status(patients):

    i = 0

    for patient in patients:

        j = 0

        for visitation in patient:

            new_involvement_status = u.transform_status(visitation['involvementstatus'])
            patients[i][j]['involvementstatus'] = new_involvement_status

            j = j + 1

        i = i + 1

    return patients

def insert_zeros(patients):

    indexes_to_exclude = ['studyid', 'type', 'sex', 'visitationstatus', 'visitationdate']

    for patient in patients:
        for visitation in patient:
            for index_label in visitation.index:
                if index_label not in indexes_to_exclude:
                    if type(visitation[index_label]) == float or type(visitation[index_label]) == int:
                        if np.isnan(visitation[index_label]):
                            visitation[index_label] = 0
                    elif type(visitation[index_label]) == str:
                        if visitation[index_label] == "nan":
                            visitation[index_label] = "0"
                    else:
                        print(f'Error other data type: index label: {index_label}, type: {type(visitation[index_label])}, value: {visitation[index_label]}')

    return patients