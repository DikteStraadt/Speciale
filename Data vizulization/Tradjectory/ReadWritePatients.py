import pandas as pd

def unify_index_labels(visitation):

    i = 0

    for column in visitation:

        label = visitation.index[i]

        if label in ['first_visitation', 'second_US', 'third_US', 'fourth_US', 'fifth_US', 'sixth_US', 'seventh_US', 'eighth_US', 'ninth_US', 'tenth_US', 'eleventh_US', 'twelfth_US', 'thirteenth_US', 'fourteenth_US', 'fifteenth_US', 'sixteenth_US', 'seventeenth_US']:
            visitation.index.values[i] = "visitationdate"
        else:
            prefix = ''.join(filter(str.isalpha, label))
            visitation.index.values[i] = prefix

        i = i + 1

    return visitation

def read_patients(data):

    visitation_headers = []
    patients_df = []

    headers_visitation_0 = ('first_visitation', 'transversal')
    headers_visitation_1 = ('second_US', 'transversal1')
    headers_visitation_2 = ('third_US', 'transversal2')
    headers_visitation_3 = ('fourth_US', 'transversal3')
    headers_visitation_4 = ('fifth_US', 'transversal4')
    headers_visitation_5 = ('sixth_US', 'transversal5')
    headers_visitation_6 = ('seventh_US', 'transversal6')
    headers_visitation_7 = ('eighth_US', 'transversal7')
    headers_visitation_8 = ('ninth_US', 'transversal8')
    headers_visitation_9 = ('tenth_US', 'transversal9')
    headers_visitation_10 = ('eleventh_US', 'transversal10')
    headers_visitation_11 = ('twelfth_US', 'transversal11')
    headers_visitation_12 = ('thirteenth_US', 'transversal12')
    headers_visitation_13 = ('fourteenth_US', 'transversal13')
    headers_visitation_14 = ('fifteenth_US', 'transversal14')
    headers_visitation_15 = ('sixteenth_US', 'transversal15')
    headers_visitation_16 = ('seventeenth_US', 'transversal16')
    headers_patient_info_first = ('ID', 'overall TMJ involvement')
    headers_patient_info_last = ('Dec-11', 'DysfTMJ_Pain_during_palpation')

    visitation_headers.extend(
        [headers_visitation_0, headers_visitation_1, headers_visitation_2, headers_visitation_3, headers_visitation_4,
         headers_visitation_5, headers_visitation_6, headers_visitation_7, headers_visitation_8, headers_visitation_9,
         headers_visitation_10, headers_visitation_11, headers_visitation_12, headers_visitation_13,
         headers_visitation_14, headers_visitation_15, headers_visitation_16])

    for i, patient in data.iterrows():

        patient_list = []

        for visitation_labels in visitation_headers:

            visitation = patient.loc[visitation_labels[0]:visitation_labels[1]]

            new_values = [patient['study_id'], patient['sex'], patient['type']]
            index_labels = ['study_id', 'sex', 'type']
            new_series = pd.Series(new_values, index=index_labels)
            visitation = pd.concat([new_series, visitation])

            visitation = unify_index_labels(visitation)

            patient_list.append(visitation)

        patients_df.append(patient_list)

    return patients_df