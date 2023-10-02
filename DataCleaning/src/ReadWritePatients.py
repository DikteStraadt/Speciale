import pandas as pd

def read_patients(data):

    visitation_headers = []
    list_patients = []

    headers_visitation_0 = ('first_visitation', 'Alder_ved_afslut')
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
         headers_visitation_14, headers_visitation_15, headers_visitation_16, headers_patient_info_first,
         headers_patient_info_last])

    for patient in data.iloc:

        for visitation in visitation_headers:

            start_index = data.columns.get_loc(visitation[0])
            end_index = data.columns.get_loc(visitation[1])
            list_visitations = patient[start_index: end_index + 1]
            list_patients.append(list_visitations)

    return list_patients