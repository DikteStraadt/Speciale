import pandas as pd
import numpy as np

def read_visitations(data):

    visitation_headers = []
    list_list_visitations = []

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

    for visitation in visitation_headers:
        start_index = data.columns.get_loc(visitation[0])
        end_index = data.columns.get_loc(visitation[1])
        list_visitations = data.iloc[:, start_index: end_index + 1]
        list_list_visitations.append(list_visitations)

    return list_list_visitations

def combine_to_dataframe(visitations_3D):

    data_frames = []

    for visitation_2D in visitations_3D[0:17]:
        data_frames.append(visitation_2D)

    result_df = pd.concat(data_frames, axis=0, ignore_index=True)

    columns_to_merge = {}
    columns_to_merge['visitationdate'] = ['first_visitation', 'second_US', 'third_US', 'fourth_US', 'fifth_US', 'sixth_US', 'seventh_US', 'eighth_US', 'ninth_US', 'tenth_US', 'eleventh_US', 'twelfth_US', 'thirteenth_US', 'fourteenth_US', 'fifteenth_US', 'sixteenth_US', 'seventeenth_US']
    column_names = ['firstvisitation', 'secondUS', 'thirdUS', 'fourthUS', 'fifthUS', 'sixthUS', 'seventhUS', 'eighthUS', 'ninthUS', 'tenthUS', 'eleventhUS', 'twelfthUS', 'thirteenthUS', 'fourteenthUS', 'fifteenthUS', 'sixteenthUS', 'seventeenthUS']

    for col in result_df.columns:
        prefix = ''.join(filter(str.isalpha, col))
        if prefix not in columns_to_merge:
            columns_to_merge[prefix] = []
        columns_to_merge[prefix].append(col)

    for value in column_names:
        del columns_to_merge[value]

    new_df = pd.DataFrame()

    for prefix, cols in columns_to_merge.items():
        new_df[prefix] = result_df[cols].apply(lambda row: ', '.join(row.dropna().astype(str)), axis=1)

    return new_df


