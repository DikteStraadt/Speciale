import numpy as np
import pandas as pd

class ReadVisitations:

    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):
        visitation_headers = []
        list_list_visitations = []

        headers_visitation_0 = ('first_visitation', 'transversal', 'diff_t0_first_visitation', 'age_at_first_visitation')
        headers_visitation_1 = ('second_US', 'transversal1', 'diff_t0_second_US', 'age_at_second_US')
        headers_visitation_2 = ('third_US', 'transversal2', 'diff_t0_third_US', 'age_at_third_US')
        headers_visitation_3 = ('fourth_US', 'transversal3', 'diff_t0_fourth_US', 'age_at_fourth_US')
        headers_visitation_4 = ('fifth_US', 'transversal4', 'diff_t0_fifth_US', 'age_at_fifth_US')
        headers_visitation_5 = ('sixth_US', 'transversal5', 'diff_t0_sixth_US', 'age_at_sixth_US')
        headers_visitation_6 = ('seventh_US', 'transversal6', 'diff_t0_seventh_US', 'age_at_seventh_US')
        headers_visitation_7 = ('eighth_US', 'transversal7', 'diff_t0_eighth_US', 'age_at_eighth_US')
        headers_visitation_8 = ('ninth_US', 'transversal8', 'diff_t0_ninth_US', 'age_at_ninth_US')
        headers_visitation_9 = ('tenth_US', 'transversal9', 'diff_t0_tenth_US', 'age_at_tenth_US')
        headers_visitation_10 = ('eleventh_US', 'transversal10', 'diff_t0_eleventh_US', 'age_at_eleventh_US')
        headers_visitation_11 = ('twelfth_US', 'transversal11', 'diff_t0_twelfth_US', 'age_at_twelfth_US')
        headers_visitation_12 = ('thirteenth_US', 'transversal12', 'diff_t0_thirteenth_US', 'age_at_thirteenth_US')
        headers_visitation_13 = ('fourteenth_US', 'transversal13', 'diff_t0_fourteenth_US', 'age_at_fourteenth_US')
        headers_visitation_14 = ('fifteenth_US', 'transversal14', 'diff_t0_fifteenth_US', 'age_at_fifteenth_US')
        headers_visitation_15 = ('sixteenth_US', 'transversal15', 'diff_t0_sixteenth_US', 'age_at_sixteenth_US')
        headers_visitation_16 = ('seventeenth_US', 'transversal16', 'diff_t0_seventeenth_US', 'age_at_seventeenth_US')
        headers_patient_info_first = ('ID', 'overall TMJ involvement')
        headers_patient_info_last = ('Dec-11', 'DysfTMJ_Pain_during_palpation')


        visitation_headers.extend(
            [headers_visitation_0, headers_visitation_1, headers_visitation_2, headers_visitation_3, headers_visitation_4,
            headers_visitation_5, headers_visitation_6, headers_visitation_7, headers_visitation_8, headers_visitation_9,
            headers_visitation_10, headers_visitation_11, headers_visitation_12, headers_visitation_13, headers_visitation_14, headers_visitation_15, headers_visitation_16])

        i = 0

        for visitation in visitation_headers:
            start_index = data.columns.get_loc(visitation[0])
            end_index = data.columns.get_loc(visitation[1])
            list_visitations = data.iloc[:, start_index: end_index + 1]

            if len(visitation) == 4:
                list_visitations.insert(0, 'diff_t0_date', data[visitation[2]])
                list_visitations.insert(0, 'age_at_visitation', data[visitation[3]])

            list_visitations.insert(0, 'type', data['type'])
            list_visitations.insert(0, 'sex', data['sex'])
            new_id_column = data['ID'].apply(lambda x: x + f'_{i}')
            list_visitations.insert(0, 'ID', new_id_column)
            list_visitations.insert(0, 'study_id', data['study_id'])
            list_list_visitations.append(list_visitations)
            i = i + 1

        print("Data converted to visitation format")

        return list_list_visitations

class CombineToDataFrame:

    def __init__(self, previous_two_values):
        self.previous_two_values = previous_two_values

    def fit(self, visitations_3D, y=None):
        return self

    def transform(self, visitations_3D, y=None):

        data_frames = []

        for visitation_2D in visitations_3D[0:17]:
            data_frames.append(visitation_2D)

        result_df = pd.concat(data_frames, axis=0, ignore_index=True)

        columns_to_merge = {}
        if self.previous_two_values == "y-1":
            columns_to_merge['visitationdate'] = ['second_US', 'third_US', 'fourth_US', 'fifth_US',
                                                  'sixth_US', 'seventh_US', 'eighth_US', 'ninth_US', 'tenth_US',
                                                  'eleventh_US', 'twelfth_US', 'thirteenth_US', 'fourteenth_US',
                                                  'fifteenth_US', 'sixteenth_US', 'seventeenth_US']

            column_names = ['secondUS', 'thirdUS', 'fourthUS', 'fifthUS', 'sixthUS', 'seventhUS',
                        'eighthUS', 'ninthUS', 'tenthUS', 'eleventhUS', 'twelfthUS', 'thirteenthUS', 'fourteenthUS', 'fifteenthUS', 'sixteenthUS', 'seventeenthUS']

        elif self.previous_two_values == "y-2":
            columns_to_merge['visitationdate'] = ['third_US', 'fourth_US', 'fifth_US',
                                                  'sixth_US', 'seventh_US', 'eighth_US', 'ninth_US', 'tenth_US',
                                                  'eleventh_US', 'twelfth_US', 'thirteenth_US', 'fourteenth_US',
                                                  'fifteenth_US', 'sixteenth_US', 'seventeenth_US']

            column_names = ['thirdUS', 'fourthUS', 'fifthUS', 'sixthUS', 'seventhUS',
                        'eighthUS', 'ninthUS', 'tenthUS', 'eleventhUS', 'twelfthUS', 'thirteenthUS', 'fourteenthUS', 'fifteenthUS', 'sixteenthUS', 'seventeenthUS']

        else:
            columns_to_merge['visitationdate'] = ['first_visitation', 'second_US', 'third_US', 'fourth_US', 'fifth_US',
                                                  'sixth_US', 'seventh_US', 'eighth_US', 'ninth_US', 'tenth_US',
                                                  'eleventh_US', 'twelfth_US', 'thirteenth_US', 'fourteenth_US',
                                                  'fifteenth_US', 'sixteenth_US', 'seventeenth_US']

            column_names = ['firstvisitation', 'secondUS', 'thirdUS', 'fourthUS', 'fifthUS', 'sixthUS', 'seventhUS',
                            'eighthUS', 'ninthUS', 'tenthUS', 'eleventhUS', 'twelfthUS', 'thirteenthUS', 'fourteenthUS',
                            'fifteenthUS', 'sixteenthUS', 'seventeenthUS']

        for col in result_df.columns:

            if not col.startswith("previous_involvement_status_"):
                prefix = ''.join(filter(str.isalpha, col))
            else:
                prefix = col.replace("_", "")

            if prefix not in columns_to_merge:
                columns_to_merge[prefix] = []
            columns_to_merge[prefix].append(col)

        for value in column_names:
            del columns_to_merge[value]

        new_df = pd.DataFrame()

        for prefix, cols in columns_to_merge.items():
            new_df[prefix] = result_df[cols].apply(lambda row: ', '.join(row.dropna().astype(str)), axis=1)

        new_df.replace(to_replace='', value=np.nan, inplace=True)

        new_df.rename(columns={'previousinvolvementstatusvisitationy-1': 'previousinvolvementstatusvisitation_y-1'}, inplace=True)
        new_df.rename(columns={'previousinvolvementstatusvisitationy-2': 'previousinvolvementstatusvisitation_y-2'}, inplace=True)

        print("Visitations combined to single data frame")

        return new_df