from datetime import datetime

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta


class CleanColumns:

    def fit(self, X, y=None):
        return self

    def transform(self, data, y=None):

        # involvement_status
        data.rename(columns={'involvment_status_1': 'involvement_status_1'}, inplace=True)
        data.rename(columns={'involvment_status_2': 'involvement_status_2'}, inplace=True)
        data.rename(columns={'involvment_status_3': 'involvement_status_3'}, inplace=True)
        data.rename(columns={'involvment_status_4': 'involvement_status_4'}, inplace=True)
        data.rename(columns={'involvment_status_5': 'involvement_status_5'}, inplace=True)
        data.rename(columns={'involvment_status_6': 'involvement_status_6'}, inplace=True)
        data.rename(columns={'involvment_status_7': 'involvement_status_7'}, inplace=True)
        data.rename(columns={'involvment_status_8': 'involvement_status_8'}, inplace=True)
        data.rename(columns={'involvment_status_9': 'involvement_status_9'}, inplace=True)
        data.rename(columns={'involvment_status_10': 'involvement_status_10'}, inplace=True)
        data.rename(columns={'involvment_status_11': 'involvement_status_11'}, inplace=True)
        data.rename(columns={'involvment_status_12': 'involvement_status_12'}, inplace=True)
        data.rename(columns={'involvering_status_13': 'involvement_status_13'}, inplace=True)
        data.rename(columns={'involvering_status_14': 'involvement_status_14'}, inplace=True)
        data.rename(columns={'involvering_status_15': 'involvement_status_15'}, inplace=True)
        data.rename(columns={'involvment_status_16': 'involvement_status_16'}, inplace=True)

        # US
        data.rename(columns={'eigth_US': 'eighth_US'}, inplace=True)
        data.rename(columns={'nineth_US': 'ninth_US'}, inplace=True)
        data.rename(columns={'twelth_US': 'twelfth_US'}, inplace=True)

        # later_palp_right
        data.rename(columns={'later_palp_righ13': 'later_palp_right13'}, inplace=True)
        data.rename(columns={'later_palp_righ16': 'later_palp_right16'}, inplace=True)

        # temp_sen_right
        data.rename(columns={'tem_sen_right1': 'temp_sen_right1'}, inplace=True)
        data.rename(columns={'tem_sen_right2': 'temp_sen_right2'}, inplace=True)
        data.rename(columns={'tem_sen_right3': 'temp_sen_right3'}, inplace=True)
        data.rename(columns={'tem_sen_right4': 'temp_sen_right4'}, inplace=True)
        data.rename(columns={'tem_sen_right5': 'temp_sen_right5'}, inplace=True)
        data.rename(columns={'tem_sen_right6': 'temp_sen_right6'}, inplace=True)
        data.rename(columns={'tem_sen_right7': 'temp_sen_right7'}, inplace=True)
        data.rename(columns={'tem_sen_right8': 'temp_sen_right8'}, inplace=True)
        data.rename(columns={'tem_sen_right9': 'temp_sen_right9'}, inplace=True)
        data.rename(columns={'tem_sen_right10': 'temp_sen_right10'}, inplace=True)
        data.rename(columns={'tem_sen_right11': 'temp_sen_right11'}, inplace=True)
        data.rename(columns={'tem_sen_right12': 'temp_sen_right12'}, inplace=True)
        data.rename(columns={'tem_sen_right13': 'temp_sen_right13'}, inplace=True)
        data.rename(columns={'tem_sen_right14': 'temp_sen_right14'}, inplace=True)
        data.rename(columns={'tem_sen_right15': 'temp_sen_right15'}, inplace=True)
        data.rename(columns={'tem_sen_right16': 'temp_sen_right16'}, inplace=True)

        # Remove click_latero columns
        data = data.drop(columns=['click_lateroleft_right'])
        data = data.drop(columns=['click_lateroright_left'])

        # Remove traction columns
        data = data.drop(columns=['traction_right'])
        data = data.drop(columns=['traction_left'])
        data = data.drop(columns=['traction_right1'])
        data = data.drop(columns=['traction_left1'])
        data = data.drop(columns=['traction_right2'])
        data = data.drop(columns=['traction_left2'])
        data = data.drop(columns=['traction_right3'])
        data = data.drop(columns=['traction_left3'])
        data = data.drop(columns=['traction_right4'])
        data = data.drop(columns=['traction_left4'])
        data = data.drop(columns=['traction_right5'])
        data = data.drop(columns=['traction_left5'])
        data = data.drop(columns=['traction_right6'])
        data = data.drop(columns=['traction_left6'])
        data = data.drop(columns=['traction_right7'])
        data = data.drop(columns=['traction_left7'])
        data = data.drop(columns=['traction_right8'])
        data = data.drop(columns=['traction_left8'])
        data = data.drop(columns=['traction_right9'])
        data = data.drop(columns=['traction_left9'])
        data = data.drop(columns=['traction_right10'])
        data = data.drop(columns=['traction_left10'])
        data = data.drop(columns=['traction_right11'])
        data = data.drop(columns=['traction_left11'])
        data = data.drop(columns=['traction_right12'])
        data = data.drop(columns=['traction_left12'])
        data = data.drop(columns=['traction_right13'])
        data = data.drop(columns=['traction_left13'])
        data = data.drop(columns=['traction_right14'])
        data = data.drop(columns=['traction_left14'])
        data = data.drop(columns=['traction_right15'])
        data = data.drop(columns=['traction_left15'])
        data = data.drop(columns=['traction_right16'])
        data = data.drop(columns=['traction_left16'])

        # rename columns in visitation 0
        data.rename(columns={'micrognathism': 'retrognathism'}, inplace=True)
        data.rename(columns={'click_lateroright_right': 'click_laterotrusion_right'}, inplace=True)
        data.rename(columns={'click_lateroleft_left': 'click_laterotrusion_left'}, inplace=True)

        print("Columns cleaned")

        return data

class EditData:

    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):
        data.at[146, 'birth'] = "4-Sep-98"  # Patient 147
        data.at[178, 'birth'] = "6-Feb-06"  # Patient 179
        data.at[917, 'birth'] = "27-Nov-00"  # Patient 918

        # data.at[421, 'first_visitation'] = ""  # Patient 422
        # data.at[532, 'first_visitation'] = ""  # Patient 533
        # data.at[654, 'first_visitation'] = ""  # Patient 655
        # data.at[971, 'first_visitation'] = ""  # Patient 972

        data.at[7, 'type'] = 0  # Patient 8
        data.at[40, 'type'] = 1  # Patient 41
        data.at[58, 'type'] = 1  # Patient 59
        data.at[68, 'type'] = 1  # Patient 69
        data.at[70, 'type'] = 0  # Patient 71
        data.at[97, 'type'] = 1  # Patient 98
        data.at[112, 'type'] = 0  # Patient 113
        data.at[140, 'type'] = 4  # Patient 141
        data.at[161, 'type'] = 8  # Patient 162
        data.at[192, 'type'] = 0  # Patient 193
        data.at[222, 'type'] = 0  # Patient 223
        data.at[234, 'type'] = 0  # Patient 235
        data.at[241, 'type'] = 1  # Patient 242
        data.at[255, 'type'] = 0  # Patient 256
        data.at[265, 'type'] = 0  # Patient 266
        data.at[276, 'type'] = 0  # Patient 277
        data.at[286, 'type'] = 0  # Patient 287
        data.at[321, 'type'] = 1  # Patient 322
        data.at[322, 'type'] = 0  # Patient 323
        data.at[357, 'type'] = 0  # Patient 358
        data.at[370, 'type'] = 1  # Patient 371
        data.at[380, 'type'] = 1  # Patient 381
        data.at[400, 'type'] = 1  # Patient 401
        data.at[401, 'type'] = 1  # Patient 402
        data.at[455, 'type'] = 1  # Patient 456
        data.at[473, 'type'] = 0  # Patient 474
        data.at[485, 'type'] = 1  # Patient 486
        data.at[504, 'type'] = 0  # Patient 505
        data.at[519, 'type'] = 1  # Patient 520
        data.at[529, 'type'] = 1  # Patient 530
        data.at[551, 'type'] = 1  # Patient 552
        data.at[564, 'type'] = 0  # Patient 565
        data.at[568, 'type'] = 1  # Patient 569
        data.at[610, 'type'] = 0  # Patient 611
        data.at[619, 'type'] = 0  # Patient 620
        data.at[647, 'type'] = 8  # Patient 648
        data.at[659, 'type'] = 0  # Patient 660
        data.at[774, 'type'] = 0  # Patient 775
        data.at[800, 'type'] = 0  # Patient 801
        data.at[813, 'type'] = 0  # Patient 814
        data.at[839, 'type'] = 0  # Patient 840
        data.at[927, 'type'] = 0  # Patient 928
        data.at[933, 'type'] = 0  # Patient 934
        data.at[964, 'type'] = 0  # Patient 965
        data.at[970, 'type'] = 0  # Patient 971
        data.at[971, 'type'] = 0  # Patient 972
        data.at[974, 'type'] = 4  # Patient 975
        data.at[1016, 'type'] = 4  # Patient 1017
        data.at[1029, 'type'] = 0  # Patient 1030
        data.at[1048, 'type'] = 1  # Patient 1049

        # asy_basis
        data.at[756, 'asy_basis5'] = 1  # Patient 757 (6018z)
        data.at[828, 'asy_basis2'] = 2  # Patient 829 (2957a)
        data.at[1051, 'asy_basis'] = 3  # Patient 1052 (7269b)

        # asy_menton
        data.at[756, 'asy_menton5'] = 1  # Patient 757 (6018z)
        data.at[828, 'asy_menton2'] = 2  # Patient 829 (2957a)
        data.at[1051, 'asy_menton'] = 3  # Patient 1052 (7269b)

        # asy_occl
        data.at[756, 'asy_occl5'] = 1  # Patient 757 (6018z)
        data.at[55, 'asy_occl10'] = 1  # Patient 56 (5617u)

        # asy_low_mi
        data.at[871, 'asy_low_mi6'] = 3  # Patient 872 (2551z)
        data.at[347, 'asy_low_mi'] = 4  # Patient 348 (3804A)

        # profile
        data.at[90, 'profile9'] = 2  # Patient 91 (1129z)

        # lower_face
        data.at[865, 'lower_face2'] = 1  # Patient 866 (2273y)

        # space_relationship
        data.at[465, 'space_relationship2'] = 0  # Patient 466 (9051u)
        data.at[750, 'space_relationship1'] = 0  # Patient 751 (8795b)
        data.at[721, 'space_relationship1'] = 0  # Patient 722 (9297b)
        data.at[639, 'space_relationship5'] = 3  # Patient 640 (6371u)
        data.at[763, 'space_relationship'] = 2  # Patient 764 (9130u)
        data.at[53, 'space_relationship'] = 1  # Patient 54 (7624y)
        data.at[124, 'space_relationship'] = 2  # Patient 125 (7062y)
        data.at[699, 'space_relationship'] = 2  # Patient 700 (7766y)
        data.at[47, 'space_relationship'] = 2  # Patient 48 (7249y)
        data.at[763, 'space_relationship'] = 0  # Patient 764 (9251u)

        data.at[91, 'space_relationship'] = 0  # Patient 92 (4934a)
        data.at[8, 'space_relationship'] = 3  # Patient 9 (2188b)
        data.at[385, 'space_relationship'] = 2  # Patient 386 (7300y)
        data.at[263, 'space_relationship'] = 0  # Patient 264 (6781y)
        data.at[477, 'space_relationship'] = 1  # Patient 478 (1989a)
        data.at[305, 'space_relationship'] = 3  # Patient 306 (3203y)
        data.at[828, 'space_relationship'] = 4  # Patient 829 (2957a)
        data.at[853, 'space_relationship'] = 0  # Patient 854 (0049T)
        data.at[219, 'space_relationship'] = 0  # Patient 220 (4056b)
        data.at[713, 'space_relationship'] = 0  # Patient 714 (9080b)

        # opening
        data.at[219, 'opening2'] = 55  # Patient 220 (4056b)

        # protrusion
        data.at[219, 'protrusion2'] = 8  # Patient 220 (4056b)

        print("Misregistered data edited")

        return data

class RemovePatients:

    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):

        # Find and remove patients older than 18
        indexes = []

        for i in range(len(data)):
            date_birth = datetime.strptime(data['birth'][i], "%d-%b-%y")
            date_visitation = datetime.strptime(data['first_visitation'][i], "%d-%b-%y")
            difference = relativedelta(date_visitation, date_birth)

            if difference.years < 0 or difference.years > 18:
                indexes.append(i)

        for i in sorted(indexes, reverse=True):
            data = data.drop(i)

        print(f'Indexes to be removed: {indexes}')
        print("Old patients removed")

        # Drop duplicate patients (7867z, 6475a, 7023a, 8228a, 8417b, 8624a, 8937a, 6593a, 1965q, 6248z)
        index_to_remove = []
        index_to_remove.append(data[data['ID'] == '7867z'].index[0])
        index_to_remove.append(data[data['ID'] == '6475a'].index[0])
        index_to_remove.append(data[data['ID'] == '7023a'].index[0])
        index_to_remove.append(data[data['ID'] == '8228a'].index[0])
        index_to_remove.append(data[data['ID'] == '8417b'].index[0])
        index_to_remove.append(data[data['ID'] == '8624a'].index[0])
        index_to_remove.append(data[data['ID'] == '8937a'].index[0])
        index_to_remove.append(data[data['ID'] == '6593a'].index[0])
        index_to_remove.append(data[data['ID'] == '1965q'].index[0])
        index_to_remove.append(data[data['ID'] == '6248z'].index[0])

        data = data.drop(index=index_to_remove)
        data.reset_index(drop=True, inplace=True)

        print("Duplicate patients removed")

        return data

class ConvertTimestamps:

    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):

        time_stamp_columns = ['birth', 'first_visitation', 'second_US', 'third_US', 'fourth_US', 'fifth_US', 'sixth_US',
                              'seventh_US', 'eighth_US', 'ninth_US', 'tenth_US', 'eleventh_US', 'twelfth_US',
                              'thirteenth_US', 'fourteenth_US', 'fifteenth_US', 'sixteenth_US', 'seventeenth_US']

        for i, patient in data.iterrows():
            for column in time_stamp_columns:
                value = data[column][i]
                if value != "#NULL!" and type(value) != float:
                    data[column][i] = datetime.strptime(value, "%d-%b-%y")

        print("Dates converted to timestamps")

        return data




