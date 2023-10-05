import numpy as np
from dateutil import relativedelta
from datetime import datetime

def clean_columns(visitations_2D):

    # involvement_status
    visitations_2D.rename(columns={'involvment_status_1': 'involvement_status_1'}, inplace=True)
    visitations_2D.rename(columns={'involvment_status_2': 'involvement_status_2'}, inplace=True)
    visitations_2D.rename(columns={'involvment_status_3': 'involvement_status_3'}, inplace=True)
    visitations_2D.rename(columns={'involvment_status_4': 'involvement_status_4'}, inplace=True)
    visitations_2D.rename(columns={'involvment_status_5': 'involvement_status_5'}, inplace=True)
    visitations_2D.rename(columns={'involvment_status_6': 'involvement_status_6'}, inplace=True)
    visitations_2D.rename(columns={'involvment_status_7': 'involvement_status_7'}, inplace=True)
    visitations_2D.rename(columns={'involvment_status_8': 'involvement_status_8'}, inplace=True)
    visitations_2D.rename(columns={'involvment_status_9': 'involvement_status_9'}, inplace=True)
    visitations_2D.rename(columns={'involvment_status_10': 'involvement_status_10'}, inplace=True)
    visitations_2D.rename(columns={'involvment_status_11': 'involvement_status_11'}, inplace=True)
    visitations_2D.rename(columns={'involvment_status_12': 'involvement_status_12'}, inplace=True)
    visitations_2D.rename(columns={'involvering_status_13': 'involvement_status_13'}, inplace=True)
    visitations_2D.rename(columns={'involvering_status_14': 'involvement_status_14'}, inplace=True)
    visitations_2D.rename(columns={'involvering_status_15': 'involvement_status_15'}, inplace=True)
    visitations_2D.rename(columns={'involvment_status_16': 'involvement_status_16'}, inplace=True)

    # US
    visitations_2D.rename(columns={'eigth_US': 'eighth_US'}, inplace=True)
    visitations_2D.rename(columns={'nineth_US': 'ninth_US'}, inplace=True)
    visitations_2D.rename(columns={'twelth_US': 'twelfth_US'}, inplace=True)

    # later_palp_right
    visitations_2D.rename(columns={'later_palp_righ13': 'later_palp_right13'}, inplace=True)
    visitations_2D.rename(columns={'later_palp_righ16': 'later_palp_right16'}, inplace=True)

    # temp_sen_right
    visitations_2D.rename(columns={'tem_sen_right1': 'temp_sen_right1'}, inplace=True)
    visitations_2D.rename(columns={'tem_sen_right2': 'temp_sen_right2'}, inplace=True)
    visitations_2D.rename(columns={'tem_sen_right3': 'temp_sen_right3'}, inplace=True)
    visitations_2D.rename(columns={'tem_sen_right4': 'temp_sen_right4'}, inplace=True)
    visitations_2D.rename(columns={'tem_sen_right5': 'temp_sen_right5'}, inplace=True)
    visitations_2D.rename(columns={'tem_sen_right6': 'temp_sen_right6'}, inplace=True)
    visitations_2D.rename(columns={'tem_sen_right7': 'temp_sen_right7'}, inplace=True)
    visitations_2D.rename(columns={'tem_sen_right8': 'temp_sen_right8'}, inplace=True)
    visitations_2D.rename(columns={'tem_sen_right9': 'temp_sen_right9'}, inplace=True)
    visitations_2D.rename(columns={'tem_sen_right10': 'temp_sen_right10'}, inplace=True)
    visitations_2D.rename(columns={'tem_sen_right11': 'temp_sen_right11'}, inplace=True)
    visitations_2D.rename(columns={'tem_sen_right12': 'temp_sen_right12'}, inplace=True)
    visitations_2D.rename(columns={'tem_sen_right13': 'temp_sen_right13'}, inplace=True)
    visitations_2D.rename(columns={'tem_sen_right14': 'temp_sen_right14'}, inplace=True)
    visitations_2D.rename(columns={'tem_sen_right15': 'temp_sen_right15'}, inplace=True)
    visitations_2D.rename(columns={'tem_sen_right16': 'temp_sen_right16'}, inplace=True)

    # Remove redundant columns in visitation 0
    visitations_2D = visitations_2D.drop(columns=['click_lateroleft_right'])
    visitations_2D = visitations_2D.drop(columns=['click_lateroright_left'])

    # rename columns in visitation 0
    visitations_2D.rename(columns={'micrognathism': 'retrognathism'}, inplace=True)
    visitations_2D.rename(columns={'click_lateroright_right': 'click_laterotrusion_right'}, inplace=True)
    visitations_2D.rename(columns={'click_lateroleft_left': 'click_laterotrusion_left'}, inplace=True)

    return visitations_2D

def edit_misregistered_data(visitations_2D):

    visitations_2D.at[146, 'birth'] = "4-Sep-98"  # Patient 147
    visitations_2D.at[178, 'birth'] = "6-Feb-06"  # Patient 179
    visitations_2D.at[917, 'birth'] = "27-Nov-00"  # Patient 918

    # visitations_2D.at[421, 'first_visitation'] = ""  # Patient 422
    # visitations_2D.at[532, 'first_visitation'] = ""  # Patient 533
    # visitations_2D.at[654, 'first_visitation'] = ""  # Patient 655
    # visitations_2D.at[971, 'first_visitation'] = ""  # Patient 972

    visitations_2D.at[7, 'type'] = 0  # Patient 8
    visitations_2D.at[40, 'type'] = 1  # Patient 41
    visitations_2D.at[58, 'type'] = 1  # Patient 59
    visitations_2D.at[68, 'type'] = 1  # Patient 69
    visitations_2D.at[70, 'type'] = 0  # Patient 71
    visitations_2D.at[97, 'type'] = 1  # Patient 98
    visitations_2D.at[112, 'type'] = 0  # Patient 113
    visitations_2D.at[140, 'type'] = 4  # Patient 141
    visitations_2D.at[161, 'type'] = 8  # Patient 162
    visitations_2D.at[192, 'type'] = 0  # Patient 193
    visitations_2D.at[222, 'type'] = 0  # Patient 223
    visitations_2D.at[234, 'type'] = 0  # Patient 235
    visitations_2D.at[241, 'type'] = 1  # Patient 242
    visitations_2D.at[255, 'type'] = 0  # Patient 256
    visitations_2D.at[265, 'type'] = 0  # Patient 266
    visitations_2D.at[276, 'type'] = 0  # Patient 277
    visitations_2D.at[286, 'type'] = 0  # Patient 287
    visitations_2D.at[321, 'type'] = 1  # Patient 322
    visitations_2D.at[322, 'type'] = 0  # Patient 323
    visitations_2D.at[357, 'type'] = 0  # Patient 358
    visitations_2D.at[370, 'type'] = 1  # Patient 371
    visitations_2D.at[380, 'type'] = 1  # Patient 381
    visitations_2D.at[400, 'type'] = 1  # Patient 401
    visitations_2D.at[401, 'type'] = 1  # Patient 402
    visitations_2D.at[455, 'type'] = 1  # Patient 456
    visitations_2D.at[473, 'type'] = 0  # Patient 474
    visitations_2D.at[485, 'type'] = 1  # Patient 486
    visitations_2D.at[504, 'type'] = 0  # Patient 505
    visitations_2D.at[519, 'type'] = 1  # Patient 520
    visitations_2D.at[529, 'type'] = 1  # Patient 530
    visitations_2D.at[551, 'type'] = 1  # Patient 552
    visitations_2D.at[564, 'type'] = 0  # Patient 565
    visitations_2D.at[568, 'type'] = 1  # Patient 569
    visitations_2D.at[610, 'type'] = 0  # Patient 611
    visitations_2D.at[619, 'type'] = 0  # Patient 620
    visitations_2D.at[647, 'type'] = 8  # Patient 648
    visitations_2D.at[659, 'type'] = 0  # Patient 660
    visitations_2D.at[774, 'type'] = 0  # Patient 775
    visitations_2D.at[800, 'type'] = 0  # Patient 801
    visitations_2D.at[813, 'type'] = 0  # Patient 814
    visitations_2D.at[839, 'type'] = 0  # Patient 840
    visitations_2D.at[927, 'type'] = 0  # Patient 928
    visitations_2D.at[933, 'type'] = 0  # Patient 934
    visitations_2D.at[964, 'type'] = 0  # Patient 965
    visitations_2D.at[970, 'type'] = 0  # Patient 971
    visitations_2D.at[971, 'type'] = 0  # Patient 972
    visitations_2D.at[974, 'type'] = 4  # Patient 975
    visitations_2D.at[1016, 'type'] = 4  # Patient 1017
    visitations_2D.at[1029, 'type'] = 0  # Patient 1030
    visitations_2D.at[1048, 'type'] = 1  # Patient 1049

    return visitations_2D


def remove_old_patients(data):

    indexes = []

    for i in range(len(data)):
        date_birth = datetime.strptime(data['birth'][i], "%d-%b-%y")
        date_visitation = datetime.strptime(data['first_visitation'][i], "%d-%b-%y")
        difference = relativedelta.relativedelta(date_visitation, date_birth)

        if difference.years < 0 or difference.years > 18:
            indexes.append(i)

    for i in sorted(indexes, reverse=True):
        data = data.drop(i)

    print(f'Indexes to be removed: {indexes}')

    return data.reset_index(drop=True)

