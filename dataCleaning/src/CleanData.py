import numpy as np
from dateutil import relativedelta
from datetime import datetime

def rename_columns(visitations_2D):

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

    return data.reset_index(drop=True)
