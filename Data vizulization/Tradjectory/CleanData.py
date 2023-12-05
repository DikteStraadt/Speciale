import numpy as np
from dateutil import relativedelta
from datetime import datetime
import pandas as pd

def clean_columns(data):

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

    return data

def convert_time_stamps(data):

    time_stamp_columns = ['birth', 'first_visitation', 'second_US', 'third_US', 'fourth_US', 'fifth_US', 'sixth_US', 'seventh_US', 'eighth_US', 'ninth_US', 'tenth_US', 'eleventh_US', 'twelfth_US', 'thirteenth_US', 'fourteenth_US', 'fifteenth_US', 'sixteenth_US', 'seventeenth_US']

    for i, patient in data.iterrows():
        for column in time_stamp_columns:
            value = data[column][i]
            if value != "#NULL!" and type(value) != float:
                data[column][i] = datetime.strptime(value, "%d-%b-%y")

    return data