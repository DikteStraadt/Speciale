import pandas as pd


class OpeningTransformer:

    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):

        age_intervals = [(0, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 18), (18, 32)]
        sex_values = [0, 1]  # 0 = boy, 1 = girl
        expected_values = { ((0, 4), 0): 34.8, ((0, 4), 1): 34.8,
                            ((4, 5), 0): 40.4, ((4, 5), 1): 40.4,
                            ((5, 6), 0): 40.7, ((5, 6), 1): 40.7,
                            ((6, 7), 0): 42.2, ((6, 7), 1): 41.8,
                            ((7, 8), 0): 43.6, ((7, 8), 1): 43.2,
                            ((8, 9), 0): 44.4, ((8, 9), 1): 44.0,
                            ((9, 10), 0): 45.5, ((9, 10), 1): 45.3,
                            ((10, 11), 0): 46.1, ((10, 11), 1): 46.2,
                            ((11, 12), 0): 47.3, ((11, 12), 1): 47.6,
                            ((12, 13), 0): 48.5, ((12, 13), 1): 48.1,
                            ((13, 14), 0): 49.2, ((13, 14), 1): 48.9,
                            ((14, 15), 0): 51.2, ((14, 15), 1): 48.8,
                            ((15, 16), 0): 51.6, ((15, 16), 1): 49.9,
                            ((16, 17), 0): 52.0, ((16, 17), 1): 50.0,
                            ((17, 18), 0): 51.2, ((17, 18), 1): 50.4,
                            ((18, 32), 0): 56.3, ((18, 32), 1): 50.4
                           }

        # Children: https://www.researchgate.net/publication/236264893_Maximal_mouth_opening_capacity_Percentiles_for_healthy_children_4-17_years_of_age#full-text
        # Adults: https://www.joms.org/article/0278-2391(89)90174-2/pdf
        # The rest: https://www-tandfonline-com.ez.statsbiblioteket.dk/doi/epdf/10.1179/crn.2007.031?needAccess=true

        data['openingmm_difference'] = 0.0  # Initialize with 0.0

        for index, row in data.iterrows():

            age, sex, openingmm = row['ageatvisitation'], row['sex'], row['openingmm']
            age_interval = next((interval for interval in age_intervals if interval[0] <= age <= interval[1]), None)

            if age_interval is not None and sex in sex_values:
                key = (age_interval, sex)
                if key in expected_values:
                    expected_value = expected_values[key]
                    difference = openingmm - expected_value
                    data.at[index, 'openingmm_difference'] = difference

        data.drop('openingmm', axis=1, inplace=True)
        data.rename(columns={'openingmm_difference': 'openingmm'}, inplace=True)
        data.loc[(data['openingmm'] > 30) | (data['openingmm'] < -30), 'openingmm'] = 0

        return data

class ProtrusionTransformer:

    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):

        age_intervals = [(0, 4), (4, 5), (5, 6), (6, 7), (7, 9), (9, 10), (10, 12), (12, 32)]
        sex_values = [0, 1]  # 0 = boy, 1 = girl
        expected_values = { ((0, 4), 0): 4, ((0, 4), 1): 4,
                            ((4, 5), 0): 3.61, ((4, 5), 1): 3.61,
                            ((5, 6), 0): 3.64, ((5, 6), 1): 3.64,
                            ((6, 7), 0): 3.42, ((6, 7), 1): 3.42,
                            ((7, 9), 0): 4.07, ((7, 9), 1): 4.07,
                            ((9, 10), 0): 5.21, ((9, 10), 1): 5.21,
                            ((10, 12), 0): 4.57, ((10, 12), 1): 4.57,
                            ((12, 32), 0): 4.57, ((10, 12), 1): 5.2,
                           }

        # https://www-tandfonline-com.ez.statsbiblioteket.dk/doi/epdf/10.1179/crn.2007.031?needAccess=true

        data['protrusionmm_difference'] = 0.0  # Initialize with 0.0

        for index, row in data.iterrows():

            age, sex, openingmm = row['ageatvisitation'], row['sex'], row['protrusionmm']
            age_interval = next((interval for interval in age_intervals if interval[0] <= age <= interval[1]), None)

            if age_interval is not None and sex in sex_values:
                key = (age_interval, sex)
                if key in expected_values:
                    expected_value = expected_values[key]
                    difference = openingmm - expected_value
                    data.at[index, 'protrusionmm_difference'] = difference

        data.drop('protrusionmm', axis=1, inplace=True)
        data.rename(columns={'protrusionmm_difference': 'protrusionmm'}, inplace=True)
        data.loc[(data['protrusionmm'] > 10), 'protrusionmm'] = 0

        return data