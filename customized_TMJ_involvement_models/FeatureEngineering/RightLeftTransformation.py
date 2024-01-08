import pandas as pd
import re

def find_feature_pairs(data):

    feature_pairs = []

    for column1 in data.columns:
        pattern = re.compile(r'^(.*?)(right|left)(.*)$')
        match1 = pattern.match(column1)

        if match1 and column1 != 'laterotrusionrightmm' and column1 != 'laterotrusionleftmm':
            prefix, indicator, suffix = match1.groups()
            column2 = prefix + ("left" if indicator == "right" else "right") + suffix

            if column2 in data.columns:
                common_part = match1.group(1) + suffix

                if common_part not in [item for tuple in feature_pairs for item in tuple]:
                    feature_pairs.append((column1, column2, common_part))

    return feature_pairs

def getHighestSeverity(right, left):
    mergedList = []
    for i in range(len(right)):
        highVal = max(right[i], left[i])
        mergedList.append(highVal)

    return mergedList


class MergeFeatures:
    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):

        new_df = data

        feature_pair_list = find_feature_pairs(new_df)

        for feature_pair in feature_pair_list:

            # Merge features using your custom logic or function (getHighestSeverity)
            merged_list = getHighestSeverity(new_df[feature_pair[0]], new_df[feature_pair[1]])

            # Create a new column with the merged feature name
            new_df[feature_pair[2]] = merged_list

            # Drop the original features
            new_df.drop([feature_pair[0], feature_pair[1]], axis=1, inplace=True)

        print("Merging of features is done")

        return new_df


