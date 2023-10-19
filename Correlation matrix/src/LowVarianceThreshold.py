import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold


def low_variance_threshold(data):

    data = data.drop(columns=['Unnamed: 0'])
    data = data.drop(columns=['involvementstatus'])
    data = data.drop(columns=['visitationdate'])
    data = data.drop(columns=['sex'])
    data = data.drop(columns=['type'])
    data = data.drop(columns=['studyid'])
    data = data.drop(columns=['tractionleft'])
    data = data.drop(columns=['tractionright'])

    thresholder = VarianceThreshold(threshold=.05)
    X_high_variance = thresholder.fit_transform(data)

    # Get the boolean mask of selected features (True for selected, False for removed)
    selected_features_mask = thresholder.get_support()

    # Get the names of the selected features
    selected_feature_names = data.columns[selected_features_mask]

    # Filter the original data to keep only the selected features
    X_high_variance = data[selected_feature_names]

    print("Done!")