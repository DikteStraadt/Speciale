import numpy as np
import pandas as pd

# LAG FEATURES (see GitHub)

class LagFeaturesForVisitations:

    def __init__(self, lag_features):
        self.lag_features = lag_features

    def fit(self, visitations_3D, y=None):
        return self

    def transform(self, visitations_3D, y=None):

        # LAG FEATURES
        transformed_data = visitations_3D

        return transformed_data