import numpy as np
import pandas as pd

# LAG FEATURES (see GitHub)

class LagFeaturesForVisitations:

    def __init__(self, previous_two_values):
        self.previous_two_values = previous_two_values

    def fit(self, visitations_3D, y=None):
        return self

    def transform(self, visitations_3D, y=None):

        # LAG FEATURES
        transformed_data = visitations_3D

        return transformed_data