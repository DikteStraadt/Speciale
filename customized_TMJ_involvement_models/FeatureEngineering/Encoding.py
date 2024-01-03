from Utils import Report as r
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
#from tensorflow.keras.layers import Input, Embedding, Concatenate, Dense, Reshape
#from tensorflow.keras.models import Model
#from tensorflow.keras.optimizers import SGD
#from tensorflow.keras.callbacks import EarlyStopping
#from tensorflow.keras.utils import plot_model
#from IPython.display import Image
#from matplotlib import pyplot
#from keras import backend as K

class OneHotEncode:

    def __init__(self, columns_to_encode):
        self.columns_to_encode = columns_to_encode

    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):

        new_df = data

        return new_df




