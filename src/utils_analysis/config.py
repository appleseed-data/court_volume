import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np

import os
import logging
import re

from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_percentage_error
from sklearn import preprocessing

# set scaler to a consistent range for comparison
# avoid division by zero issues by starting at 1 instead of 0
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

# breaking down MAPE for comparison
def MAPE(Y_actual,Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape

