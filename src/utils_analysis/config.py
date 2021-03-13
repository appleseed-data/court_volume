import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np

import os
import logging
import re

from sklearn.metrics import mean_squared_error


def eval_prophet(ytrue, yhat):

    df = pd.merge(ytrue, yhat, left_on='ds', right_on='ds')
    error = round(mean_squared_error(df['y'].values, df['yhat'].values),3)

    df = df.set_index('ds')
    plt.plot()
    sns.lineplot(data=df)
    plt.show()

    print(error)

    return df, error