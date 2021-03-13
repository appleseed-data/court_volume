import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np

import os
import logging

from fbprophet import Prophet

class suppress_stdout_stderr(object):
    '''
    # https://github.com/facebook/prophet/issues/223
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


def predict_prophet_arrests(x, y, ds_col='ds', predict_col='yhat'):

    m = Prophet(uncertainty_samples=False)

    with suppress_stdout_stderr():
        m.fit(x)

    future_idx = y[[ds_col]].reset_index(drop=True)

    yhat = m.predict(future_idx)[[ds_col, predict_col]]

    df = pd.merge(y, yhat, left_on=ds_col, right_on=ds_col)

    df[predict_col] = np.where(df[predict_col] < 0, 0, df[predict_col])

    return df