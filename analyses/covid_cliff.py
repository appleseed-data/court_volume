import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error
from fbprophet import Prophet

from utils.config import Columns
from utils.config import ov1_disposition

c = Columns()

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


def run_prophet_dispo(train, ds_col='ds', predict_col='yhat'):
    """
    :param train: a 2-tuple of (data, data category), idx 0 comprising a pandas dataframe, idx 1 a string
    :param ds_col: default to 'ds' per facebook prophet api
    :param predict_col: default to 'yhat' per facebook prophet api
    :return:
    """

    # data, a 2-tuple within train
    data = train[0]
    # x_i is the train data y_i is the target data
    x_i, y_i = data

    results = []

    for idx, x in enumerate(x_i):
        # store current target by indexing y_i with x_i
        y = y_i[idx]

        m = Prophet(uncertainty_samples=False)

        # fit model to current x and supress annoying output
        with suppress_stdout_stderr():
            m.fit(x)

        # make a dataframe for future forecast based on current y index
        future_idx = y[[ds_col]].reset_index(drop=True)

        # return a prediction based on the index and slice only desired cols
        yhat = m.predict(future_idx)[[ds_col, predict_col]]

        # unify targets and predictions into a single dataframe
        df = pd.merge(y, yhat, left_on=ds_col, right_on=ds_col)

        # intervene with predictions, any negative number is 0
        df[predict_col] = np.where(df[predict_col] < 0, 0, df[predict_col])

        # capture each iteration in a list of dataframes
        results.append(df)

    return results

def decimal_str(x: float, decimals: int = 10) -> str:
    return format(x, f".{decimals}f").lstrip().rstrip('0')

def prep_prophet(x):
    """
    :param x: a dataframe having data for train and test (everything)
    :return: a list of 2-tuples having (dataframe, string) as data and data label for prophet
    """

    # map for col names per facebook api
    key_map = {c.disposition_date: 'ds', 'count': 'y'}

    # apply col name remap
    df = x.rename(columns=key_map)

    # group data by desired category
    df = df.groupby(c.charge_disposition_cat)

    # init empty list for output
    data_list = []

    for name, group in df:
        # loop through each data from of group
        # hard-coded values for window and step size
        year_idx_start = 2011
        window_size = 2
        step_size = 1
        year_idx_max = 2021
        steps = list(range(year_idx_start, year_idx_max+1))

        x_i = []
        y_i = []

        df = group.copy()

        # within each loop, extract data frame into train and test pairs
        for step in steps:
            # train on two years, predict the third, break when year 3 is at end of index
            if step + 3 > max(steps):
                break

            # in the first loop, 2011 < x.dt.year < 2013 and y.dt.year == 2014
            X = df[(df['ds'].dt.year >= step) & (df['ds'].dt.year <= step + window_size)].reset_index(drop=True)
            Y = df[(df['ds'].dt.year == step + window_size + step_size)].reset_index(drop=True)

            # save sequenced dataframes in lists
            x_i.append(X)
            y_i.append(Y)

        # save data as 2-tuple pairs
        data = (x_i, y_i)

        # accumulate data as a list of tuples
        data_list.append((data, name))

    return data_list

def eval_prophet(ytrue, yhat):

    df = pd.merge(ytrue, yhat, left_on='ds', right_on='ds')
    error = round(mean_squared_error(df['y'].values, df['yhat'].values),3)

    df = df.set_index('ds')
    plt.plot()
    sns.lineplot(data=df)
    plt.show()

    print(error)

    return df, error


def prep_disposition_data(x):
    restricted_list = ['Death', 'Mental Health', 'Other']

    df = ov1_disposition(x)
    df = df[~(df[c.charge_disposition_cat].isin(restricted_list))].copy()
    df[c.charge_disposition_cat] = df[c.charge_disposition_cat].cat.remove_unused_categories()
    df = df[[c.disposition_date, c.charge_disposition_cat, c.case_length]]

    df.to_pickle('data/covid_cliff_disposition_data.pickle')
    df.to_csv('data/covid_cliff_disposition_data.csv')


def predict_disposition_data():
    print('------ Ran court forecast')
    df = pd.read_pickle('data/covid_cliff_disposition_data.pickle')

    grouper = pd.Grouper(key=c.disposition_date, freq='D')

    df = df.groupby([c.charge_disposition_cat, grouper])[c.charge_disposition_cat].agg('count')
    df = df.reset_index(name='count')
    df = df.sort_values(by=[c.disposition_date])
    df = df.reset_index(drop=True)

    data = prep_prophet(df)

    return data


def export_disposition_data(predictions):
    df = pd.concat([pd.concat(i) for i in predictions])
    df = df.rename(columns={'ds': c.disposition_date
                            ,'y': 'case_count'
                            ,'yhat': 'predicted_case_count'})

    df = df[[c.disposition_date, 'case_count', c.charge_disposition_cat, 'predicted_case_count']]

    grouper = pd.Grouper(key=c.disposition_date, freq='M')

    df = df.groupby([c.charge_disposition_cat, grouper]).agg({'case_count':'sum','predicted_case_count':'sum'})

    df = df.reset_index()

    df = df.sort_values(by=[c.disposition_date])
    df = df.reset_index(drop=True)

    df = df[[c.disposition_date, 'case_count', c.charge_disposition_cat, 'predicted_case_count']]

    df.to_csv('data/covid_cliff_court_data_predicted.csv')
    df.to_pickle('data/covid_cliff_court_data_predicted.pickle')


def prep_arrest_data(df):
    felony_flag = "F"
    frequency = "M"

    df = df[df['charge_1_type'] == felony_flag].copy()

    df['date'] = pd.to_datetime(df['arrest_year'].astype(str) + '-' + df['arrest_month'].astype(str) + '-01')
    df = df[['date', 'charge_1_class']].groupby([pd.Grouper(key='date', freq=frequency)]).agg('count').reset_index()
    df = df.sort_values('date')
    df['type'] = 'Felony Arrest'
    df.rename(columns={'charge_1_class': 'count'}, inplace=True)

    df.to_pickle('data/covid_cliff_arrest_data.pickle')
    df.to_csv('data/covid_cliff_arrest_data.csv')


def run_prophet_arrest(x, y, ds_col='ds', predict_col='yhat'):

    m = Prophet(uncertainty_samples=False)

    with suppress_stdout_stderr():
        m.fit(x)

    future_idx = y[[ds_col]].reset_index(drop=True)

    yhat = m.predict(future_idx)[[ds_col, predict_col]]

    df = pd.merge(y, yhat, left_on=ds_col, right_on=ds_col)

    df[predict_col] = np.where(df[predict_col] < 0, 0, df[predict_col])

    return df


def predict_arrest_data():
    print('------ Ran arrest forecast')
    df = pd.read_pickle('data/covid_cliff_arrest_data.pickle')

    df = df.rename(columns={'date':'ds'
                            ,'count':'y'})

    x = df[df['ds'].dt.year < 2019]
    y = df[df['ds'].dt.year >= 2014]

    results = run_prophet_arrest(x,y)

    df = results.rename(columns={'ds':'arrest_date'
                                ,'y':'arrest_count'
                                ,'yhat':'predicted_arrest_count'
                                ,'type':'arrest_type' })

    df.to_csv('data/covid_cliff_arrest_data_predicted.csv')
    df.to_pickle('data/covid_cliff_arrest_data_predicted.pickle')

    return df


def prep_analysis_for_mongo():
    court_df = pd.read_pickle('data/covid_cliff_court_data_predicted.pickle')
    arrest_df = pd.read_pickle('data/covid_cliff_arrest_data_predicted.pickle')
    
    df = pd.merge(court_df, arrest_df, left_on=c.disposition_date, right_on='arrest_date')

    df = df.rename(columns={c.disposition_date:'date'})

    df = df.drop(columns=['arrest_date'])

    df['case_count'] = list(zip(df[c.charge_disposition_cat],df['case_count']))
    df['predicted_case_count'] = list(zip(df[c.charge_disposition_cat],df['predicted_case_count']))

    df['arrest_count'] = list(zip(df['arrest_type'], df['arrest_count']))
    df['predicted_arrest_count'] = list(zip(df['arrest_type'], df['predicted_arrest_count']))

    df.drop(columns=['arrest_type', c.charge_disposition_cat], inplace=True)

    return df


def estimate_court_backlog():
    print('------ Generated backlog estimates')
    court_df = pd.read_pickle('data/covid_cliff_court_data_predicted.pickle')
    arrest_df = pd.read_pickle('data/covid_cliff_arrest_data_predicted.pickle')

    df = court_df[court_df[c.disposition_date] > pd.to_datetime("2020-02-01")].copy()
    df['backlog'] = df['predicted_case_count'] - df['case_count']

    df = df.reset_index(drop=True)

    return df






