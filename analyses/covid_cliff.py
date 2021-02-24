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

restricted_list = ['Death', 'Mental Health', 'Other']

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

    data = train[0]
    x_i, y_i = data

    results = []

    for idx, x in enumerate(x_i):
        y = y_i[idx]

        m = Prophet(uncertainty_samples=False)

        with suppress_stdout_stderr():
            m.fit(x)

        future_idx = y[[ds_col]].reset_index(drop=True)

        yhat = m.predict(future_idx)[[ds_col, predict_col]]

        df = pd.merge(y, yhat, left_on=ds_col, right_on=ds_col)

        df[predict_col] = np.where(df[predict_col] < 0, 0, df[predict_col])

        results.append(df)

    return results

def decimal_str(x: float, decimals: int = 10) -> str:
    return format(x, f".{decimals}f").lstrip().rstrip('0')

def prep_prophet(x):
    key_map = {c.disposition_date: 'ds', 'count': 'y'}

    df = x.rename(columns=key_map)
    df = df.groupby(c.charge_disposition_cat)

    data_list = []

    for name, group in df:

        year_idx_start = 2011
        window_size = 2
        step_size = 1
        year_idx_max = 2021
        steps = list(range(year_idx_start, year_idx_max+1))

        x_i = []
        y_i = []

        df = group.copy()

        for step in steps:
            if step + 3 > max(steps):
                break

            X = df[(df['ds'].dt.year >= step) & (df['ds'].dt.year <= step + window_size)].reset_index(drop=True)
            Y = df[(df['ds'].dt.year == step + window_size + step_size)].reset_index(drop=True)

            x_i.append(X)
            y_i.append(Y)

        data = (x_i, y_i)

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
    df = ov1_disposition(x)
    df = df[~(df[c.charge_disposition_cat].isin(restricted_list))].copy()
    df[c.charge_disposition_cat] = df[c.charge_disposition_cat].cat.remove_unused_categories()
    df = df[[c.disposition_date, c.charge_disposition_cat, c.case_length]]
    df.to_pickle('data/covid_cliff_disposition_data.pickle')
    df.to_csv('data/covid_cliff_disposition_data.csv')

def predict_disposition_data():
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
        , 'y': 'case_count'
        , 'yhat': 'predicted_case_count'})

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
    df = df[df['charge_1_type'] == "F"].copy()

    df['date'] = pd.to_datetime(df['arrest_year'].astype(str) + '-' + df['arrest_month'].astype(str) + '-01')
    df = df[['date', 'charge_1_class']].groupby([pd.Grouper(key='date', freq='M')]).agg('count').reset_index()
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


def covid_cliff_analysis():
    court_df = pd.read_pickle('data/covid_cliff_court_data_predicted.pickle')
    arrest_df = pd.read_pickle('data/covid_cliff_arrest_data_predicted.pickle')
    
    df = pd.merge(court_df, arrest_df, left_on=c.disposition_date, right_on='arrest_date')

    df = df.rename(columns={c.disposition_date:'date'})

    df = df.drop(columns=['arrest_date'])

    court_df = court_df.set_index(c.disposition_date)
    arrest_df = arrest_df.set_index('arrest_date')

    plt.figure()
    sns.lineplot(data=court_df)
    sns.lineplot(data=arrest_df)
    plt.show()

    return df

