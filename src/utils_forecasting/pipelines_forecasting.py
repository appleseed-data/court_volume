from utils_forecasting.config import *

def run_prophet_dispositions(train, ds_col='ds', predict_col='yhat'):
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

def run_prophet_arrests(df
                      , data_folder
                      , csv_filename='covid_cliff_arrest_data_predicted.csv'
                      , pickle_filename='covid_cliff_arrest_data_predicted.pickle'
                        ):

    print('------ Ran arrest forecast')
    # df = pd.read_pickle('data/covid_cliff_arrest_data.pickle')

    df = df.rename(columns={'date':'ds'
                            ,'count':'y'})

    x = df[df['ds'].dt.year < 2019]
    y = df[df['ds'].dt.year >= 2014]

    results = predict_prophet_arrests(x,y)

    df = results.rename(columns={'ds':'arrest_date'
                                ,'y':'arrest_count'
                                ,'yhat':'predicted_arrest_count'
                                ,'type':'arrest_type' })

    data_file = os.sep.join([data_folder, csv_filename])
    df.to_csv(data_file, index=False)
    data_file = os.sep.join([data_folder, pickle_filename])
    df.to_pickle(data_file)

    return df

