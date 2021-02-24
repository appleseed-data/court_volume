# court_volume

An Analysis of Court Volume produced for [Chicago Appleseed](https://www.chicagoappleseed.org/).

## About

This project analyzes data about Court operations in Cook County during the COVID-19 Pandemic.

Central Question to be answered: How many cases would have been disposed during the pandemic that would have been resolved in a normal year?

Motivation: Identify the potential impact of disposing cases by estimating the court backlog.

## Core Concepts

1. Estimate how many cases "would have been" by forecasting the volume of disposed cases into the pandemic months.

2. Train a forecasting model (Facebook Prohpet) on court and arrest volume data from 2014 to 2018, then predict next in sequence.

3. Calculate the difference between actual disposed cases and forecased cases of the same type.

4. Definition: A disposed case in this analysis is one of Dismissal, Guilty Plea, Not Guilty, or Guilty at Trial.

## Data Source

Cook County [Open Data Portal](https://datacatalog.cookcountyil.gov/)

## Data Visuals

Data was processed with Python, Pandas, and [Facebook Prophet](https://facebook.github.io/prophet/). The data was then uploaded to MongoDB where the following visualizations are hosted.

One way to think about the court backlog is to take the difference between the forecast and the actual counts. If the forecast is accurate and the actual counts reflect a reduced operational tempo, then the difference represents 'what would have been'.

[![backlog](https://github.com/justinhchae/court_volume/blob/main/figures/Cook%20County%20Court%20Backlog%20During%20the%20Pandemic.png)](https://charts.mongodb.com/charts-court_volume-nmlff/embed/charts?id=41d4d178-fcb6-4e0e-a4bf-687239cd72cc&theme=light)

How is the forecast created? Is it accurate? Forecasting is notoriously difficult and is never perfect but we can make reasonable attempts at it. We learned a model of trends in both arrests and court volumes with Facebook Prophet. Then, using the model of past data, we forecasted future trends for periods before and during the pandemic.

As the visualization shows, the model forecasts a trend line that is similar to the actual counts for historic periods. With an accurate model, going into the pandemic months, we can see that the trend indicates a count of cases that would have been disposed.

[![timeline](https://github.com/justinhchae/court_volume/blob/main/figures/Cook%20County%20Court%20Dispositions%20and%20Felony%20Arrest%20Volumes.png)](https://charts.mongodb.com/charts-court_volume-nmlff/embed/charts?id=5efdfb3e-a237-41e6-b353-0f697aa0ec2e&theme=light)

## Disclaimer

Disclaimer: "This site provides applications using data that has been modified for use from its original source, www.cityofchicago.org, the official website of the City of Chicago.  The City of Chicago makes no claims as to the content, accuracy, timeliness, or completeness of any of the data provided at this site.  The data provided at this site is subject to change at any time.  It is understood that the data provided at this site is being used at one’s own risk."

## Code

A few high-level notes about the scripts used in this analysis.

1. Helper and data cleaning functions in utils/config.py
2. Functions packaged into pipelines in utils/pipelines_data.py
3. Disposition Data pipeline starts with source file, dispositions.bz2 - this file is a compressed pandas pickle file which can be read with Python with

```python
import pandas as pd

df = pd.read_pickle('data/dispositions.bz2')
```

... or for arrest data

```python
df = pd.read_pickle('data/arrests_analysis_public.bz2')
```

4. From analyses/covid_cliff.py, records are aggregated and filtered for the scope of this analysis and then used to learn a model of time series trends.

5. Concept of Predictions: For court data, train sequence is a period of two years with daily counts of disposition categories; the following year is forecasted. For arrest data, train sequence is at a monthly-level within a range from 2014 to 2018 with prediction from 2018 to 2020.

6. The model forecasts (the predictions) are combined with actual data. The primary forecasting script is provided in this readme for quick reference.

```python
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
```

7. The disposition data forecast is a special case - predictions for 4 types of disposition events are run in paralell with multi-processing to improve performance but is not necessary. Disable pooling if pooling is not desired.

8. The source project terminates with an upload to MongoDB for data visualization, but further analysis can be continued without MongoDB.

## How to Get Started With this Code

1. Will most likely work best - create a new conda environment from the environment.yml file

```terminal
conda env create -f environment.yml
```

2. Should work, but I had some issues with starting from requirements.

```terminal
pip install -r requirements.txt
```

4. See the [conda docs](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)


5. Run main.py which will read the source file, process it, and generate forecasts. You know it worked if the script generates .pickle files and says that the forecasts have finished running. 
