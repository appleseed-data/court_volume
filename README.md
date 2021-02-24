# court_volume

An Analysis of Court Volume produced for [Chicago Appleseed](https://www.chicagoappleseed.org/).

## About

This project analyzes data about Court operations in Cook County during the COVID-19 Pandemic.

Central Question to be answered: How many cases would have been disposed during the pandemic that would have been resolved in a normal year?

Motivation: Identify the potential impact of diposing cases by estimating the court backlog.

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
df = pd.read_pickle('data/arrests_analysis_public.pickle')
```

4. From analyses/covid_cliff.py, records are aggregated and filtered for the scope of this analysis and then used to learn a model of time series trends.

5. Concept of Predictions: For court data, train sequence is a period of two years with daily counts of disposition categories; the following year is forecasted. For arrest data, train sequence is at a monthly-level within a range from 2014 to 2018 with prediction from 2018 to 2020.

6. The model forecasts (the predictions) are combined with actual data.

7. The disposition data forecast is a special case - predictions for 4 types of disposition events are run in paralell with multi-processing to improve performance but is not necessary. Disable pooling if pooling is not desired.

8. The source project terminates with an upload to MongoDB for data visualization, but further analysis can be continued without MongoDB.
