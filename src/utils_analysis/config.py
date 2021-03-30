import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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

def make_clearance_figs(df, analysis_type):
    logging.info(f'Running analysis for {analysis_type}.')
    plt.figure(figsize=(10,6))
    sns.barplot(data=df, x='year-quarter', y='clearance_rate',  hue='organization')
    plt.title(f'{analysis_type} Analysis')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def make_surplus_figs(df, analysis_type, figs_folder, filename='fig_2_surplus.png', starting_date='2019-04', ending_date='2020-03'):
    logging.info(f'Running analysis for {analysis_type}.')
    plt.figure(figsize=(10,6))
    sns.barplot(data=df, x='category', hue='organization', y='surplus_count')
    plt.title(f'Court Case Surplus Analysis from Q4 2019 to Q3 2020')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel('Charge Category')
    plt.ylabel('Surplus Count')
    plt.tight_layout()
    data_path = os.sep.join([figs_folder, filename])
    plt.savefig(data_path)
    plt.show()

def make_stats_figs(df, clearance_rates, analysis_type, figs_folder, filename='fig_1_overall.png'):
    logging.info(f'Running analysis for {analysis_type}.')

    col_order = df['year-quarter'].unique().tolist()
    col_order.sort()

    row_order = df['organization'].unique().tolist()
    row_order.sort()

    df = df.drop(columns=['quarter', 'year'])

    df = df.rename(columns={'year-quarter':'quarter'})

    bar_order = df['category'].unique().tolist()
    bar_order.sort()

    def ant(x,y, **kwargs):
        year_quarter = x.values[0]
        organization = y.values[0]

        z = clearance_rates[clearance_rates['organization'] == organization]
        total_rate = z[(z['year-quarter'] == year_quarter) & (z['charge_class'] == 'T')]['clearance_rate']
        total_rate = total_rate.values[0]
        felony_rate = z[(z['year-quarter'] == year_quarter) & (z['charge_class'] == 'F')]['clearance_rate']
        felony_rate = felony_rate.values[0]
        # z = z[z['year-quarter']==year_quarter]['clearance_rate'].values[0]

        plt.annotate(f'Clearance Rates\nT :{total_rate}%\nF: {felony_rate}%', (0,75000))

    plt.figure()
    g = sns.FacetGrid(df
                      , col='quarter'
                      , col_order=col_order
                      , row='organization'
                      , row_order=row_order
                      , hue='category'
                      , margin_titles=True
                      , legend_out=False
                      )
    g.map(sns.barplot, 'category', 'total', order=bar_order, color='blue')
    g.map(sns.barplot, 'category', 'misdemeanor_count', order=bar_order, color='orange')
    g.map(sns.barplot, 'category', 'felony_count', order=bar_order, color='red')
    g.map(sns.barplot, 'category', 'dui_count', order=bar_order, color='green')
    g.map(ant, 'quarter', 'organization')
    g.set_xticklabels(rotation=30)
    g.set_ylabels('Count')
    g.fig.suptitle('Overall Court Statistics between Cook County and Downstate from Q4 2019 to Q3 2020')

    custom_lines = [Line2D([0], [0], color="blue", lw=4),
                    Line2D([0], [0], color="orange", lw=4),
                    Line2D([0], [0], color="red", lw=4),
                    Line2D([0], [0], color="green", lw=4)
                    ]

    plt.legend(custom_lines, ['Total', 'Misdemeanor', 'Felony', 'DUI'])
    plt.tight_layout()
    data_path = os.sep.join([figs_folder, filename])
    plt.savefig(data_path)
    plt.show()


