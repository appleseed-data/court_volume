from src.stages.run_court_forecasting import run_court_forecasting
from src.stages.run_arrest_forecasting import run_arrest_forecasting
from src.stages.run_data_merge import run_data_merge
from src.stages.run_court_stats import run_court_stats

import time
import os

time_string = time.strftime("%Y%m%d-%H%M%S")

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',datefmt='%m/%d/%Y %I:%M:%S %p')


def run_forecasting_project():
    # organize predictions into a single df and export to disk
    disposition_data, raw_dispositions_predictions, df_backlog = run_court_forecasting(data_folder=data_folder)
    arrest_predictions = run_arrest_forecasting(data_folder)
    # merge court and arrest forecasting data into a single dataframe
    run_data_merge(data_folder=data_folder, disposition_data=disposition_data, arrest_predictions=arrest_predictions)


if __name__ == '__main__':
    ## get source data and prepare for forecasting
    # set path to data folder
    data_folder = os.sep.join([os.environ['PWD'], 'data'])
    figs_folder = os.sep.join([os.environ['PWD'], 'figures'])

    run_court_stats(data_folder=data_folder, figs_folder=figs_folder)
