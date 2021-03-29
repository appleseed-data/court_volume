from src.utils_data.pipelines_data import run_disposition_pipeline, run_arrests_pipeline, run_mongo_pipeline
from src.utils_forecasting.pipelines_forecasting import run_prophet_dispositions, run_prophet_arrests, run_postpredict_dispositions
from src.utils_analysis.pipelines_analysis import eval_prophet
from src.utils_data.config import Columns, etl_disposition_data, merge_dispositions_arrests, filter_court_backlog
from io import BytesIO, StringIO
import pickle
import requests
import joblib

import time
import os

time_string = time.strftime("%Y%m%d-%H%M%S")

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',datefmt='%m/%d/%Y %I:%M:%S %p')
from multiprocessing import Pool, cpu_count
import pandas as pd
import pickle
import bz2

if __name__ == '__main__':

    # configure logging
    # logs_folder = os.sep.join([os.environ['PWD'], 'logs'])
    # if not os.path.exists(logs_folder):
    #     os.makedirs(logs_folder)
    #
    # logs_filename = f'{time_string}_log.log'
    # logs_file = os.sep.join([logs_folder, logs_filename])
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',datefmt='%m/%d/%Y %I:%M:%S %p')
    # logging.basicConfig(filename=logs_file, level=logging.INFO, format='%(asctime)s %(message)s',
    #                     datefmt='%m/%d/%Y %I:%M:%S %p')

    # configure optimal number of processes to run
    CPUs = cpu_count() // 2
    # configure global variable to reference common strings and structures
    c = Columns()

    ## get source data and prepare for forecasting
    # set path to data folder
    data_folder = os.sep.join([os.environ['PWD'], 'data'])
    # set full path to target data
    filename = 'dispositions.bz2'
    datafile = os.sep.join([data_folder, filename])
    # run disposition data processing pipeline and return a dataframe
    sequenced_disposition_data = run_disposition_pipeline(datafile, data_folder)

    # filename = 'arrests_analysis_public.bz2'
    filename = 'arrests_redacted_classified.bz2'
    datafile = os.sep.join([data_folder, filename])
    # run arrests data processing pipeline and return a dataframe
    sequenced_arrest_data = run_arrests_pipeline(datafile, data_folder)

    ## make forecasts
    # predict for disposition data with pooling
    p = Pool(CPUs)
    disposition_predictions = list(p.imap(run_prophet_dispositions, sequenced_disposition_data))
    p.close()
    p.join()

    # organize predictions into a single df and export to disk
    disposition_data, raw_dispositions_predictions = etl_disposition_data(disposition_predictions, data_folder)

    # evaluate prediction results
    eval_prophet(raw_dispositions_predictions, data_type='disposition', data_folder=data_folder)

    # predict for arrest data without pooling
    # export for arrests is embedded into helper functions
    arrest_predictions = run_prophet_arrests(sequenced_arrest_data, data_folder)

    ## merge data for for analysis
    # combine actual and predicted for each of court and arrest data
    df_merged = merge_dispositions_arrests(disposition_data, arrest_predictions)
    # set full path to target data
    filename = 'arrest_volume.csv'
    data_file = os.sep.join([data_folder, filename])
    # export results to csv for quick reference
    df_merged.to_csv(data_file, index=False)
    logging.info(f'main() Wrote to disk {data_file}')

    # filter merged data to focus on estimated backlog time period
    df_backlog = filter_court_backlog(court_df=disposition_data)
    # export results to csv for quick reference
    filename = 'court_backlog.csv'
    data_file = os.sep.join([data_folder, filename])
    df_backlog.to_csv(data_file, index=False)
    logging.info(f'main() Wrote to disk {data_file}')

    ## upload to online database (only if configured)
    run_mongo = False

    if run_mongo == True:
        # the mongo pipeline is only available if env and mongo account are established
        run_mongo_pipeline(df_merged, collection_name='court_arrest_volumes')
        run_mongo_pipeline(df_backlog, collection_name='court_backlog_estimate')
