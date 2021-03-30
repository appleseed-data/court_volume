from src.utils_data.pipelines_data import run_disposition_pipeline
from src.utils_forecasting.pipelines_forecasting import run_prophet_dispositions
from src.utils_analysis.pipelines_analysis import eval_prophet
from src.utils_data.config import etl_disposition_data, filter_court_backlog

from multiprocessing import Pool, cpu_count
import os
import logging

def run_court_forecasting(data_folder):
    ## get source data and prepare for forecasting
    # set full path to target data
    filename = 'dispositions.bz2'
    datafile = os.sep.join([data_folder, filename])
    # run disposition data processing pipeline and return a dataframe
    sequenced_disposition_data = run_disposition_pipeline(datafile, data_folder)

    ## make forecasts
    # predict for disposition data with pooling
    # configure optimal number of processes to run
    CPUs = cpu_count() // 2
    p = Pool(CPUs)
    disposition_predictions = list(p.imap(run_prophet_dispositions, sequenced_disposition_data))
    p.close()
    p.join()

    # organize predictions into a single df and export to disk
    disposition_data, raw_dispositions_predictions = etl_disposition_data(disposition_predictions, data_folder)

    # evaluate prediction results
    eval_prophet(raw_dispositions_predictions, data_type='disposition', data_folder=data_folder)

    # filter merged data to focus on estimated backlog time period
    df_backlog = filter_court_backlog(court_df=disposition_data)
    # export results to csv for quick reference
    filename = 'court_backlog.csv'
    data_file = os.sep.join([data_folder, filename])
    df_backlog.to_csv(data_file, index=False)
    logging.info(f'main() Wrote to disk {data_file}')

    return disposition_data, raw_dispositions_predictions, df_backlog