from utils.pipelines_data import *
from analyses.covid_cliff import *
from utils.config import Columns
import logging
logging.basicConfig(level=logging.INFO)
import os

from multiprocessing import Pool, cpu_count
# deactivate current: conda deactivate
# delete conda env: conda remove --name court_volume --all
# restore conda env: conda env create -f environment.yml
# activate new env: conda activate court_volume
# testing code rebase
if __name__ == '__main__':
    # configure optimal number of processes to run
    CPUs = cpu_count() // 2
    # configure global variable to reference common strings and structures
    c = Columns()

    ## get source data and prepare for forecasting
    # set path to data folder
    data_folder = os.sep.join([os.environ['PWD'], 'src', 'data'])
    # set full path to target data
    filename = 'dispositions.bz2'
    datafile = os.sep.join([data_folder, filename])
    # run disposition data processing pipeline and return a dataframe
    sequenced_disposition_data = run_disposition_pipeline(datafile, data_folder)

    # set full path to target data
    filename = 'arrests_analysis_public.bz2'
    datafile = os.sep.join([data_folder, filename])
    # run arrests data processing pipeline and return a dataframe
    sequenced_arrest_data = run_arrests_pipeline(datafile, data_folder)

    ## make forecasts
    # predict for disposition data with pooling
    p = Pool(CPUs)
    disposition_predictions = list(p.imap(run_prophet_dispo, sequenced_disposition_data))
    p.close()
    p.join()

    # organize predictions into a single df and export to disk
    disposition_data = export_disposition_data(disposition_predictions, data_folder)

    # predict for arrest data without pooling
    # export for arrests is embedded into helper functions
    arrest_predictions = predict_arrest_data(sequenced_arrest_data, data_folder)

    ## merge data for for analysis
    # combine actual and predicted for each of court and arrest data
    df_merged = merge_dispositions_arrests(disposition_data, arrest_predictions)
    # set full path to target data
    filename = 'arrest_volume.csv'
    datafile = os.sep.join([data_folder, filename])
    # export results to csv for quick reference
    df_merged.to_csv(datafile, index=False)

    # filter merged data to focus on estimated backlog time period
    df_backlog = filter_court_backlog(court_df=disposition_data)
    # export results to csv for quick reference
    filename = 'court_backlog.csv'
    datafile = os.sep.join([data_folder, filename])
    df_backlog.to_csv(datafile, index=False)

    ## upload to online database (only if configured)
    run_mongo = False

    if run_mongo == True:
        # the mongo pipeline is only available if env and mongo account are established
        run_mongo_pipeline(df_merged, collection_name='court_arrest_volumes')
        run_mongo_pipeline(df_backlog, collection_name='court_backlog_estimate')







