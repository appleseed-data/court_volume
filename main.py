from src.stages.run_court_stats import run_court_stats
from src.stages.run_forecasting_project import run_forecasting_project

import time
import os

time_string = time.strftime("%Y%m%d-%H%M%S")

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',datefmt='%m/%d/%Y %I:%M:%S %p')



if __name__ == '__main__':
    ## get source data and prepare for forecasting
    # set path to data folder
    data_folder = os.sep.join([os.environ['PWD'], 'data'])
    figs_folder = os.sep.join([os.environ['PWD'], 'figures'])

    run_court_stats(data_folder=data_folder, figs_folder=figs_folder)

    # uncomment next line to run forecasting project
    # run_forecasting_project(data_folder)
