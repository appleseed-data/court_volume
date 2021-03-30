from src.utils_data.pipelines_data import run_arrests_pipeline
from src.utils_forecasting.pipelines_forecasting import run_prophet_arrests
from src.utils_data.config import get_git_pickle, arrest_data_path

import logging
import os

def run_arrest_forecasting(data_folder):
    # get prepared arrest data file
    arrest_df = get_git_pickle(arrest_data_path)
    # run arrests data processing pipeline and return a dataframe
    sequenced_arrest_data = run_arrests_pipeline(data_folder=data_folder, df=arrest_df)
    # predict for arrest data and export results
    arrest_predictions = run_prophet_arrests(sequenced_arrest_data, data_folder)

    return arrest_predictions