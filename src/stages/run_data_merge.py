from src.utils_data.config import merge_dispositions_arrests

import os
import logging

def run_data_merge(data_folder, disposition_data, arrest_predictions):
    ## merge data for for analysis
    # set full path to target data
    filename = 'arrest_volume.csv'
    data_file = os.sep.join([data_folder, filename])
    # export results to csv for quick reference
    # combine actual and predicted for each of court and arrest data
    df_merged = merge_dispositions_arrests(disposition_data, arrest_predictions)
    df_merged.to_csv(data_file, index=False)
    logging.info(f'main() Wrote to disk {data_file}')