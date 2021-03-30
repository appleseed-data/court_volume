from src.stages.run_court_forecasting import run_court_forecasting
from src.stages.run_arrest_forecasting import run_arrest_forecasting
from src.stages.run_data_merge import run_data_merge

def run_forecasting_project(data_folder):
    # organize predictions into a single df and export to disk
    disposition_data, raw_dispositions_predictions, df_backlog = run_court_forecasting(data_folder=data_folder)
    arrest_predictions = run_arrest_forecasting(data_folder)
    # merge court and arrest forecasting data into a single dataframe
    run_data_merge(data_folder=data_folder, disposition_data=disposition_data, arrest_predictions=arrest_predictions)