from src.utils_data.pipelines_data import run_court_stats_pipeline
from src.utils_analysis.pipelines_analysis import eval_court_stats, eval_clearance_rates, eval_pending_surplus

import pandas as pd
import os

def run_court_stats(data_folder, figs_folder, filename='court_stats.csv'):

    df, clearance_rates, pending_surplus = run_court_stats_pipeline(data_folder=data_folder, filename=filename)

    eval_court_stats(df=df, clearance_rates=clearance_rates, figs_folder=figs_folder)

    eval_clearance_rates(df=clearance_rates)

    eval_pending_surplus(df=pending_surplus, figs_folder=figs_folder)