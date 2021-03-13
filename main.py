from utils.pipelines_data import *
from analyses.covid_cliff import *
from utils.config import Columns
import logging
#TODO replace print statements with logging

from multiprocessing import Pool, cpu_count
# deactivate current: conda deactivate
# delete conda env: conda remove --name court_volume --all
# restore conda env: conda env create -f environment.yml
# activate new env: conda activate court_volume
# testing code rebase 
if __name__ == '__main__':
    CPUs = cpu_count() // 2
    c = Columns()

    dispositions = run_disposition_pipeline()
    prep_disposition_data(dispositions)
    arrests = run_arrests_pipeline()
    prep_arrest_data(arrests)

    dispo_data = predict_disposition_data()
    arrest_data = predict_arrest_data()

    p = Pool(CPUs)
    dispo_predictions = list(p.imap(run_prophet_dispo, dispo_data))
    p.close()
    p.join()

    export_disposition_data(dispo_predictions)

    df_mongo = prep_analysis_for_mongo()
    # run_mongo_pipeline(df_mongo)
    df_mongo.to_csv('data/arrest_volume.csv', index=False)

    df_backlog = estimate_court_backlog()
    # run_mongo_pipeline(df, collection_name='court_backlog_estimate')
    df_backlog.to_csv('data/court_backlog.csv', index=False)






