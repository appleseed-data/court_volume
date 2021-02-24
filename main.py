from utils.pipelines_data import *
from analyses.covid_cliff import *
from utils.config import Columns

from multiprocessing import Pool, cpu_count

CPUs = cpu_count() - 2
c = Columns()

if __name__ == '__main__':

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

    df_backlog = estimate_court_backlog()
    # run_mongo_pipeline(df, collection_name='court_backlog_estimate')






