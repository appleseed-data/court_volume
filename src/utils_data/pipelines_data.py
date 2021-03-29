from src.utils_data.config import *

def run_disposition_pipeline(filename, data_folder):
    logging.info('run_disposition_pipeline() Starting data pipeline for dispositions')
    # get source file, hard coded to the latest dispositions file (compressed as bz2)
    df = get_source_file(filename)

    # known cols to dtype as categorical
    cols_to_categorize = [c.charge_id
                        , c.charge_version_id
                        , c.disposition_charged_chapter
                        , c.disposition_charged_act
                        , c.disposition_charged_section
                        , c.disposition_charged_aoic
                        , c.disposition_charged_offense_title
                        , c.charge_disposition
                        , c.charge_disposition_reason
                        , c.incident_city
                        , c.law_enforcement_agency
                        , c.law_enforcement_unit
                        , c.felony_review_result]

    # run data through cleaning and prep pipeline
    x = (df.pipe(parse_cols)
           .pipe(parse_date_cols)
           .pipe(remove_conversion_records, col_name=c.offense_category)
           .pipe(remove_conversion_records, col_name=c.disposition_court_name)
           .pipe(remove_conversion_records, col_name=c.disposition_court_facility)
           .pipe(impute_dates, col1=c.disposition_date, col2=c.received_date, date_type='disposition')
           .pipe(map_disposition_categories, col1=c.charge_disposition)
           .pipe(typeas_string, cols=[c.case_id, c.case_participant_id])
           .pipe(classer, col_name=c.disposition_charged_class)
           .pipe(classer, col_name=c.updated_offense_category)
           .pipe(classer, col_name=c.gender)
           .pipe(classer, col_name=c.race)
           .pipe(classer, col_name=cols_to_categorize)
           .pipe(make_caselen, col1=c.received_date, col2=c.disposition_date)
           .pipe(reduce_precision)
           .pipe(filter_disposition_data, data_folder=data_folder)
           .pipe(prep_disposition_data_for_prophet)
        )
    # return x-> a sequenced data struct of dfs for prophet prediction
    logging.info('run_disposition_pipeline() Completed data pipeline for dispositions')
    return x

def run_arrests_pipeline(data_folder, filename=None, df=None):
    logging.info('run_arrests_pipeline() Starting data pipeline for arrests')
    # return a prepared dataset of arrests from compressed file
    # the pipeline is shorter than dispositions because it is pre-processed from another project
    if df is not None:
        df = df
        filename = None

    if filename is not None:
        df = get_source_file(filename)

    x = (df.pipe(prep_arrest_data_for_prophet, data_folder=data_folder))

    logging.info('run_arrests_pipeline() Completed data pipeline for arrests')
    # return x-> a sequenced data struct of df for prophet prediction
    return x

def run_mongo_pipeline(df, collection_name):
    logging.info('run_mongo_pipeline() Starting data pipeline for MongoDB upload')
    # insert/update database
    MakeMongo().insert_df(collection=collection_name, df=df)