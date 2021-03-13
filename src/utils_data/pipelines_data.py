from utils_data.config import *

def run_disposition_pipeline(filename, data_folder):
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
           .pipe(parse_dates)
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
    return x

def run_arrests_pipeline(filename, data_folder):
    # return a prepared dataset of arrests from compressed file
    # the pipeline is shorter than dispositions because it is pre-processed from another project
    df = get_source_file(filename)
    x = (df.pipe(prep_arrest_data_for_prophet, data_folder=data_folder)
         )
    # return x-> a sequenced data struct of df for prophet prediction
    return x

def run_mongo_pipeline(df, collection_name):
    # insert/update database
    MakeMongo().insert_df(collection=collection_name, df=df)