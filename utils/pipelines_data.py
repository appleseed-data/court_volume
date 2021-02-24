from utils.config import *

def run_disposition_pipeline(filename='dispositions.bz2'):
    df = get_source_file(filename)
    df = parse_cols(df)
    df = parse_dates(df)

    df = remove_conversion_records(df, c.offense_category)
    df = remove_conversion_records(df, c.disposition_court_name)
    df = remove_conversion_records(df, c.disposition_court_facility)

    df = impute_dates(df, col1=c.disposition_date, col2=c.received_date, date_type='disposition')
    df = map_disposition_categories(df, c.charge_disposition)

    df = typeas_string(df, [c.case_id, c.case_participant_id])

    df = classer(df, c.disposition_charged_class)
    df = classer(df, c.updated_offense_category)
    df = classer(df, c.gender)
    df = classer(df, c.race)
    df = classer(df,[c.charge_id
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
                   , c.felony_review_result])

    df = make_caselen(df, c.received_date, c.disposition_date)
    df = reduce_precision(df)

    return df


def run_arrests_pipeline(filename='arrests_analysis_public.pickle'):
    df = get_source_file(filename)
    return df