import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np

import os
import logging
import re

from pymongo import MongoClient
from decouple import config

np_days = np.timedelta64(1, 'D')
np_hours = np.timedelta64(1, 'h')

import multiprocessing as mp
from tqdm import tqdm

from io import BytesIO
import requests
import joblib


arrest_data_path = "https://github.com/appleseed-data/arrest_clusters/blob/main/data/arrests_redacted_classified.bz2?raw=true"

def make_pending_surplus(df):
    temp = df[df['category'] == 'Pending'].reset_index(drop=True)

    temp = temp.groupby(['organization'])

    cols = ['felony_count', 'misdemeanor_count', 'dui_count', 'total']

    start_date = '2019-4'
    end_date = '2020-3'

    results = []
    for k, v in temp:

        starting_vals = v[v['year-quarter'] == start_date]
        ending_vals = v[v['year-quarter'] == end_date]


        for col in cols:
            starting_val = starting_vals[col].values[0]
            ending_val = ending_vals[col].values[0]
            surplus_value = ending_val - starting_val

            result = [k, col, surplus_value, start_date, end_date]
            results.append(result)

    results = pd.DataFrame(results, columns=['organization', 'category', 'surplus_count', 'starting_quarter', 'ending_quarter'])

    return results


def make_clearance_rates(df):
    temp = df.groupby(['organization', 'year-quarter'])

    clearance_rates = []

    for k, v in temp:
        row = list(k)
        x = v[['category', 'total']].set_index('category')
        demominator = x.loc['New/Reinstated'].values[0]
        nominator = x.loc['Disposed'].values[0]
        clearance_rate = (nominator / demominator) * 100
        clearance_rate = round(clearance_rate, 2)
        row.append(clearance_rate)
        row.append('T')
        clearance_rates.append(row)

    for k, v in temp:
        row = list(k)
        x = v[['category', 'felony_count']].set_index('category')
        demominator = x.loc['New/Reinstated'].values[0]
        nominator = x.loc['Disposed'].values[0]
        clearance_rate = (nominator / demominator) * 100
        clearance_rate = round(clearance_rate, 2)
        row.append(clearance_rate)
        row.append('F')
        clearance_rates.append(row)

    clearance_rates = pd.DataFrame(clearance_rates, columns=['organization', 'year-quarter', 'clearance_rate', 'charge_class'])

    return clearance_rates

def get_git_pickle(data_path):
    logging.info(f'Reading file from remote location at {data_path}')
    data_stream = BytesIO(requests.get(data_path).content)
    tgt_file = joblib.load(data_stream)

    logging.info(f'Read in target as a {type(tgt_file)} with len({len(tgt_file)})')

    return tgt_file

def to_dt(x):
    # a helper function for conversion to pandas date time
    # coerce errors here but currect them later
    x = pd.to_datetime(x, errors='coerce', infer_datetime_format=False)
    return x

def parse_date_cols(df, date_strings=('_date', 'date_')):
    logging.info('Date columns conversion to dtype datetime.')
    cols_to_convert = []
    col_names = []

    for col in df.columns:
        col_type = df[col].dtype
        if 'datetime' in col_type.name or any(i in col for i in date_strings):
            cols_to_convert.append(df[col])
            col_names.append(col)

    pbar = tqdm(cols_to_convert, desc='Running Datetime Conversion')

    CPUs = mp.cpu_count() // 2
    pool = mp.Pool(CPUs)
    results = list(pool.imap(to_dt, pbar))
    pool.close()
    pool.join()

    for result in results:
        df[result.name] = result.dt.tz_localize(tz=None)

    return df


def get_source_file(filename):
    """
    a convenience wrapper for reading dataframes
    """
    logging.info(f'get_source_file() Reading data from {filename}')
    if '.csv' in filename:
        df = pd.read_csv(filename, low_memory=False)
        return df
    elif '.pickle' in filename or '.bz2' in filename:
        df = pd.read_pickle(filename)
        return df

class Columns():
    """
    a custom class to call column names or data fields common to cook county court data
    """

    def __init__(self):
        self.act = 'act'
        self.age_at_incident = 'age_at_incident'
        self.aoic = 'aoic'
        self.arraignment_date = 'arraignment_date'
        self.arrest_date = 'arrest_date'
        self.bond_amount_current = 'bond_amount_current'
        self.bond_amount_initial = 'bond_amount_initial'
        self.bond_date_current = 'bond_date_current'
        self.bond_date_initial = 'bond_date_initial'
        self.bond_electroinic_monitor_flag_current = 'bond_electroinic_monitor_flag_current'
        self.bond_electronic_monitor_flag_initial = 'bond_electronic_monitor_flag_initial'
        self.bond_type_current = 'bond_type_current'
        self.bond_type_initial = 'bond_type_initial'
        self.case_id = 'case_id'
        self.case_length = 'case_length'
        self.case_participant_id = 'case_participant_id'
        self.chapter = 'chapter'
        self.charge_count = 'charge_count'
        self.charge_disposition = 'charge_disposition'
        self.charge_disposition_cat = 'charge_disposition_cat'
        self.charge_disposition_reason = 'charge_disposition_reason'
        self.charge_id = 'charge_id'
        self.charge_offense_title = 'charge_offense_title'
        self.charge_version_id = 'charge_version_id'
        self.charged_class_difference = 'charged_class_difference'
        self.charge_class = 'class'
        self.commitment_days = 'commitment_days'
        self.commitment_dollars = 'commitment_dollars'
        self.commitment_type = 'commitment_type'
        self.commitment_unit = 'commitment_unit'
        self.commitment_weight = 'commitment_weight'
        self.current_sentence_flag = 'current_sentence_flag'
        self.disposition_charged_act = 'disposition_charged_act'
        self.disposition_charged_aoic = 'disposition_charged_aoic'
        self.disposition_charged_chapter = 'disposition_charged_chapter'
        self.disposition_charged_class = 'disposition_charged_class'
        self.disposition_charged_offense_title = 'disposition_charged_offense_title'
        self.disposition_charged_section = 'disposition_charged_section'
        self.disposition_court_facility = 'disposition_court_facility'
        self.disposition_court_name = 'disposition_court_name'
        self.disposition_date = 'disposition_date'
        self.disposition_date_days_pending = 'disposition_date_days_pending'
        self.event = 'event'
        self.event_date = 'event_date'
        self.felony_review_date = 'felony_review_date'
        self.felony_review_result = 'felony_review_result'
        self.finding_no_probable_cause = 'finding_no_probable_cause'
        self.gender = 'gender'
        self.incident_begin_date = 'incident_begin_date'
        self.incident_city = 'incident_city'
        self.incident_end_date = 'incident_end_date'
        self.initial_charged_class = 'initial_charged_class'
        self.judge = 'judge'
        self.law_enforcement_agency = 'law_enforcement_agency'
        self.law_enforcement_unit = 'law_enforcement_unit'
        self.life_term = 'life_term'
        self.offense_category = 'offense_category'
        self.primary_charge_flag = 'primary_charge_flag'
        self.primary_charge_flag_init = 'primary_charge_flag_init'
        self.race = 'race'
        self.received_date = 'received_date'
        self.section = 'section'
        self.sentence_court_facility = 'sentence_court_facility'
        self.sentence_court_name = 'sentence_court_name'
        self.sentence_date = 'sentence_date'
        self.sentence_judge = 'sentence_judge'
        self.sentence_phase = 'sentence_phase'
        self.sentence_type = 'sentence_type'
        self.updated_offense_category = 'updated_offense_category'

        self.key_district = {
            1: 'District 1 - Chicago'
            , 2: 'District 2 - Skokie'
            , 3: 'District 3 - Rolling Meadows'
            , 4: 'District 4 - Maywood'
            , 5: 'District 5 - Bridgeview'
            , 6: 'District 6 - Markham'
        }

        self.fac_name = 'Fac_Name'

        self.key_dispo = {
            'Finding Guilty': 'Guilty at Trial'
            , 'Charge Vacated': 'Other'
            , 'Case Dismissed': 'Dismissal'
            , 'Verdict Guilty - Amended Charge': 'Guilty at Trial'
            , 'Verdict Guilty - Lesser Included': 'Guilty at Trial'
            , 'Nolle Prosecution': 'Dismissal'
            , 'Finding Guilty But Mentally Ill': 'Mental Health'
            , 'FNG': 'Not Guilty'
            , 'Death Suggested-Cause Abated': 'Death'
            , 'BFW': 'Other'
            , 'Charge Rejected': 'Dismissal'
            , 'Verdict Guilty': 'Guilty at Trial'
            , 'FNPC': 'Dismissal'
            , 'Plea Of Guilty': 'Guilty Plea'
            , 'SOL': 'Dismissal'
            , 'Nolle On Remand': 'Other'
            , 'Finding Guilty - Amended Charge': 'Guilty at Trial'
            , 'Plea of Guilty But Mentally Ill': 'Guilty Plea'
            , 'Sexually Dangerous Person': 'Mental Health'
            , 'Verdict-Not Guilty': 'Not Guilty'
            , 'Plea of Guilty - Amended Charge': 'Guilty Plea'
            , 'FNG Reason Insanity': 'Mental Health'
            , 'Charge Reversed': 'Other'
            , 'Transferred - Misd Crt': 'Dismissal'
            , 'Superseded by Indictment': 'Other'
            , 'Plea of Guilty - Lesser Included': 'Guilty Plea'
            , 'Finding Not Not Guilty': 'Mental Health'
            , 'Mistrial Declared': 'Other'
            , 'Verdict Guilty But Mentally Ill': 'Mental Health'
            , 'Hold Pending Interlocutory': 'Other'
            , 'Finding Guilty - Lesser Included': 'Guilty at Trial'
            , 'Withdrawn': 'Dismissal'
            , 'WOWI': 'Other'
            , 'SOLW': 'Dismissal'
            , np.nan: np.nan
        }

        self.key_facname = {
            '26Th Street': 'Criminal Courts (26th/California)'
            , 'Markham Courthouse': 'Markham Courthouse (6th District)'
            , 'Skokie Courthouse': 'Skokie Courthouse (2nd District)'
            , 'Rolling Meadows Courthouse': 'Rolling Meadows Courthouse (3rd District)'
            , np.nan: np.nan
            , 'Maywood Courthouse': 'Maywood Courthouse (4th District)'
            , 'Bridgeview Courthouse': 'Bridgeview Courthouse (5th District)'
            , 'Dv Courthouse': 'Domestic Violence Courthouse'
            , 'Dnu_3605 W. Fillmore St (Rjcc)': 'RJCC'
            , 'Daley Center': 'Daley Center'
            , '3605 W. Fillmore (Rjcc)': 'RJCC'
            , 'Grand & Central (Area 5)': 'Circuit Court Branch 23/50'
            , 'Harrison & Kedzie (Area 4)': 'Circuit Court Branch 43/44'
            , '51St & Wentworth (Area 1)': 'Circuit Court Branch 34/38'
            , 'Belmont & Western (Area 3)': 'Circuit Court Branch 29/42'
            , '727 E. 111Th Street (Area 2)': 'Circuit Court Branch 35/38'
        }

    def names(self):
        initiation_table = [
            'case_id'
            , 'case_participant_id'
            , 'received_date'
            , 'offense_category'
            , 'primary_charge_flag'
            , 'charge_id'
            , 'charge_version_id'
            , 'charge_offense_title'
            , 'charge_count'
            , 'chapter'
            , 'act'
            , 'section'
            , 'class'
            , 'aoic'
            , 'event'
            , 'event_date'
            , 'finding_no_probable_cause'
            , 'arraignment_date'
            , 'bond_date_initial'
            , 'bond_date_current'
            , 'bond_type_initial'
            , 'bond_type_current'
            , 'bond_amount_initial'
            , 'bond_amount_current'
            , 'bond_electronic_monitor_flag_initial'
            , 'bond_electroinic_monitor_flag_current'
            , 'age_at_incident'
            , 'race'
            , 'gender'
            , 'incident_city'
            , 'incident_begin_date'
            , 'incident_end_date'
            , 'law_enforcement_agency'
            , 'law_enforcement_unit'
            , 'arrest_date'
            , 'felony_review_date'
            , 'felony_review_result'
            , 'updated_offense_category'
        ]

        disposition_table = [
            'case_id'
            , 'case_participant_id'
            , 'received_date'
            , 'offense_category'
            , 'primary_charge_flag'
            , 'charge_id'
            , 'charge_version_id'
            , 'disposition_charged_offense_title'
            , 'charge_count'
            , 'disposition_date'
            , 'disposition_charged_chapter'
            , 'disposition_charged_act'
            , 'disposition_charged_section'
            , 'disposition_charged_class'
            , 'disposition_charged_aoic'
            , 'charge_disposition'
            , 'charge_disposition_reason'
            , 'judge'
            , 'disposition_court_name'
            , 'disposition_court_facility'
            , 'age_at_incident'
            , 'race'
            , 'gender'
            , 'incident_city'
            , 'incident_begin_date'
            , 'incident_end_date'
            , 'law_enforcement_agency'
            , 'law_enforcement_unit'
            , 'arrest_date'
            , 'felony_review_date'
            , 'felony_review_result'
            , 'arraignment_date'
            , 'updated_offense_category'
        ]

        derived_table = ['case_id'
            , 'case_participant_id'
            , 'primary_charge_flag_init'
            , 'class'
            , 'received_date'
            , 'event'
            , 'judge'
            , 'disposition_court_name'
            , 'disposition_court_facility'
            , 'charge_disposition'
            , 'case_length'
            , 'disposition_date'
            , 'disposition_date_days_pending']

        initiation_modified = ['case_id', 'case_participant_id', 'received_date', 'offense_category',
                               'primary_charge_flag', 'charge_id', 'charge_version_id', 'charge_offense_title',
                               'charge_count', 'chapter', 'act', 'section', 'class', 'aoic', 'event', 'event_date',
                               'finding_no_probable_cause', 'arraignment_date', 'bond_date_initial',
                               'bond_date_current', 'bond_type_initial', 'bond_type_current', 'bond_amount_initial',
                               'bond_amount_current', 'bond_electronic_monitor_flag_initial',
                               'bond_electroinic_monitor_flag_current', 'age_at_incident', 'race', 'gender',
                               'incident_city', 'incident_begin_date', 'incident_end_date', 'law_enforcement_agency',
                               'law_enforcement_unit', 'arrest_date', 'felony_review_date', 'felony_review_result',
                               'updated_offense_category', 'disposition_date_days_pending']
        disposition_modified = ['case_id', 'case_participant_id', 'received_date', 'offense_category',
                                'primary_charge_flag', 'charge_id', 'charge_version_id',
                                'disposition_charged_offense_title', 'charge_count', 'disposition_date',
                                'disposition_charged_chapter', 'disposition_charged_act', 'disposition_charged_section',
                                'disposition_charged_class', 'disposition_charged_aoic', 'charge_disposition',
                                'charge_disposition_reason', 'judge', 'disposition_court_name',
                                'disposition_court_facility', 'age_at_incident', 'race', 'gender', 'incident_city',
                                'incident_begin_date', 'incident_end_date', 'law_enforcement_agency',
                                'law_enforcement_unit', 'arrest_date', 'felony_review_date', 'felony_review_result',
                                'arraignment_date', 'updated_offense_category', 'charge_disposition_cat', 'case_length',
                                'initial_charged_class', 'charged_class_difference']
        sentencing_modified = ['case_id', 'case_participant_id', 'received_date', 'offense_category',
                               'primary_charge_flag', 'charge_id', 'charge_version_id',
                               'disposition_charged_offense_title', 'charge_count', 'disposition_date',
                               'disposition_charged_chapter', 'disposition_charged_act', 'disposition_charged_section',
                               'disposition_charged_class', 'disposition_charged_aoic', 'charge_disposition',
                               'charge_disposition_reason', 'sentence_judge', 'sentence_court_name',
                               'sentence_court_facility', 'sentence_phase', 'sentence_date', 'sentence_type',
                               'current_sentence_flag', 'commitment_type', 'commitment_unit', 'age_at_incident', 'race',
                               'gender', 'incident_city', 'incident_begin_date', 'incident_end_date',
                               'law_enforcement_agency', 'law_enforcement_unit', 'arrest_date', 'felony_review_date',
                               'felony_review_result', 'arraignment_date', 'updated_offense_category',
                               'commitment_days', 'commitment_dollars', 'commitment_weight', 'life_term', 'case_length',
                               'charge_disposition_cat']

        all_cols = list(set(
            initiation_table + disposition_table + derived_table + sentencing_modified + disposition_modified + initiation_modified))

        all_cols.sort()

        self_list = [(str('self.' + x + '=' + "\'" + x + "\'")) for x in all_cols]
        self_list = [x if 'self.class=' not in x else x.replace('self.class=', 'self.charge_class=') for x in self_list]
        # https://www.geeksforgeeks.org/python-ways-to-print-list-without-quotes/
        # return a list to create class variables
        print('[%s]' % ' '.join(map(str, self_list)))

c = Columns()

def parse_cols(df):
    logging.info('parse_cols() Parsing columns text with lower string and underscores')
    df.columns = map(str.lower, df.columns)
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.replace('-', '_')
    return df

def get_date_cols(df):
    logging.info('get_date_cols() Getting a list of datetime columns')
    date_pattern = r'_date'
    date_cols = [c for c in df.columns if re.search(date_pattern, c)]
    return date_cols

def parse_dates(df, date_cols=None):
    # filter erroneous dates where arrest date < received in lock up < released from lockup
    logging.info('parse_dates() Parsing datetime columns as datetime')

    if date_cols is None:
        date_cols = get_date_cols(df)

    df[date_cols] = df[date_cols].apply(lambda x: pd.to_datetime(x, errors='coerce', infer_datetime_format=False))

    return df

def remove_conversion_records(df, col_name):
    logging.info('remove_conversion_records() Removing records marked as PROMIS Conversion')
    start = len(df)

    df[col_name] = df[col_name].str.title()

    cond1 = df[col_name] != 'Promis Conversion'
    df = df[cond1]

    cond2 = df[col_name] != 'Promis'
    df = df[cond2]

    end = len(df)
    count = str(start-end)

    return df

def impute_dates(df, col1=None, col2=None, date_type=None):
    """
    assumptions:
    1. if event date greater than this year + 1, it is a mistake and the year should be the same year as received
    2. if event date less than 2011, it is a mistake and should be same as received year
    """
    logging.info('impute_dates() Filling in missing dates based on ruleset')
    today = pd.Timestamp.now()
    curr_year = today.year
    past_year = 2010
    change_log = []
    col_new = str(col1 + '_new')

    if date_type == 'initiation':

        if col1 == 'event_date' or col1 == 'felony_review_date' or col1 == 'arraignment_date':

            impute = lambda x: x[col1].replace(year=x[col2].year) if x[col1].year > curr_year \
                    else x[col1].replace(year=x[col2].year) if x[col1].year < past_year \
                    else x[col1]

            df[col_new] = df.apply(impute, axis=1)

            print('------ Impute Dates for', col1, 'given', col2, ' ', len(df))

            df[col1] = df[col_new]

            df = df.drop(columns=col_new)

    if date_type == 'disposition':

        if col1 == 'disposition_date':

            df['diff'] = (df[col1] - df[col2]) / np_hours
            df['diff'] = df['diff'].rolling(100, min_periods=1).median()

            def new(x):
                new = x[col2] + pd.to_timedelta(x['diff'], unit='h')
                return new

            impute = lambda x: new(x) if x[col1] > today else x[col1]
            df[col_new] = df.apply(impute, axis=1)

            df[col1] = df[col_new]

            df = df.drop(columns=[col_new, 'diff'])

    if date_type == 'sentence':
        today = pd.Timestamp.now()

        impute = lambda x: x[col2] if x[col1] > today else x[col1]

        df[col_new] = df.apply(impute, axis=1)

        print('------ Impute Dates for', col1, 'given', col2, ' ', len(df))

        df[col1] = df[col_new]

        df = df.drop(columns=col_new)

    return df

def typeas_string(df, cols):
    df[cols] = df[cols].astype('str')
    return df

def classer(df, col_name, echo=False):
    if echo:
        logging.info(f'classer() Classifying to Categorical: {col_name}')

    def reverse(lst):
        lst.reverse()
        return lst

    if col_name == 'offense_category' or col_name == 'updated_offense_category':
        df[col_name] = df[col_name].str.strip()
        df[col_name] = df[col_name].str.upper()
        df[col_name] = df[col_name].astype('category')

    if col_name == 'charge_class' or col_name == 'class' or col_name == 'disposition_charged_class':
        ordered_charges = ['M', 'X', '1', '2', '3', '4', 'A', 'B', 'C', 'O', 'P', 'Z']
        ordered_charges.reverse()

        df[col_name] = df[col_name].str.strip()

        start = len(df)
        df = df[~(df[col_name] == 'U')].copy()
        end = len(df)

        diff = start-end
        if echo:
            logging.info(f'classer() Dropped Records with U: {diff}')

        df[col_name] = df[col_name].astype('category')
        df[col_name] = df[col_name].cat.as_ordered()
        df[col_name] = df[col_name].cat.reorder_categories(ordered_charges, ordered=True)

    if col_name == c.disposition_court_name or col_name == c.disposition_court_facility or col_name == 'sentence_court_name':
        df[col_name] = df[col_name].str.strip()
        df[col_name] = df[col_name].str.title()
        df[col_name] = df[col_name].astype('category')

    if col_name == 'race':
        try:
            df[col_name] = df[col_name].str.strip()
            df[col_name] = df[col_name].str.title()
            df[col_name] = df[col_name].astype('category')
        except:
            pass

    if col_name == 'gender':
        try:
            key = {'Unknown Gender': 'Unknown'}
            df[col_name] = df[col_name].str.strip()
            df[col_name] = df[col_name].str.title()
            df[col_name] = np.where(df[col_name] == 'Unknown Gender', df[col_name].map(key), df[col_name])
            df[col_name] = df[col_name].astype('category')
        except:
            pass

    if col_name == 'judge' or col_name == 'sentence_judge':
        try:
            df[col_name] = df[col_name].str.strip()
            df[col_name] = df[col_name].str.replace('\.', '')
            df[col_name] = df[col_name].str.replace('\s+', ' ')
            df[col_name] = df[col_name].str.title()

            df['temp'] = df[col_name].str.contains(',', na=False)
            df['names'] = np.where(df['temp'] == True, df[col_name].str.split(','), df[col_name])

            temp = df[(df['temp'] == True)]
            names = temp['names'].tolist()

            backwards = [list(x) for x in set(tuple(x) for x in names)]
            x1 = [', '.join(str(c).strip() for c in s) for s in backwards]

            forwards = [reverse(x) for x in backwards]
            x2 = [' '.join(str(c).strip() for c in s) for s in forwards]

            key = {x1[i]: x2[i] for i in range(len(x1))}

            df[col_name] = np.where(df['temp'] == True, df[col_name].map(key, na_action='ignore'), df[col_name])
            df[col_name] = df[col_name].astype('category')

            df = df.drop(columns=['names', 'temp'])
        except:
            pass

    if isinstance(col_name, list):
        try:
            df[col_name] = df[col_name].astype('category')
        except:
            pass

    return df


def make_caselen(df, col1=None, col2=None):
    logging.info('make_caselen() Calculating case length')
    df['case_length'] = (df[col2] - df[col1]) / np_days
    return df


def reduce_bool_precision(df, col=None):
    logging.info('reduce_bool_precision() Parsing least precision for boolean')
    cols = list(df.columns)

    bool_types = ['flag', 'finding_no_probable_cause']
    to_convert = [x for x in cols if any(i in x for i in bool_types)]

    df[to_convert] = np.where(df[to_convert].isnull(), pd.NA,
                              np.where(df[to_convert]==1., True, df[to_convert]))

    df[to_convert] = df[to_convert].astype('boolean')

    return df

def map_disposition_categories(df, col1=None):
    logging.info(f'map_disposition_categories() Mapping Categories for {col1}')
    cat_col = str(col1 + '_cat')
    df[cat_col] = df[col1].map(c.key_dispo)
    df[cat_col] = df[cat_col].astype('category')

    return df

def reduce_precision(df, run_spellcheck=False):
    """
    :param df: attempts to auto-optimize a dataframe by applying least precision to each col type
    :return: the same dataframe but with different col dtypes, if applicable
    """
    logging.info('reduce_precision() Optimizing DataFrame Memory')

    cols_to_convert = []
    date_strings = ['_date', 'date_', 'date']

    for col in df.columns:
        col_type = df[col].dtype
        if 'string' not in col_type.name and col_type.name != 'category' and 'datetime' not in col_type.name:
            cols_to_convert.append(col)

    def _reduce_precision(x):
        col_type = x.dtype
        unique_data = list(x.unique())
        bools = [True, False, 'true', 'True', 'False', 'false']
        #TODO: account for only T or only F or 1/0 situations
        n_unique = float(len(unique_data))
        n_records = float(len(x))
        cat_ratio = n_unique / n_records

        try:
            unique_data.remove(np.nan)
        except:
            pass

        if 'int' in str(col_type):
            c_min = x.min()
            c_max = x.max()

            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                x= x.astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                x = x.astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                x = x.astype(np.int32)
            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                x = x.astype(np.int64)

                # TODO: set precision to unsigned integers with nullable NA

        elif 'float' in str(col_type):
            c_min = x.min()
            c_max = x.max()
            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    x = x.astype(np.float16)
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                x = x.astype(np.float32)
            else:
                x = x.astype(np.float64)

        elif 'datetime' in col_type.name or any(i in str(x.name).lower() for i in date_strings):
            try:
                x = pd.to_datetime(x)
            except:
                pass

        elif any(i in bools for i in unique_data):
            x = x.astype('boolean')
            #TODO: set precision to bool if boolean not needed

        elif cat_ratio < .1 or n_unique < 20:
            try:
                #TODO: work on standardizing text data within narrow edit distance
                # x = x.fillna('Missing')
                x = x.str.title()
                # if run_spellcheck:
                #     correct_spellings = [str(TextBlob(i).correct()).title() for i in list(x.unique())]
                #     key = dict(zip(list(x.unique()), correct_spellings))
                #     x = np.where(isinstance(x, str), x.map(key), x)
            except:
                pass

            x = pd.Categorical(x)

        elif all(isinstance(i, str) for i in unique_data):
            x = x.astype('string')

        return x

    df[cols_to_convert] = df[cols_to_convert].apply(lambda x: _reduce_precision(x))
    # self.dfs_new = df_dict[get_report_ctx().session_id]
    return df

def max_disp_charge(df):
    """
    for each disposition charge and person, return the most 'severe' charge
    """
    charged_class_code = 'charged_class_category'
    cols = [c.case_id, c.case_participant_id, c.received_date, c.updated_offense_category,
            c.disposition_charged_class]
    df1 = df[cols].copy()
    df1[charged_class_code] = df1[c.disposition_charged_class].cat.codes

    cols = [c.case_id, c.case_participant_id, c.received_date, c.updated_offense_category]
    idx = df1.groupby(cols, sort=False)[charged_class_code].transform(max) == df1[charged_class_code]
    df = df[idx].drop_duplicates(subset=cols)

    return df

def ov1_disposition(df):
    """
    return the most severe allegation for a given case (not always the primary charge
    https://stackoverflow.com/questions/15705630/get-the-rows-which-have-the-max-count-in-groups-using-groupby
    """
    df = max_disp_charge(df)

    cols = [
          c.case_id
        , c.case_participant_id
        , c.received_date
        , c.disposition_date
        , c.updated_offense_category
        , c.disposition_charged_class
        , c.charge_disposition_cat
        , c.case_length
            ]

    df = df[cols]

    df['year'] = df[c.disposition_date].dt.year.astype('float32').fillna(value=0)
    df['year'] = df.apply(lambda x: x[c.received_date].year if x['year'] == 0 else x['year'], axis=1)
    df['year'] = df['year'].astype('int16')

    df = df.reset_index(drop=True)
    logging.info('ov1_disposition() Returning Disposition Data for Most Severe Charge in a Given Case')

    return df

def filter_disposition_data(df
                        , data_folder
                        , csv_filename='covid_cliff_disposition_data.csv'
                        , pickle_filename='covid_cliff_disposition_data.pickle'
                          ):

    restricted_list = ['Death', 'Mental Health', 'Other']

    df = ov1_disposition(df)
    df = df[~(df[c.charge_disposition_cat].isin(restricted_list))].copy()
    df[c.charge_disposition_cat] = df[c.charge_disposition_cat].cat.remove_unused_categories()
    df = df[[c.disposition_date, c.charge_disposition_cat, c.case_length]]

    data_file = os.sep.join([data_folder, csv_filename])
    df.to_csv(data_file)
    data_file = os.sep.join([data_folder, pickle_filename])
    df.to_pickle(data_file)

    return df

def make_sequence_for_prophet(x):
    """
    A helper function for
    :param x: a dataframe having data for train and test (everything)
    :return: a list of 2-tuples having (dataframe, string) as data and data label for prophet
    """

    # map for col names per facebook api
    key_map = {c.disposition_date: 'ds', 'count': 'y'}

    # apply col name remap
    df = x.rename(columns=key_map)

    # group data by desired category
    df = df.groupby(c.charge_disposition_cat)

    # init empty list for output
    data_list = []

    for name, group in df:
        # loop through each data from of group
        # hard-coded values for window and step size
        year_idx_start = 2011
        window_size = 2
        step_size = 1
        year_idx_max = 2021
        steps = list(range(year_idx_start, year_idx_max+1))

        x_i = []
        y_i = []

        df = group.copy()

        # within each loop, extract data frame into train and test pairs
        for step in steps:
            # train on two years, predict the third, break when year 3 is at end of index
            if step + 3 > max(steps):
                break

            # in the first loop, 2011 < x.dt.year < 2013 and y.dt.year == 2014
            X = df[(df['ds'].dt.year >= step) & (df['ds'].dt.year <= step + window_size)].reset_index(drop=True)
            Y = df[(df['ds'].dt.year == step + window_size + step_size)].reset_index(drop=True)

            # save sequenced dataframes in lists
            x_i.append(X)
            y_i.append(Y)

        # save data as 2-tuple pairs
        data = (x_i, y_i)

        # accumulate data as a list of tuples
        data_list.append((data, name))

    return data_list

def prep_disposition_data_for_prophet(df):

    grouper = pd.Grouper(key=c.disposition_date, freq='D')

    df = df.groupby([c.charge_disposition_cat, grouper])[c.charge_disposition_cat].agg('count')
    df = df.reset_index(name='count')
    df = df.sort_values(by=[c.disposition_date])
    df = df.reset_index(drop=True)

    data = make_sequence_for_prophet(df)

    return data

def prep_arrest_data_for_prophet(df
                               , data_folder
                               , csv_filename='covid_cliff_arrest_data.csv'
                               , pickle_filename='covid_cliff_arrest_data.pickle'
                                 ):
    felony_flag = "F"
    frequency = "M"

    df = df[df['charge_1_type'] == felony_flag].copy()

    df['date'] = pd.to_datetime(df['arrest_year'].astype(str) + '-' + df['arrest_month'].astype(str) + '-01')
    df = df[['date', 'charge_1_class']].groupby([pd.Grouper(key='date', freq=frequency)]).agg('count').reset_index()
    df = df.sort_values('date')
    df['type'] = 'Felony Arrest'
    df.rename(columns={'charge_1_class': 'count'}, inplace=True)

    data_file = os.sep.join([data_folder, csv_filename])
    df.to_csv(data_file, index=False)
    logging.info(f'prep_arrest_data_for_prophet() Wrote to disk {data_file}')

    data_file = os.sep.join([data_folder, pickle_filename])
    df.to_pickle(data_file)
    logging.info(f'prep_arrest_data_for_prophet() Wrote to disk {data_file}')
    return df

class MakeMongo():
    def __init__(self):
        self.pw = config('PASSWORD')
        self.db_name = config('DATABASE_NAME')
        self.un = config('USERNAME')
        self.host = config('HOST')
        """ References
        https://able.bio/rhett/how-to-set-and-get-environment-variables-in-python--274rgt5
        """

    def insert_df(self, database=None, collection=None, df=None):
        logging.info('MakeMongo.insert_df() Uploading data to MongoDB')

        header = "mongodb+srv://"

        connection_string = str(header+self.un+":"+self.pw+"@"+self.host+"/"+"?retryWrites=true&w=majority")

        client = MongoClient(connection_string)

        if database:
            db = client[database]
        else:
            db = client[self.db_name]


        if df is not None and collection:
            data_dict = df.to_dict("records")
            db[collection].delete_many({})
            db[collection].insert(data_dict)
            logging.info('MakeMongo.insert_df() Completed MongoDB Upload')

def etl_disposition_data(predictions
                            , data_folder
                            , raw_predictions='raw_court_data_predictions.csv'
                            , csv_filename='covid_cliff_court_data_predicted.csv'
                            , pickle_filename='covid_cliff_court_data_predicted.pickle'
                            ):
    df = pd.concat([pd.concat(i) for i in predictions])
    df = df.rename(columns={'ds': c.disposition_date
                            ,'y': 'case_count'
                            ,'yhat': 'predicted_case_count'})

    df = df[[c.disposition_date, 'case_count', c.charge_disposition_cat, 'predicted_case_count']]
    data_file = os.sep.join([data_folder, raw_predictions])
    # export data to file for analysis
    raw_dispositions_predictions = df.copy(deep=True)
    raw_dispositions_predictions.to_csv(data_file, index=False)
    logging.info(f'etl_disposition_data() Wrote to disk {data_file}')

    grouper = pd.Grouper(key=c.disposition_date, freq='M')

    df = df.groupby([c.charge_disposition_cat, grouper]).agg({'case_count':'sum','predicted_case_count':'sum'})

    df = df.reset_index()

    df = df.sort_values(by=[c.disposition_date])
    df = df.reset_index(drop=True)

    df = df[[c.disposition_date, 'case_count', c.charge_disposition_cat, 'predicted_case_count']]

    data_file = os.sep.join([data_folder, csv_filename])
    df.to_csv(data_file, index=False)
    logging.info(f'etl_disposition_data() Wrote to disk {data_file}')
    data_file = os.sep.join([data_folder, pickle_filename])
    df.to_pickle(data_file)
    logging.info(f'etl_disposition_data() Wrote to disk {data_file}')

    return df, raw_dispositions_predictions

def merge_dispositions_arrests(court_df, arrest_df):

    df = pd.merge(court_df, arrest_df, left_on=c.disposition_date, right_on='arrest_date')

    df = df.rename(columns={c.disposition_date:'date'})

    df = df.drop(columns=['arrest_date'])

    df['case_count'] = list(zip(df[c.charge_disposition_cat],df['case_count']))
    df['predicted_case_count'] = list(zip(df[c.charge_disposition_cat],df['predicted_case_count']))

    df['arrest_count'] = list(zip(df['arrest_type'], df['arrest_count']))
    df['predicted_arrest_count'] = list(zip(df['arrest_type'], df['predicted_arrest_count']))

    df.drop(columns=['arrest_type', c.charge_disposition_cat], inplace=True)

    # this is a unified dataframe of dispositions and arrest data
    return df

def filter_court_backlog(court_df, arrest_df=None):
    logging.info('filter_court_backlog() Generated backlog estimates')
    # TODO: add arrest data as a prediction feature layer
    # arrest_df = pd.read_pickle('data/covid_cliff_arrest_data_predicted.pickle')

    df = court_df[court_df[c.disposition_date] > pd.to_datetime("2020-02-01")].copy()
    df['backlog'] = df['predicted_case_count'] - df['case_count']

    df = df.reset_index(drop=True)

    return df

def decimal_str(x: float, decimals: int = 10) -> str:
    return format(x, f".{decimals}f").lstrip().rstrip('0')


