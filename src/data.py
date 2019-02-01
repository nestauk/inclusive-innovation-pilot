import re
import sys
import pandas as pd
import ethnicolr
from data_getters.core import get_engine
import pickle
import numpy as np
from lists import postgrad, undergrad, phil, mba, judge


def concat_chunks(chunks):
    """Concatenate a list of DataFrames to a DataFrame.

    Args:
        chunks (:obj:`list` of :obj:`pandas.DataFrame`): List of DataFrames.

    Return:
        (:obj: of `pandas.DataFrame`)

    """
    return pd.concat([chunk for chunk in chunks])


# "../innovation-mapping.config"
def read_data(config_file):
    con = get_engine(config_file)

    orgs = concat_chunks(pd.read_sql_table('crunchbase_organizations', con,
                                           chunksize=1000))
    cats = concat_chunks(
                        pd.read_sql_table(
                                         'crunchbase_organizations_categories',
                                         con, chunksize=1000))
    cat_groups = concat_chunks(pd.read_sql_table('crunchbase_category_groups',
                                                 con, chunksize=1000))
    geo = concat_chunks(pd.read_sql_table('geographic_data', con,
                                          chunksize=1000))
    degrees = concat_chunks(pd.read_sql_table('crunchbase_degrees', con,
                                              chunksize=1000))
    jobs = concat_chunks(pd.read_sql_table('crunchbase_jobs', con,
                                           chunksize=1000))
    people = concat_chunks(pd.read_sql_table('crunchbase_people', con,
                                             chunksize=1000))
    return orgs, cats, cat_groups, geo, degrees, jobs, people


def company_size(val):
    regex = re.compile(r'\d+')
    if isinstance(val, str):
        values = [int(v) for v in re.findall(regex, val)]
    else:
        return val
    if val == 'unknown':
        return np.nan
    elif all(v > 250 for v in values):
        return 'Large'
    elif values[0] > 50 and values[1] <= 250:
        return 'Medium'
    elif values[0] > 10 and values[1] <= 50:
        return 'Small'
    elif values[1] <= 10:
        return 'Micro'
    else:
        return val


def change_degree_type(val):
    if val == 'unknown' or val == 'Unknown':
        return np.nan
    # PhD
    elif any(d == val for d in phil):
        return 'PhD'
    # Postgraduate
    elif any(d == val for d in postgrad):
        return 'Postgraduate'
    # Undergraduate
    elif any(d == val for d in undergrad):
        return 'Undergraduate'
    elif any(d == val for d in mba):
        return 'MBA'
    elif any(d == val for d in judge):
        return 'JD'
    else:
        return val


def prepare_data():
    orgs, cats, cat_groups, geo, degrees, jobs, people = read_data(sys.argv[1])
    with open(sys.argv[2], 'rb') as h:
        org_ids = pickle.load(h)

    orgs = orgs[(orgs.id.isin(org_ids))]

    oj = orgs[['id', 'funding_total_usd', 'founded_on', 'city', 'country',
               'employee_count', 'primary_role']].merge(
                                                jobs[['person_id', 'org_id',
                                                      'job_id', 'is_current',
                                                      'job_type']],
                                                left_on='id',
                                                right_on='org_id')

    categories = cats.merge(cat_groups, left_on='category_name',
                            right_on='category_name')

    oj = oj.merge(categories[['organization_id', 'category_group_list']],
                  left_on='id', right_on='organization_id')

    ojp = oj.merge(people[['id', 'first_name', 'last_name', 'gender']],
                   left_on='person_id', right_on='id')
    # Predict ethnicity given first and last name
    ojp = ethnicolr.pred_wiki_name(df=ojp, lname_col='last_name',
                                   fname_col='first_name')
    ojpd = ojp.merge(degrees[['person_id', 'degree_type', 'degree_id',
                              'institution_id']],
                     how='left', left_on='id_y', right_on='person_id')
    ojpd.drop(['person_id_x', 'person_id_y', 'organization_id'],
              axis=1, inplace=True)
    ojpd.rename(index=str, inplace=True, columns={'id_x': 'org_id',
                                                  'id_y': 'person_id'})

    ojpd.degree_type = ojpd.degree_type.apply(change_degree_type)
    ojpd.employee_count = ojpd.employee_count.apply(company_size)

    ojpd.to_csv('../data/processed/ojpd.csv')
    print(ojpd.shape)


if __name__ == '__main__':
    prepare_data()
