# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

## NOTES:
# 1. double check to make sure th data was imported correctly
# 2. look into adding meta-data for incentives(local and states' gov't)
# 3. explore data with numerous missing values. Are they for the same city or state?
# 4. focus on major cities when building a model or exploring incentives
# 5. How long would it take to break even on investment?



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import requests
import json
from pandas.io.json import json_normalize


# =============================================================================
# import data
# =============================================================================
#, index_col='date_installed', parse_dates=True
# infer_datetime_format=True
pv_df_orig = pd.read_csv('openpv_all.csv', sep=',', low_memory=False, na_values=np.nan)

pv_df_orig.info()
pv_df_orig[pv_df_orig[['pbi_length']].notnull()]

pv_df_orig['tracking_type'].unique()
# =============================================================================
# Data Wrangling
# =============================================================================


# =============================================================================
# Remove columns with all null values
# =============================================================================
'''
NOTE:
pd.isnull(pv_df.size_kw).value_counts() # count null values
pd.notnull(pv_df.size_kw).all() # returns TRUE when ALL values are not null in column
pd.notnull(pv_df.size_kw).any() # returns TRUE when ANY values are not null in column
pd.isnull(pv_df_orig.inv_model_clean).all() # True if ALL values in column are NULL
pd.notnull(pv_df.size_kw).value_counts() # number of missing values

'''

# FUNCTION: remove NULL columns
def remove_null_col(df, percent=80):
  '''Remove all columns with too many null values'''
  keeper_list = []
  # loser_list = []
  col_list = list(df.columns)
  # iterate through each column
  for col_name in col_list:
    # A and B are the conditions for removing columns
    nan_count = pd.isnull(df[col_name]).sum()
    A = df[col_name].isnull().all()
    B = nan_count/len(df.index)*100 > percent
    # if column is empty or more than 20% of columns is NaN, then discard, else keep
    if (A | B):
      # place column into a keeper list
      # loser_list.append(col_name)
      print('column removed:', col_name)
    else:
      keeper_list.append(col_name)
  #pv_df = df[keeper_list]
  return df[keeper_list]
pv_df = remove_null_col(pv_df_orig, 80.0)
# NOTE: use dropna(axis=1, how='all') to drop entire column
pv_df = pv_df[['date_installed', 'city','state','zipcode', 'county', 'size_kw','annual_insolation',
               'annual_PV_prod', 'reported_annual_energy_prod',
               'cost_per_watt','cost', 'sales_tax_cost', 'rebate', 'incentive_prog_names', 'install_type',
               'installer','utility_clean', 'tracking_type', 'tilt1', 'azimuth1']]
pv_df.head()
pv_df.info()
# =============================================================================
# missing values
# =============================================================================


### NOTE: select indexes where 'cost'==0, and set all rows to np.nan for all columns
# 1. pv_df[pv_df.loc[:, 'cost']==0]=np.nan
# 2. pv_df = pv_df.dropna(how='all')

# CONVERT DATA TYPE OF 'ZIPCODES'
# convert to numpy array, replace missing values with 0, then convert data type
pv_df['zipcode'] = np.nan_to_num(pv_df['zipcode'].values).astype(int)


def missing_val_count(df, col_name, show_missing=False):
  '''Count missing values and show its index.'''
  print('\nColumn:', col_name, '\nMissing values:', np.count_nonzero(df.loc[:, col_name].isnull().values))
  if show_missing == True:
    print(df[df.loc[:, col_name].isnull()])
print(missing_val_count(pv_df, 'state'))


def drop_n_reset(df):
  '''Drop all missing rows, duplicates, and reset the index.'''
  df = df.dropna(axis=0, how='all').drop_duplicates()
  df = df.reset_index(drop=True)
drop_n_reset(pv_df)

# =============================================================================
# CONVERT DATA TYPES AND REMOVE STRINGS/SYMBOLS
# =============================================================================
# NUMBER OF STATES
pv_df.state.nunique() # number of unique states
pv_df.state.unique() # OH, MD and PA contain an empty string, PR and DC are not states

# NOTE: cost and rebate columns are dtype 'object' although they are supposed to be numeric
# test for any digits within character strings
col = list(pv_df.columns)
for item in col:
  if pv_df[item].dtype == np.object:
      if pv_df[item].str.isnumeric().any():
        print(item, '<- dtype is object but contains digits')

# CONVERT TO FLOAT
# NOTE: doesn't convert due to error. Strings must be replaced before dtype conversion
pv_df.loc[:, 'cost'] = pv_df.loc[:, 'cost'].values.astype(float)
pv_df.loc[:, 'rebate'] = pv_df.loc[:, 'rebate'].values.astype(float)

# FUNCTION: look for string 'n/a', dollar sign '$', string 'null', and comma ','
# strip all leading and trailing whitespaces
def remove_symbols(df):
  '''For columns which are dtype 'object', strip all leading and trailing whitespaces
  remove unwanted strings/symbols such as dollar sign and comma used in currency.
  Replace the string 'n/a' and 'null' with a numeric zero'''
  cols = list(df.columns)
  for item in cols:
    if df.loc[:, item].dtype == np.object:
      df.loc[:, item] = df.loc[:, item].str.strip('$')
      df.loc[:, item] = df.loc[:, item].replace(['n/a', 'null'], 0)

  return df
pv_df = remove_symbols(pv_df)
# pv_df.info()

pv_df.loc[:, 'rebate'] = pv_df.loc[:, 'rebate'].str.replace(',', '')
pv_df.loc[:, 'state'] = pv_df.loc[:, 'state'].str.rstrip()

# confirm removal of unwanted characters
conditions = [pv_df['cost'].str.contains('n/a').any(), pv_df['rebate'].str.contains('\$').any(),
              pv_df['rebate'].str.contains(',').any(), pv_df['rebate'].str.contains('null').any()]

# check for conditions being satisfied
for item in conditions:
  if item:
    print(conditions)
    print("Test failed. You're not done")

# convert zeros to NaNs since cost cannot be zero
pv_df.loc[:, 'cost']= pv_df.loc[:, 'cost'].replace(0, np.nan)

# CONVERT TO FLOAT
pv_df.loc[:, 'cost'] = pv_df.loc[:, 'cost'].values.astype(float)
pv_df.loc[:, 'rebate'] = pv_df.loc[:, 'rebate'].values.astype(float)


#
# =============================================================================
# TIMING
%timeit np.count_nonzero(pv_df['cost'].isnull().values) # fastest
%timeit np.count_nonzero(pv_df.loc[:, 'cost'].isnull())

%timeit pd.isnull(pv_df['cost_per_watt'].values).sum()
%timeit pv_df['cost_per_watt'].isnull().values.sum()
%timeit pv_df['cost_per_watt'].isnull().values.ravel().sum()


# =============================================================================
# # LOWER CASE
# convert data types prior to Lower Case
# =============================================================================

# create function to to test data type and convert to lower case
def upper_to_lower(df):
  ''' Test each column for data type 'object', then convert to lower case '''

  col_name = list(df.columns)
  # iterate over columns
  for item in col_name:
    # if data type is object, then convert column to lower case
    if df[item].dtype == np.object:
      df.loc[:, item] = df.loc[:, item].str.lower()
      print(item, '-converted to lower case')
  return df
pv_df = upper_to_lower(pv_df)


# =============================================================================
# MISSING VALUES AND SUMMARY STATS
# =============================================================================


# =============================================================================
# # ZIPCODE
# =============================================================================
# create missing values in 'state' column then drop rows
pv_df[pv_df['zipcode']==0]=np.nan
# pv_df['zipcode'] = np.nan_to_num(pv_df['zipcode'].values).astype(int)

# count missing values created in the 'state' column
missing_val_count(pv_df, 'state', True)

# pv_df.info()
# drop NA rows, duplicates and reset index
pv_df = pv_df.dropna(axis=0, how='all').drop_duplicates().reset_index(drop=True)

# =============================================================================
# Remove zipcodes
# =============================================================================
# ZIPCODES
# for each zipcode, extract annual radiation, city, state
# NOTE: some zipcodes have only 3 or 4 digits with missing zeros. Consider REMOVING
pv_df = pv_df.sort_values(by = ['zipcode'])
pv_df.loc[1950:2000, 'zipcode']

# =============================================================================
# Remove columns
# =============================================================================
pv_df[pv_df['county'].isnull()]

pv_df[['install_type', 'installer', 'utility_clean']] # keep
pv_df[['tilt1', 'tracking_type', 'azimuth1']] # remove

pv_df = pv_df[['date_installed', 'city','state', 'county', 'size_kw','annual_insolation',
               'annual_PV_prod', 'reported_annual_energy_prod',
               'cost_per_watt','cost', 'sales_tax_cost', 'rebate', 'incentive_prog_names', 'install_type',
               'installer','utility_clean']]

pv_df.info()
pv_df = pv_df.dropna(axis=0, how='all').drop_duplicates().reset_index(drop=True)

# =============================================================================
# States
# =============================================================================

# count rows for each state
state_count = pv_df['state'].value_counts(dropna=False)
state_count.describe()

# filter states to remove, keep states which appear > 100
states = state_count[state_count<646408]
states = dict(states)
# create a list of states to remove
state_list = list(states.keys())

# create for-loop to filter out states
# for states to remove, create NaNs in those rows
for item in state_list:
  pv_df[pv_df['state']==item]=np.nan
pv_df = pv_df.dropna(axis=0, how='all').reset_index(drop=True)
pv_df.info()

def row_count_drop(df, col_name, count=100):
  ''' Count amount of rows belonging to each unique item in col_name.
  Remove rows which appear less than "count" times.'''
  # count rows for each state
  count_series = df[col_name].value_counts(dropna=False)
  # filter states to remove, keep states which appear > 100
  count_ = count_series[count_series <= count].astype(str)
  count_ = dict(count_)
  # create a list of states to remove
  count_list = list(count_.keys())
  for item in count_list:
    df[df[col_name]==item]=np.nan
  df = df.dropna(axis=0, how='all')
  df = df.reset_index(drop=True)
  return df
row_count_drop(pv_df, 'state', 100)

pv_df = pv_df.dropna(axis=0, how='all').drop_duplicates().reset_index(drop=True)

# =============================================================================
# =============================================================================
# # WRITE TO CSV:
# =============================================================================

# found character '\r' within 'city' column and removed it
pv_df['city'].str.contains('\r').any()
pv_df['city'] = pv_df['city'].str.strip('\r')
pv_df[pv_df['city'].isnull().values] # 220,713

# write to csv partially cleaned file
pv_df.to_csv('pv_df_short.csv', encoding='utf-8', na_rep='NA', index=False)

# read PV_DF csv
# NOTE: file does not load properly, rows appear to skip
pv_df = pd.read_csv('pv_df_short.csv', sep=',', low_memory=False)

# NOTE: columns become distorted in 4 rows when reading CSV
pv_df['state'].unique()
pv_df[pv_df['state']=='los angeles']
missing_val_count(pv_df, 'state', True)

# inspect indexes
pv_df.loc[678355:678359,[ 'date_installed', 'city','state', 'county', 'size_kw','annual_insolation',
               'annual_PV_prod', 'reported_annual_energy_prod',
               'cost_per_watt','cost','rebate', 'incentive_prog_names', 'install_type',
               'installer','utility_clean']]



# =============================================================================
# MERGE DATAFRAMES
# =============================================================================
def load_csv(file_name):
  df = pd.read_csv(file_name, sep=',', low_memory=False)
  return df

# open files
pv_df = load_csv('pv_df_short.csv')

sol_df = load_csv('solar_rad_state.csv')
incent_df = load_csv('incentives_state.csv')
states_abbv = load_csv('states_abbreviation.csv')
pop_df = load_csv('population_df.csv')

incent_df.head()
states_abbv.head()
sol_df.head()
pop_df.info()
pv_df.info()

# merge population abd state abbv
population_state = pd.merge(pop_df, states_abbv, left_on='states', right_on='full', how='left')
population_state.head()
population_state.info()
population_state[['states', 'abbreviation']]
drop_n_reset(population_state)
population_state = population_state[['cities', 'population', 'abbreviation', 'full']]
population_state.columns = ['city', 'pop', 'state_short', 'state']

# merge
df2 = incent_df.merge(population_state, left_on='state', right_on='state', how='right')
df2.info()
drop_n_reset(df2)
incentive_state_df = df2

sol_df.info()
incentive_state_df.head()
df3 = incentive_state_df.merge(sol_df, left_on='state_short', right_on='states', how='left')
df3.head()

df3['incentive_program'].nunique()

df3 = df3[['incentive_program', 'states', 'cities', 'solar_rad']]
df3.info()
df3.head()
# drop duplicates if any
df3 = df3.dropna(axis=0, how='all').drop_duplicates().reset_index(drop=True)
df3[df3['states'].isnull()]

df3.loc[1990:2003, ['incentive_program', 'states', 'state', 'cities', 'solar_rad']]
incent_df[incent_df['incentive_program']=='renewable portfolio standard']

# sort
df3 = df3.sort_values(by=['states'])
df3.tail(20)
sol_incent_df = df3


# =============================================================================
# USE THIS METHOD, merge only with population and abbreviations for states
# merge one at a time
# =============================================================================
# merge pop_df and pv_df, MERGE on STATES
pop_df.head()
states_abbv.head()
pv_df2 = pd.merge(pv_df, states_abbv, left_on='state', right_on='abbreviation', how='left').dropna(axis=0, how='all').drop_duplicates()
pv_df2 = pv_df2.reset_index(drop=True)
pv_df2.head()
pv_df2 = pv_df2[['date_installed', 'city','state', 'full', 'county', 'size_kw','annual_insolation',
               'annual_PV_prod', 'reported_annual_energy_prod','cost_per_watt','cost','sales_tax_cost','rebate',
               'incentive_prog_names', 'install_type', 'installer','utility_clean', 'tracking_type', 'tilt1', 'azimuth1']]
pv_df2.info()
pv_df.info()

# merge population
pv_df3 = pd.merge(pv_df2, pop_df, left_on=['city','full'], right_on=['cities','states'], how='left')
pv_df3 = pv_df3.dropna(axis=0, how='all').drop_duplicates().reset_index(drop=True)
pv_df3.info()
pv_df3.head()
pv_df3 = pv_df3[['date_installed', 'city','state', 'full', 'county', 'population', 'size_kw','annual_insolation',
               'annual_PV_prod', 'reported_annual_energy_prod','cost_per_watt','cost', 'sales_tax_cost', 'rebate',
               'incentive_prog_names', 'install_type', 'installer','utility_clean', 'tracking_type', 'tilt1', 'azimuth1']]


# =============================================================================
# DO NOT MERGE INCENTIVES AT THIS POINT
# creates a huge file 4.2GB+
# =============================================================================
# merge: pv_df2 and sol_df
sol_df.info()
pv_df.info()

pv_df4 = pd.merge(pv_df2, sol_df, left_on=['state'], right_on=['states'], how='left').drop_duplicates().reset_index(drop=True)
pv_df4.head()
pv_df4 = pv_df4[['date_installed', 'city','state', 'county','size_kw','annual_insolation',
               'annual_PV_prod', 'reported_annual_energy_prod',
               'cost_per_watt','cost','rebate', 'incentive_prog_names', 'install_type',
               'installer','utility_clean']]

# *** DROP DUPLICATES AND RESET INDEX TO REDUCE THE SIZE
pv_df4 = pv_df4.dropna(axis=0, how='all').drop_duplicates().reset_index(drop=True)
pv_df4.info()

# FILE TOO BIG, 3.5+ GB
incentive_state_df = incentive_state_df.dropna(axis=0, how='all').drop_duplicates().reset_index(drop=True)
incentive_state_df.info()

pv_df5 = pv_df4.merge(incentive_state_df, left_on='state', right_on='state', how='left')
pv_df5.info()
pv_df5 = pv_df5.dropna(axis=0, how='all').drop_duplicates().reset_index(drop=True)

drop_n_reset(pv_df)

# =============================================================================
# WRITE TO FILE FOR NEXT USE
# =============================================================================
pv_df = pv_df3
# rename columns
pv_df3.columns = ['date_installed', 'city', 'state_short', 'state', 'county', 'population',
                 'size_kw', 'annual_insolation', 'annual_pv_prod',
                 'reported_annual_energy_prod', 'cost_per_watt', 'cost','sales_tax_cost', 'rebate',
                 'incentive_prog_names', 'install_type', 'installer', 'utility', 'tracking_type', 'tilt1', 'azimuth1']

pv_df.to_csv('merged_dfs.csv', encoding='utf-8', index=False)
pv_df = load_csv('merged_dfs.csv')
pv_df.info()


pv_df['county'].str.contains('st. louis').any()
pv_df['county'].str.contains('st louis').any()
# found both: 'st. louis' and 'st louis' in county column
pv_df[pv_df['county']=='st. louis']
pv_df[pv_df['state']=='mo']

# remove the period from 'st.' to maintain consistency
pv_df['city'] = pv_df['city'].str.replace('.', '')
pv_df['county'] = pv_df['county'].str.replace('.', '')

# =============================================================================
# INCENTIVES
# =============================================================================
incent_df.head()
pv_df.head()

# fill incentives by referenceing from 'sol_incent_df'
incent_df = incent_df.dropna(axis=0, how='all').drop_duplicates().reset_index(drop=True)

incent_df[incent_df['state'].isnull()]
incent_df['state'].nunique()


# create a new DF for each state with its incentive program
row = incent_df[(incent_df['state']=='arkansas') & (incent_df['incentive_program'])].index[0]
incent_df.loc[row, 'incentive_program'] # replacement name for state 'arkansas'


# check again for missing values
pv_df[pv_df['incentive_prog_names'].isnull()][['incentive_prog_names', 'state']] # 208,810
pv_df[['incentive_prog_names', 'state']]


def reference_fill(df_target, col1, col2, df_ref, col3, col4):
  '''Fill in missing values in df_fill based on known values from df_ref column'''
  # iterate over the known values in reference column
  ls = list(df_ref[col3])
  for item in df_ref[col3]:
    # 1. REFERENCE: find position of unique item in column
    # 2. REFERENCE: extract row and contents
    # 3. TARGET: find position to fill
    # 4. TARGET: fill in value in found position with reference value
    # REFERENCE
    try:
      row = df_ref[(df_ref[col3]==item)].index[0]
      name = df_ref.loc[row, col4] # replacement value
      # TARGET
      rows = df_target[(df_target.loc[:, col1]==item)].index
      df_target.loc[rows, col2] = df_target.loc[rows, col2].fillna(name)
    except:
      pass

reference_fill(pv_df, 'state', 'incentive_prog_names', incent_df, 'state', 'incentive_program')
pv_df.info()

incent_df.head() # REFERENCE DF, 'INCENTIVE_PROGRAM', 'STATE'
pv_df.head() # TARGET DF, 'INCENTIVE_PROG_NAMES', 'STATE'

# INSOLATION
sol_df.head()
pv_df.head()

reference_fill(pv_df, 'state_short', 'annual_insolation', sol_df, 'states', 'solar_rad')
missing_val_count(pv_df, 'annual_insolation', True) # 225,394
pv_df['annual_insolation'].describe()

# missing city for capital, DC
# fill with 'washington'
missing_val_count(pv_df, 'annual_insolation', True) # missing only in DC
pv_df[pv_df['state_short']=='dc']
rows = pv_df[(pv_df.loc[:, 'state_short']=='dc')].index
pv_df.loc[rows, 'city'] = pv_df.loc[rows, 'city'].fillna('washington')


sol_df[sol_df['states']=='dc']
sol_df['states'].nunique()
sol_df[sol_df['states']=='va']

# fill 'insolation' for DC with VA's
rows = pv_df[(pv_df.loc[:, 'state_short']=='dc')].index
pv_df.loc[rows, 'annual_insolation'] = pv_df.loc[rows, 'annual_insolation'].fillna(4.6741948)
missing_val_count(pv_df, 'annual_insolation')

# =============================================================================
# WRITE TO FILE FOR NEXT USE
# =============================================================================
drop_n_reset(pv_df)
pv_df.info()
pv_df.to_csv('pv_df_short.csv', encoding='utf-8', index=False)
pv_df = load_csv('pv_df_short.csv')

# =============================================================================
#
# =============================================================================

# =============================================================================
# States, county, city
# =============================================================================
# count rows for each state
state_count = pv_df['state'].value_counts(dropna=False)
city_count = pv_df['city'].value_counts(dropna=False)
county_count = pv_df['county'].value_counts(dropna=False)
city_count.describe()
county_count.describe()
state_count.describe()
pv_df['county'].value_counts(dropna=False)

# filter states to remove, keep states which appear > 200
states = state_count[state_count<200].astype(str)
states = dict(states)
# create a list of states to remove
state_list = list(states.keys())

# create for-loop to filter out states
# for states to remove, create NaNs in those rows
for item in state_list:
  pv_df[pv_df['state']==item]=np.nan
pv_df.dropna(axis=0, how='all').drop_duplicates().reset_index(drop=True)
pv_df.info()

def row_count_drop(df, col_name, count=100):
  ''' Count amount of rows belonging to each unique item in col_name.
  Remove rows which appear less than "count" times.'''
  # count rows for each state
  count_series = df[col_name].value_counts(dropna=False)
  # filter states to remove, keep states which appear > 100
  count_ = count_series[count_series <= count].astype(str)
  count_ = dict(count_)
  # create a list of states to remove
  count_list = list(count_.keys())
  for item in count_list:
    df[df[col_name]==item]=np.nan
  df = df.dropna(axis=0, how='all').reset_index(drop=True)
  return df
row_count_drop(pv_df, 'state', 200)

# CITY: drop cities below 25th percentile
row_count_drop(pv_df, 'city', np.percentile(city_count, 25))
city_count.describe()
city_count[city_count<=1].astype(str)

# COUNTY: drop counties below 25th percentile
row_count_drop(pv_df, 'county', 1)
county_count.describe()
county_count[county_count<=4].astype(str)
pv_df[pv_df['county']=='indiana']




# =============================================================================
# WRITE FILE 4/8/2018
# =============================================================================

# write to csv partially cleaned file
drop_n_reset(pv_df)
pv_df.to_csv('pv_df_short.csv', encoding='utf-8', na_rep='NA', index=False)
pv_df = load_csv('pv_df_short.csv')


missing_val_count(pv_df, 'city', True)
# ca, contra costa county:
pv_df[pv_df['county']=='contra costa']




# =============================================================================
# SIZE KW
# =============================================================================
# show summary statistics
pv_df.size_kw.describe()
missing_val_count(pv_df, 'size_kw', True)
pv_df[pv_df['size_kw'].isnull()]=np.nan
drop_n_reset(pv_df)

# =============================================================================
# INCENTIVE_PROG_NAMES
# =============================================================================
# explore the missing values
missing_val_count(pv_df, 'incentive_prog_names', True)
pv_df.incentive_prog_names.describe()
pv_df.incentive_prog_names.unique()

# unique values
pv_df['incentive_prog_names'].nunique()

# frequency count of programs
pv_df.groupby(['incentive_prog_names'])['state'].value_counts().sort_values()


# REBATE for each state
# median rebate offered by each program
pv_df.groupby(['incentive_prog_names'])['rebate'].median().sort_values()

pv_df.groupby(['state'])['rebate'].median().sort_values()
pv_df.groupby(['state'])['cost'].median().sort_values()





# =============================================================================
# MISSING VALUES: cost
# =============================================================================
missing_val_count(pv_df, 'cost', True) # 266,670
pv_df['cost'].describe()

# fill cost by install type
# groupby install_type, and calculate median size for each


# =============================================================================
# # SUMMARY STATS: cost_per_watt
# =============================================================================
pv_df['cost_per_watt'].describe()
missing_val_count(pv_df, 'cost_per_watt', True) # 266,914
missing_val_count(pv_df, 'size_kw', True)

# explore the columns
pv_df[['cost_per_watt', 'cost', 'size_kw', 'annual_pv_prod']]

# =============================================================================
# MISSING VALUES: cost_per_watt
# =============================================================================

# FILL COST_PER_WATT
# cost per watt = cost / kw*1000
# fill in missing COST_PER_WATT based on cost and size
pv_df['cost_per_watt'] = pv_df['cost_per_watt'].fillna(pv_df['cost'] / (pv_df['size_kw']*1000))

# FILL COST
# cost = cost_per_watt * (size*1000)
# fill in missing COST values based on SIZE and COST_PER_WATT
fill_cost = (pv_df['cost_per_watt']*(pv_df['size_kw']*1000))
pv_df['cost'] = pv_df['cost'].fillna(fill_cost)

cost = pv_df['cost'] / (pv_df['size_kw']*1000)
cost.describe()

# =============================================================================
# # SUMMARY STATS: annual_PV_prod (estimated production)
# =============================================================================

# having all values for 'size_kw', fill in missing values in 'annual_pv_prod'
pv_df['annual_pv_prod'].describe()
missing_val_count(pv_df, 'annual_pv_prod',True) # 226010
pv_df[['annual_pv_prod', 'size_kw', 'annual_insolation']]


# annual energy production = annual_pv_produced/system_size
common_denom = pv_df['annual_pv_prod']/pv_df['size_kw']
common_denom.describe()
np.count_nonzero(common_denom.isnull().values) # 239608

# fill missing values within common_denom using its median
common_denom = common_denom.fillna(common_denom.median())

# infered annual production = common_denom*size
annual_pv_infer = common_denom*pv_df['size_kw']
annual_pv_infer.describe()

# check the error between actual and calculated energy production
error = abs(annual_pv_infer - pv_df['annual_pv_prod'])/pv_df['annual_pv_prod']
error.describe()
# fill missing values
pv_df['annual_pv_prod'] = pv_df['annual_pv_prod'].fillna(annual_pv_infer)
pv_df['annual_pv_prod'].describe()


# =============================================================================
# SUMMARY STATS: reported_annual_energy_prod
# =============================================================================
# count and explore missing values
# 833,466 missing values
missing_val_count(pv_df, 'reported_annual_energy_prod', True)
# this may be too many missing values to fill as it may lead to a lrager error

pv_df['reported_annual_energy_prod'].describe()

pv_df[['reported_annual_energy_prod', 'annual_pv_prod']]


# =============================================================================
# MISSING VALUES: reported_annual_energy_prod
# =============================================================================

# calculate error between estimated and reported annual PV production
error = abs(pv_df['reported_annual_energy_prod'] - pv_df['annual_pv_prod'])/pv_df['annual_pv_prod']
error.describe()
error.mean() # 0.1083
error.median() # 0.0691

# fill REPORTED_ANNUAL_ENERGY_PROD missing values based on error from estimated ANNUAL_PV_PROD
infer_reported = pv_df['annual_pv_prod']*error.mean()
pv_df['reported_annual_energy_prod'] = pv_df['reported_annual_energy_prod'].fillna(infer_reported)
pv_df['reported_annual_energy_prod'].describe()


# =============================================================================
# SUMMARY STATS: annual_insolation
# use API to fill missing values
# =============================================================================
missing_val_count(pv_df, 'annual_insolation')
pv_df['annual_insolation'].describe()

# explore: insolation rate for each state
# insolation too high for some states: DO NOT USE
state_insol = pv_df.groupby('state')['annual_insolation'].mean()
state_insol.sort_values()
state_insol.describe()

# =============================================================================
# install type, consolidate
# =============================================================================
pv_df['install_type'].unique()

pv_df[pv_df['install_type']=="customer"]
pv_df[pv_df['install_type']=="unknown"]
pv_df['install_type'].str.contains("gov't/np").value_counts()

# consolidate all commercial type
comr_ls = ['commerical','commercial - agriculture', 'small business', 'commercial - small business',
               'commercial - builders', 'commercial - other', 'commercial']
pv_df['install_type'] = pv_df['install_type'].replace(comr_ls, 'commercial')

# government
pv_df['install_type'] = pv_df['install_type'].replace("gov't/np", 'government')

# educational
pv_df['install_type'] = pv_df['install_type'].replace("education", 'educational')

# agricultural
pv_df['install_type'] = pv_df['install_type'].replace("agriculture", 'agricultural')

# residential
pv_df['install_type'] = pv_df['install_type'].replace('residential/sf', 'residential')

# residential
pv_df['install_type'] = pv_df['install_type'].replace('not stated', 'unknown')

# view the missing values
pv_df[pv_df['install_type'].isnull()]

# fill missing values with 'unknown'
pv_df['install_type'] = pv_df['install_type'].fillna('unknown')

pv_df['install_type'] = pv_df['install_type'].astype('category')

# =============================================================================
# WRITE/LOAD FILE
# =============================================================================
pv_df.to_csv('pv_df_short.csv', encoding='utf-8', na_rep='NA', index=False)
pv_df = load_csv('pv_df_short.csv')

drop_n_reset(pv_df)
pv_df.info()
pv_df['city'].nunique()

# extract city and state where the county is missing
county_missing = pv_df[pv_df['county'].isnull()]
city_state = county_missing[['city', 'state']].drop_duplicates()

# =============================================================================
# Population filter
# =============================================================================

pv_df['population'].describe()
# median population
pv_df['population'].median()

# filter DF, filter by population
pv_pop = pv_df[pv_df['population']>3.688800e+04]
pv_pop = pv_pop.reset_index(drop=True)
pv_pop.info()

# 'installer' contains mistakes
pv_pop['installer'].unique()
pv_pop['installer'] = pv_pop['installer'].fillna('unknown')

pv_pop['utility'].unique()
pv_pop['utility'] = pv_pop['utility'].fillna('unknown')


# missing 'counties'
# use API to find missing counties given the known city
missing_val_count(pv_pop, 'county', True)
county = pv_pop[pv_pop['county'].isnull()]
county['city'].nunique()

# missing 'cost'
missing_val_count(pv_pop, 'cost', True)

# =============================================================================
#
# =============================================================================
def reference_fill(df_target, col1, col2, df_ref, col3, col4):
  '''Fill in missing values in df_fill based on known values from df_ref column'''
  # iterate over the known values in reference column
  ls = list(df_ref[col3])
  for item in df_ref[col3]:
    # 1. REFERENCE: find position of unique item in column
    # 2. REFERENCE: extract row and contents
    # 3. TARGET: find position to fill
    # 4. TARGET: fill in value in found position with reference value
    # REFERENCE
    try:
      row = df_ref[(df_ref[col3]==item)].index[0]
      name = df_ref.loc[row, col4] # replacement value
      # TARGET
      rows = df_target[(df_target.loc[:, col1]==item)].index
      df_target.loc[rows, col2] = df_target.loc[rows, col2].fillna(name)
    except:
      pass


# fill in incentives' missing values
# use incent_df as a reference to fill in missing values of incentives for state
reference_fill(pv_pop, 'state', 'incentive_prog_names', incent_df, 'state', 'incentive_program')
# count missing values
missing_val_count(pv_pop, 'incentive_prog_names') # 703
missing_val_count(pv_pop, 'annual_insolation') # 6943


# CITY COUNT
city_count = pv_pop['city'].value_counts(dropna=False)
city_count.describe()
row_count_drop(pv_pop, 'city', 2)
pv_pop = pv_pop.dropna(axis=0, how='all')
pv_pop = pv_pop.reset_index(drop=True)


# =============================================================================
# INSOLATION
# =============================================================================
sol_df.head()
pv_df.head()

reference_fill(pv_pop, 'state_short', 'annual_insolation', sol_df, 'states', 'solar_rad')
missing_val_count(pv_pop, 'annual_insolation', True)
pv_df['annual_insolation'].describe()

# missing city for washington, DC
# fill with 'washington'
missing_val_count(pv_df, 'annual_insolation', True) # missing only in DC
pv_df[pv_df['state_short']=='dc']
rows = pv_df[(pv_df.loc[:, 'state_short']=='dc')].index
pv_df.loc[rows, 'city'] = pv_df.loc[rows, 'city'].fillna('washington')

# use neighbor's value for annual insolation
sol_df[sol_df['states']=='dc']
sol_df['states'].nunique()
sol_df[sol_df['states']=='va']

# fill 'insolation' for DC with VA's
rows = pv_df[(pv_df.loc[:, 'state_short']=='dc')].index
pv_df.loc[rows, 'annual_insolation'] = pv_df.loc[rows, 'annual_insolation'].fillna(4.6741948)
# check again for missing values
missing_val_count(pv_df, 'annual_insolation')



# write file
# =============================================================================

pv_pop[pv_pop['install_type'] == 'unknown']['size_kw'].describe()
# replace the unknown category with residential
# size and costs appear to be very similiar
pv_pop['install_type'] = pv_pop['install_type'].str.replace('unknown', 'residential')
pv_pop['install_type'] = pv_pop['install_type'].astype('category')
pv_pop = pv_pop.dropna(axis=0, how='all').reset_index(drop=True)

# write file
pv_pop.to_csv('pv_pop_clean.csv', encoding='utf-8', na_rep='NA', index=False)

pv_pop.info()



