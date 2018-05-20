# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 16:16:11 2018

@author: mouz
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import json
from pandas.io.json import json_normalize

# =============================================================================
# XML
# =============================================================================
import urllib3
from xml.etree import ElementTree as ET

address='washington,dc'; sys_cap=3; azimuth=180; tilt=40; tracking=1
url = 'https://developer.nrel.gov/api/pvwatts/v5.xml?api_key={API_KEY}&address={address}&system_capacity={sys_cap}&azimuth={azimuth}&tilt={tilt}&array_type={tracking}&module_type=1&losses=10'
url_new = url.format(API_KEY='0DkoSjFm3OJ21FRtM05Smfi9bPNoFRcJHpFNgNJw',address=address, sys_cap=sys_cap, azimuth=azimuth, tilt=tilt, tracking=tracking)
# Package the request, send the request and catch the response: r
r = requests.get(url_new)
xm = r.content
root = ET.iterparse(xm)
type(root)
for child in root.iter('*'):
  print(child.tag)

for child in root.iter():
  print(child.attrib, child.tag)




# =============================================================================
# APIs
# =============================================================================

# CREATE ADDRESS
# create city, state combo for use in the APIs
df = pv_df[['city', 'state', 'county']]
df = pv_df[['state', 'county']].drop_duplicates()
len(df)
# concat city and state; county and state
# create an address list
county_state_ls = list(df['county'] + ', ' + df['state'])
len(county_state_ls)
# create a state list
state_ls = list(pv_df['state'].unique())
len(state_ls)


# =============================================================================
# SOLAR RADIATION
# =============================================================================
address='washington,dc'; sys_cap=3; azimuth=180; tilt=40; tracking=1

cities = []
states = []
solrad_ = []
url = 'https://developer.nrel.gov/api/pvwatts/v5.json?api_key={API_KEY}&address={address}&system_capacity={sys_cap}&azimuth={azimuth}&tilt={tilt}&array_type={tracking}&module_type=1&losses=10'
for address in county_state_ls:
  url_new = url.format(API_KEY='0DkoSjFm3OJ21FRtM05Smfi9bPNoFRcJHpFNgNJw',address=address, sys_cap=sys_cap, azimuth=azimuth, tilt=tilt, tracking=tracking)
  # Package the request, send the request and catch the response: r
  req = requests.get(url_new)
  programs_data = req.json()
  programs_data.keys()
  try:
    city = programs_data['station_info']['city']
    state = programs_data['station_info']['state']
    solrad = programs_data['outputs']['solrad_annual']
    cities.append(city)
    states.append(state)
    solrad_.append(solrad)
  except:
    pass

sol_df = pd.DataFrame({'cities': cities, 'states':states, 'solar_rad':solrad_})
# clean df; drop duplicates, remove underscore, and reset index
sol_df = sol_df.drop_duplicates()
sol_df['cities'] = sol_df['cities'].str.replace('_', ' ')
sol_df = sol_df.reset_index(drop=True)

# convert to lower case
sol_df = upper_to_lower(sol_df)

# write file
sol_df.to_csv('solar_rad_state.csv', encoding='utf-8', index=False)


sol_df['states'].nunique()
sol_df['cities'].nunique()





# =============================================================================
# UTILITY RATES
# =============================================================================
residential_pv = pd.read_csv('residential_pv.csv', sep=',', low_memory=False)
residential_pv.head()

# rates units: $/kWh

address_in = residential_pv.loc[:, 'city'] + str(',') + residential_pv.loc[:, 'state']
address_in = address_in.drop_duplicates()
len(address_in.drop_duplicates())
# iterate over indexe
address_in.index.values

url = 'https://developer.nrel.gov/api/utility_rates/v3.json?api_key={API_KEY}&address={address}'
url_new = url.format(API_KEY='0DkoSjFm3OJ21FRtM05Smfi9bPNoFRcJHpFNgNJw', address=address_in[0])
req = requests.get(url_new)
programs_data = req.json()

type("residential" in req.text)
req.text.findall('residential')
req.text[389:]

utility_name = programs_data['outputs']['utility_info'][0]['utility_name']
res_rate = programs_data['outputs']['residential']
comm_rate = programs_data['outputs']['commercial']


url = 'https://developer.nrel.gov/api/utility_rates/v3.json?api_key={API_KEY}&address={address}'
for i in address_in.index.values:
  try:
    address_in = residential_pv.loc[i, 'city'] + str(',') + residential_pv.loc[i, 'state']
    url_new = url.format(API_KEY='0DkoSjFm3OJ21FRtM05Smfi9bPNoFRcJHpFNgNJw', address=address_in)
    req = requests.get(url_new)
    util_data = req.json()
    # extract info
#    utility_name = programs_data['outputs']['utility_info'][0]['utility_name']
    res_rate = util_data['outputs']['residential']
    # store value in new column at index i
    residential_pv.loc[i, 'rates'] = res_rate
  except:
    pass

residential_pv.info()
residential_pv.loc[:, 'address'] =  residential_pv.loc[:, 'city'] + str(',') + residential_pv.loc[:, 'state']
residential_pv['rates'] = residential_pv.sort_values(by='address')['rates'].ffill()
# find missing value's index
residential_pv[residential_pv['rates'].isnull()].index
residential_pv.loc[164723:164725, ['address', 'rates']]
# fill missing value
residential_pv['rates'] = residential_pv['rates'].fillna(0.1098)
# write file
residential_pv.to_csv('residential_pv.csv',  encoding='utf-8', na_rep='NA', index=False)

#
# =============================================================================
# sample DF
sample_df = pv_pop.sample(20).reset_index(drop=True)

url = 'https://developer.nrel.gov/api/utility_rates/v3.json?api_key={API_KEY}&address={address}'
for i in range(len(pv_pop)):
  try:
    address_in = pv_pop.loc[i, 'city'] + str(',') + pv_pop.loc[i, 'state']
    url_new = url.format(API_KEY='0DkoSjFm3OJ21FRtM05Smfi9bPNoFRcJHpFNgNJw', address=address_in)
    req = requests.get(url_new)
    programs_data = req.json()
    # extract info
    utility_name = programs_data['outputs']['utility_info'][0]['utility_name']
    res_rate = programs_data['outputs']['residential']
    comm_rate = programs_data['outputs']['commercial']
    # if residential type, store residential value
    if pv_pop.loc[i, 'install_type'] == 'residential':
      pv_pop.loc[i, 'rates'] = res_rate
    else:
      pv_pop.loc[i, 'rates'] = comm_rate
  except:
    pass

# write file
pv_pop.to_csv('pv_pop_clean.csv', encoding='utf-8', na_rep='NA', index=False)

pv_pop.loc[3, 'rates']
pv_pop.head()


# create sample df
sample_df.loc[5:10, ['city', 'size_kw', 'rates']]
sample_df['dol_per_yr'].describe()
sample_df['reported_annual_energy_prod'].describe()

sample_df['dol_per_yr'] = sample_df['reported_annual_energy_prod'] * sample_df['rates']
sample_df.head()

pv_pop[pv_pop['city'] == 'new york']
pv_pop[pv_pop['population'] > 1000000].info()




# =============================================================================
# # INCENTIVES BY STATE
# =============================================================================
incentive_missing = pv_pop[pv_pop['incentive_prog_names'].isnull()]
incentives = incentive_missing[['city', 'state', 'incentive_prog_names']].drop_duplicates()
address_list = list(incentives['city'] + ', ' + incentives['state'])
address_list[0]

# test
# =============================================================================
address_in = residential_pv.loc[56, 'city'] + str(',') + residential_pv.loc[56, 'state']
url = 'https://developer.nrel.gov/api/energy_incentives/v2/dsire.json?api_key={API_KEY}&address={address}&category=solar_technologies&technology=solar_photovoltaics'
url_new = url.format(API_KEY='0DkoSjFm3OJ21FRtM05Smfi9bPNoFRcJHpFNgNJw',address=address_in)
req = requests.get(url_new)
programs_data = req.json()
print(programs_data.keys())

# extract the name of program and rebate amount
state = []
incentive = []
for item in address_list:
  url = 'https://developer.nrel.gov/api/energy_incentives/v2/dsire.json?api_key={API_KEY}&address={address}&category=solar_technologies&technology=solar_photovoltaics'
  url_new = url.format(API_KEY='0DkoSjFm3OJ21FRtM05Smfi9bPNoFRcJHpFNgNJw',address=item)
  req = requests.get(url_new)
  programs_data = req.json()

  try:
    # iterate over each program
    num_prog = len(programs_data['result'])
    for i in range(num_prog):
      state_name = (programs_data['result'][i]['regions'][0]['name']).lower()
      incentive_name = programs_data['result'][i]['program_name']
#      category = (programs_data['result'][i]['category_name']).lower()
      regions = programs_data['result'][i]['regions'][0]['type']

      # extract info: state, and incentive program for state
      # if program is for the state (not federal), append info to list
      if regions == 'state':
        state.append(state_name)
        incentive.append(incentive_name)
  except:
    pass

incentive_df = pd.DataFrame({'state': state, 'incentive_program': incentive})
incentive_df = incentive_df.drop_duplicates().reset_index(drop=True)
# lower case
incentive_df['incentive_program'] = incentive_df['incentive_program'].str.lower()
# write file
incentive_df.to_csv('incentives_state.csv', index=False)



# =============================================================================
#
# =============================================================================


# =============================================================================
# POPULATION OF CITIES
# =============================================================================
cities = []
states = []
population = []
for i in range(999):
  url = 'https://public.opendatasoft.com/api/records/1.0/search/?dataset=1000-largest-us-cities-by-population-with-geographic-coordinates&rows=1000&sort=-rank&facet=city&facet=state'
  req = requests.get(url)
  pop_data = req.json()
  pop_data.keys()
  city = pop_data['records'][i]['fields']['city']
  state = pop_data['records'][i]['fields']['state']
  pop = pop_data['records'][i]['fields']['population']
  cities.append(city)
  states.append(state)
  population.append(pop)

pop_df = pd.DataFrame({'cities': cities, 'states':states, 'population':population})
# lower case
pop_df = upper_to_lower(pop_df)

# wrtie to csv
pop_df.to_csv('population_df.csv', index=False)
# read csv
pop_df = pd.read_csv('population_df.csv', sep=',', low_memory=False)
pop_df['states'].unique()
pop_df['cities'].nunique()
pop_df[pop_df['states']=='district of columbia']



# =============================================================================
# county, state, city, zipcode
# =============================================================================
import requests
pv_pop.info()

# extract city and state where the county is missing
county_missing = pv_pop[pv_pop['county'].isnull()]
city_state = county_missing[['city', 'state']].drop_duplicates()

# create an address list
city = []
state = []
for index, item in city_state.iterrows():
#  city_name = item[0]
#  print(city_name)
  city.append(item[0]) # city
  state.append(item[1]) # state
len(city)

# =============================================================================
#
auth_token = 'hVP4IDlYlhogVQQjPCkr'
auth_id = '1fed214b-f8a4-6319-10f1-a36570671f4b'
url = 'https://us-zipcode.api.smartystreets.com/lookup?city={city}&state={state}&auth-id=1fed214b-f8a4-6319-10f1-a36570671f4b&auth-token=hVP4IDlYlhogVQQjPCkr'
cities = []
states = []
zipcodes = []
counties = []
for index, item in city_state.iterrows():
  city_in=str(item[0])
  state_in=str(item[1])
  url_new = url.format(city=city_in, state=state_in)
  req = requests.get(url_new)
  pop_data = req.json()
  try:
  #  print(city_in)
  #  print(pop_data[0].keys())
    data_len = int(len(pop_data[0]['zipcodes']))
    for i in range(data_len):
      zipcode = pop_data[0]['zipcodes'][i]['zipcode']
      county = pop_data[0]['zipcodes'][i]['county_name']
      cities.append(city_in)
      states.append(state_in)
      zipcodes.append(zipcode)
      counties.append(county)
  except:
    pass

address_df = pd.DataFrame({'city': cities, 'state':states, 'county':counties, 'zipcode':zipcodes})
#
# =============================================================================

# try
url = 'https://us-zipcode.api.smartystreets.com/lookup?city=astoria&state=ny&auth-id=1fed214b-f8a4-6319-10f1-a36570671f4b&auth-token=hVP4IDlYlhogVQQjPCkr'
req = requests.get(url)
pop_data = req.json()
pop_data[0].keys()
len(pop_data[0]['city_states'])
pop_data[0]['zipcodes'][5]['zipcode']
pop_data[0]['zipcodes'][5]['state']





# =============================================================================
# GOOGLE MAPS API
# =============================================================================

address_in = pv_pop.loc[250, 'city'] + str(',') + pv_pop.loc[250, 'state']
key = 'AIzaSyDbC_tSCjq7WE8HsrsDulmvVwfgJSce5fg'

for i in range(len(pv_pop)):
    address_in = pv_pop.loc[i, 'city'] + str(',') + pv_pop.loc[i, 'state']
    try:
    # Geocoding an address
      geocode_result = gmaps.geocode(address_in)
      result_len = len(geocode_result[0]['address_components'])
    # iterate through list to find 'County'
      for i in range(result_len):
        result = geocode_result[0]['address_components'][i]['long_name']
        if 'County' in result:
          if pv_pop.loc[i, 'county'] == np.nan:
            pv_pop.loc[i, 'county'] = result
            break
  except:
    pass





#
# =============================================================================
import googlemaps
key = 'AIzaSyDbC_tSCjq7WE8HsrsDulmvVwfgJSce5fg'

# extract city and state where the county is missing
county_missing = pv_pop[pv_pop['county'].isnull()]
city_state = county_missing[['city', 'state']].drop_duplicates().dropna().reset_index(drop=True)
# creeate empty column to fil
city_state['county'] = np.nan
city_state.loc[:, 'county']
len(city_state)

# create address list
address_ls = city_state['city'] + ', ' + city_state['state']

cities = []
states = []
counties = []
key = 'AIzaSyDbC_tSCjq7WE8HsrsDulmvVwfgJSce5fg'
for item in address_ls:
  try:
    # Geocoding an address
    geocode_result = gmaps.geocode(item)
    county = geocode_result[0]['address_components'][1]['long_name']
    city = geocode_result[0]['address_components'][0]['long_name']
    state = geocode_result[0]['address_components'][2]['long_name']
    counties.append(county)
    cities.append(city)
    states.append(state)
  except:
    pass

len(counties)

city_state['county'] = counties
city_state.to_csv('city_state_county.csv', encoding='utf-8', index=False)
city_state = pd.read_csv('city_state_county.csv', sep=',')

# fix counties
# explore: where county is not found
sub_county = (city_state[city_state['county'].str.contains('County')==False]).reset_index(drop=True)
sub_list = sub_county['city'] + ', ' + sub_county['state']
sub_county['county'] = 'unknown'



# =============================================================================
gmaps = googlemaps.Client(key='AIzaSyDbC_tSCjq7WE8HsrsDulmvVwfgJSce5fg')
counties2 = []
for item in sub_list:
  try:
    # Geocoding an address
    geocode_result = gmaps.geocode(item)
    result_len = len(geocode_result[0]['address_components'])
    # iterate through list to find 'County'
    for i in range(result_len):
      result = geocode_result[0]['address_components'][i]['long_name']
      if 'County' in result:
        # locate row to use for replacement
        row = sub_list[sub_list==item].index[0]
        sub_county.loc[row, 'county'] = sub_county.loc[row, 'county'].replace('unknown', result)
        break
  except:
    pass
print(sub_county)

# replace 'unknown' county names with its city name
for item in sub_county['county']:
  if item =='unknown':
    row = sub_county[sub_county.loc[:, 'county']==item].index
    city_name = sub_county.loc[row, 'city']
    sub_county.loc[row, 'county'] = sub_county.loc[row, 'county'].replace('unknown', city_name)

# remove label 'County'
sub_county['county'].str.replace('County', '')

# =============================================================================



