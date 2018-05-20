# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 10:21:55 2018

@author: mouz
"""

# =============================================================================
# EDA
# =============================================================================

'''
GOALS:

1. Which states have the cheapest and most expensive installations; which states have highest incentives?
2. How have the prices changed over the years?
3. Which factors contribute the most to the total cost?
4. Predict the total cost before any incentives given the size of an installation.

'''


# =============================================================================
# LOAD FILE
# =============================================================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

# read
pv_pop = pd.read_csv('pv_pop_clean.csv', sep=',', low_memory=False)
pv_pop = pv_pop.reset_index(drop=True)
pv_pop.info()

pv_pop.describe()
pv_pop['install_type'].unique()
# =============================================================================
# TAXES
# =============================================================================
# rebate_cost_ratio = rebate/cost
pv_pop['rebate_cost_ratio'] = (pv_pop['rebate'] / pv_pop['cost'])
pv_pop['rebate_cost_ratio'].describe()

pv_pop['sales_tax_percent'] = pv_pop['sales_tax_cost'] / pv_pop['cost'] * 100
pv_pop['sales_tax_percent'].describe()


plt.figure(figsize=(10,6))
tax_group = pv_pop.groupby('state')['sales_tax_percent'].mean().sort_values()
tax_group['maine':'new hampshire'].plot(kind='bar')
plt.title('Mean Tax Rate')
plt.ylabel('Tax Rate (%)')
plt.xlabel('State')
plt.xticks(rotation=75)
plt.margins(0.02)
plt.tight_layout()
plt.show()


pv_pop[pv_pop['state'] == 'new hampshire'].sales_tax_percent.describe()
pv_pop[pv_pop['state'] == 'new hampshire']



# =============================================================================
# REBATE_COST_RATIO
# =============================================================================

plt.figure(figsize=(10,6))
rebate_cost_ratio = pv_pop.groupby('state')['rebate_cost_ratio'].mean().sort_values()
rebate_cost_ratio['maryland':'texas'].plot(kind='bar')
plt.title('Mean Tax Rate')
plt.ylabel('rebate_cost_ratio')
plt.xlabel('State')
plt.xticks(rotation=75)
plt.margins(0.02)
plt.show()


# =============================================================================
#
# =============================================================================
plt.figure(figsize=(10,6))
plt.scatter(x=residential_sub['size_kw'], y=residential_sub['cost'],
            s=residential_sub['rebate']/20,
            alpha=0.3, marker='.')
plt.xlabel('Size (kW)')
plt.ylabel('Cost ($)')
plt.title('Cost vs Size')
plt.show()



# =============================================================================
#
# =============================================================================
pv_pop[pv_pop['cost'].isnull()][['size_kw', 'install_type']].describe()
pv_pop[pv_pop['install_type'] != 'residential'].describe()



# =============================================================================
# INSTALL TYPE MEDIAN COST
# =============================================================================
pv_pop['install_type'].unique()
type_group = pv_pop.groupby('install_type')['cost'].median().sort_values()
type_group['residential':'government'].plot(kind='bar')
plt.title('Median Cost')
plt.ylabel('Cost')
plt.xlabel('Type of Installation')
plt.xticks(rotation=60)
plt.tight_layout()
plt.show()





# =============================================================================
# INSTALL TYPE MEDIAN REBATE
# =============================================================================
type_group2 = pv_pop.groupby('install_type')['rebate'].median().sort_values()
type_group['residential':'government'].plot(kind='bar')
plt.title('Median Cost')
plt.ylabel('Cost')
plt.xlabel('Type of Installation')
plt.xticks(rotation=60)
plt.tight_layout()
plt.show()

# strip plot
# DO NOT USE
sns.stripplot(y='cost', x='install_type', data=pv_pop.sort_values(['cost']), jitter=True)
plt.xticks(rotation=60)
plt.show()

# =============================================================================
# INSTALL TYPE CAPACITY
# =============================================================================
# SIZE, bar graph
install_type_size = pv_pop.groupby('install_type')['size_kw'].median().sort_values()
install_type_size[:'government'].plot(kind='bar')
type_group['residential':'government'].plot(kind='bar')
plt.title('Median Size')
plt.ylabel('Size')
plt.xlabel('Type of Installation')
plt.xticks(rotation=60)
plt.tight_layout()
plt.show()

# PLT Bar Plot
# =============================================================================
left = np.arange(len(install_type_size[:'government']))
height = type_group['residential':'government'].values

plt.bar(left, height, width=0.5)
plt.xticks(range(len(type_group)), type_group['residential':'government'].keys(), rotation = 45)
plt.title('Median Size')
plt.ylabel('Size')
plt.xlabel('Type of Installation')
plt.tight_layout()
plt.show()

# population distribution
pv_pop['population'].describe()
pv_pop['population'].plot(kind='area')
plt.show()

# =============================================================================
# INSTALL TYPE COUNT
# =============================================================================

install_count = pv_pop['install_type'].value_counts().sort_values()
install_count.plot(kind='bar')
plt.title('Dataset Distribution')
plt.ylabel('Frequency')
plt.xlabel('Type of Installation')
plt.xticks(rotation=60)
plt.margins(0.02)
plt.tight_layout()
plt.show()




# =============================================================================
# RESIDENTIAL EXPLORATION
# =============================================================================

# subset residential install type
residential_sub = pv_pop[pv_pop['install_type']=='residential']

residential_sub['size_kw'].describe()

# scatter plot
# Cost vs Size
plt.scatter(x=residential_sub['size_kw'], y=residential_sub['sales_tax_cost'], marker='.', alpha=0.5)
plt.xlabel('Size (kW)')
plt.ylabel('Cost ($)')
plt.title('Cost vs Size')
#plt.xlim([0, 750])
#plt.ylim([0, 3000000])
plt.show()

# =============================================================================
# OUTLIERS
# =============================================================================

#
residential_sub[residential_sub.size_kw==737319]
# size: 737,319 kW
# produced (annual): 124,569,500 kWh
124569500 / 12 # kWh per month
10380791.666666666 / 30 # kWh per day
# 346026.3888888889 kWh per day

# average home usage monthly = 900kWh
# monthly kWh = (produced / 12 months)
# 900 * 3STND

# FILTER
# =============================================================================

# filter by percentile
percent90 = np.percentile(residential_sub['annual_pv_prod'], 90)
res_sub90 = residential_sub[(residential_sub['annual_pv_prod']) < percent90][['annual_pv_prod', 'reported_annual_energy_prod', 'cost', 'size_kw']]
res_sub90.describe()
sns.boxplot(x=res_sub90['annual_pv_prod'])
sns.boxplot(x=res_sub90['size_kw'])

# scatter plot
# Cost vs Size
plt.scatter(x=res_sub90['size_kw'], y=res_sub90['cost'], alpha=0.5)
plt.xlabel('Size (kW)')
plt.ylabel('Cost ($)')
plt.title('Cost vs Size')
#plt.xlim([0, 750])
#plt.ylim([0, 3000000])
plt.show()

# filter the outliers
res_sub90 = residential_sub[(residential_sub['annual_pv_prod']) < percent90]
res_sub90 = res_sub90.reset_index(drop=True)
res_sub90.info()
residential_sub = res_sub90

residential_sub.describe()

# write residential subset to file
#residential_sub.to_csv('residential_pv.csv', encoding='utf-8', na_rep='NA', index=False)


# =============================================================================
# CUT INTERVAL FOR SIZE
# =============================================================================
size = residential_sub['size_kw']
size.describe()

residential_sub['size_catg'] = pd.qcut(size.values, 3, labels=['small', 'normal', 'large'])
residential_sub.head()

# create time-series plots for each category

residential_sub.groupby('size_catg')['size_kw'].describe()




# =============================================================================
# REGIONS COUNT
# =============================================================================
# STATE COUNT
state_count = residential_sub['state'].value_counts(dropna=False)
state_count.describe()
state_count['california':'arkansas'].plot(kind='bar')
plt.xticks(rotation=60)
plt.ylabel('Count')
plt.xlabel('State')
plt.show()

# state insolation
state_group_insol = residential_sub.groupby('state')['annual_insolation'].mean().sort_values()
state_group_insol.plot(kind='bar')
plt.xticks(rotation=60)
plt.ylabel('Insolation (kWh/m2/day)')
plt.xlabel('State')
plt.show()

# states' capacity
state_group_size = residential_sub.groupby('state')['size_kw'].mean().sort_values()
state_group_size.plot(kind='bar')
plt.xticks(rotation=60)
plt.ylabel('Size (kW)')
plt.xlabel('State')
plt.show()

#hue='origin',
sns.lmplot(x='size_kw', y='cost', data=clean_pv,  palette='Set1')

# size and energy scatter
plt.scatter(x=residential_sub['reported_annual_energy_prod'],
            y=residential_sub['size_kw'],
            #s=residential_sub['size_kw'].values*(1/100),
            marker='.',
            alpha=0.3,)
plt.ylabel('Capacity (kW)')
plt.xlabel('Annual Production (Actual, kWh)')
plt.xlim([0,20000])
plt.ylim([0,16])
plt.show()




# states' production
state_group_prod = residential_sub.groupby('state')['reported_annual_energy_prod'].mean().sort_values()
state_group_prod.plot(kind='bar')


left_ = np.arange(len(state_group_prod.index))
height_ = state_group_prod.values
plt.bar(left_, height_, width=0.8)

plt.xticks(left_, state_group_prod.index, rotation=75)
plt.title('Power Generation by State')
plt.ylabel('Production (kW/yr)')
plt.xlabel('State')
plt.show()


# hours of sunlight
# estimated energy / size_kw / 365 days
#
sunlight_hrs_actual = residential_sub['reported_annual_energy_prod'] / (residential_sub['size_kw']*365)
sunlight_hrs_theo = residential_sub['annual_pv_prod'] / (residential_sub['size_kw']*365)



residential_sub[residential_sub.state=='louisiana']
pv_pop[pv_pop.state=='tennessee']
pv_pop.state.unique()


# error between estimated and reported energy production
residential_sub['pv_prod_err'] = residential_sub['reported_annual_energy_prod'] / residential_sub['annual_pv_prod']
residential_sub['pv_prod_err'].describe()
state_group_prod_err = residential_sub.groupby('state')['pv_prod_err'].mean().sort_values()
state_group_prod_err.plot(kind='bar')
plt.xticks(rotation=75)
plt.ylabel('Error')
plt.xlabel('State')
plt.show()



# National Average Cost and Capacity
residential_sub.cost.median()
residential_sub.cost_per_watt.median()
residential_sub.size_kw.median()
residential_sub.annual_pv_prod.median()


residential_sub.size_kw.describe()




# =============================================================================
# SIZE_KW distribution
# =============================================================================
# median size of residential installation
residential_sub['size_kw'].median()


# plot histogram after sorting

mean_si = round(residential_sub['size_kw'].mean(), 2)
median_si = round(residential_sub['size_kw'].median(), 2)
residential_sub['size_kw'].plot(kind='hist', bins=50)
plt.xlabel('System Capacity (kW)')
plt.ylabel('Count')
plt.margins(0.02)
plt.axvline(x=np.mean(residential_sub['size_kw']), color='red')
plt.axvline(x=np.median(residential_sub['size_kw']), color='black')
plt.title('System Size Distribution')
plt.legend(('mean', 'median'), loc='upper right')
plt.text(13, 20000, ('Median:', median_si))
plt.text(13, 19000, ('Mean:', mean_si))
plt.show()

## sub plot
#plt.subplot(2,1,2)
#mean_en = round(residential_sub['reported_annual_energy_prod'].mean(), 2)
#median_en = round(residential_sub['reported_annual_energy_prod'].median(), 2)
#residential_sub['reported_annual_energy_prod'].plot(kind='hist', bins=10000)
#plt.xlabel('reported_annual_energy_prod (kW)')
#plt.ylabel('Count')
##plt.xlim([0,2000])
#plt.margins(0.02)
#plt.axvline(x=np.mean(residential_sub['reported_annual_energy_prod']), color='red')
#plt.axvline(x=np.median(residential_sub['reported_annual_energy_prod']), color='black')
#plt.title('reported_annual_energy_prod')
#plt.legend(('mean', 'median'), loc='upper right')
##plt.text(13, 20000, ('Median:', median_en))
##plt.text(13, 19000, ('Mean:', mean_en))
#plt.show()


# =============================================================================
# Utility Rates Distribution
# =============================================================================
residential_pv = pd.read_csv('residential_pv.csv', sep=',', low_memory=False)

util_rate = residential_pv.groupby('state')['rates'].mean().sort_values()

rate_mean = round(residential_pv['rates'].mean(), 4)
rate_median = round(residential_pv['rates'].median(), 4)

left_rate = np.arange(len(util_rate))
height_rate = util_rate.values

plt.figure(figsize=(12,8))
rect1 = plt.bar(left_rate, height_rate, width=0.8, color='green', alpha=0.6)

plt.xticks(left_rate - 0.25, util_rate.index, rotation = 75)
plt.title('Utility Rates')
plt.ylabel('Price ($ / kWh)')
plt.xlabel('State')
plt.axhline(y=rate_mean, color='red')
#plt.text(0, 28000, s= ('Average cost: ' + str('$') + str(rate_mean)))
#plt.legend((rect1[0], rect2[0]) ,['cost', 'rebate'], loc='upper left')
plt.show()





# =============================================================================
# ECDF
# =============================================================================
def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)
    # x-data for the ECDF: x
    x = np.sort(data)
    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n
    return x, y

# ECDF for SIZE
x, y = ecdf(residential_sub['size_kw'])
#size_theo_ = np.random.normal(np.mean(residential_sub['size_kw'].values), np.std(residential_sub['size_kw'].values), 10000)
#x_thr, y_thr = ecdf(size_theo_)
# Generate plot
plt.plot(x, y, marker='.', linestyle='none', alpha=0.3)
#plt.plot(x_thr, y_thr, marker='.', linestyle='none')
plt.margins(0.02)
plt.ylabel('CDF')
plt.xlabel('System Capacity (kW)')
#plt.xscale('log')
# Display the plot
plt.show()


# =============================================================================
# COST
# =============================================================================
residential_sub['cost'].describe()
residential_sub['cost'].sort_values(ascending=False)
residential_sub[['cost', 'size_kw']].sort_values(['cost'],ascending=False)

# plot cost distribution less than 1mil
# limit x-axis under 200,000
mean_c = round(residential_sub['cost'].mean(), 2)
median_c = round(residential_sub['cost'].median(), 2)

residential_sub['cost'].plot(kind='hist', bins=50)
plt.xlabel('Cost ($)')
plt.ylabel('Count')
plt.margins(0.02)
plt.xlim([0, 100000])
plt.axvline(x=mean_c, color='red')
plt.axvline(x=median_c, color='black')
plt.title('System Cost Distribution')
plt.legend(('mean', 'median'), loc='upper right')
#plt.text(13, 20000, ('Median:', median_c))
#plt.text(13, 19000, ('Mean:', mean_c))
plt.show()





# ECDF
x_, y_ = ecdf(residential_sub['cost'])
plt.plot(x_, y_, marker='.', linestyle='none', alpha=0.3)
plt.margins(0.02)
plt.xscale('log')
plt.ylabel('CDF')
plt.xlabel('cost')
plt.show()
# NOTE: probability of cost being less than 10^3 is about 0.35


# COST PROBABILITY
# =============================================================================
residential_sub['cost'].describe()
# amount of points at medium or below
mean_cost = residential_sub['cost'].mean()
median_cost = residential_sub['cost'].median()
median_cost_pts = len(residential_sub['cost'][residential_sub['cost'] <= median_cost])
# total amount of points
total_pts = len(residential_sub['cost'])
# probability of cost being at or below median cost
cost_prob = median_cost_pts/total_pts

n_ = np.random.binomial(n=total_pts, p=cost_prob, size=10000)
x,y = ecdf(n_)
plt.plot(x, y, marker='.', linestyle='none')
plt.margins(0.02)
plt.xscale('log')
# Label the axes
plt.ylabel('ECDF')
plt.xlabel('cost')
plt.title('Cost Distribution')
plt.show()

np.sum(n_<=137500)/len(n_)

plt.hist(n_, bins=100, normed=True)
plt.margins(0.02)
plt.xlabel('number of successes')
plt.ylabel('probability')
plt.show()

# =============================================================================
# COST AND SIZE
# =============================================================================
def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    # Compute correlation matrix: corr_mat
    corr_mat = np.corrcoef(x,y)

    # Return entry [0,1]
    return corr_mat[0,1]

# scatter plot
# Cost vs Size
plt.scatter(x=residential_sub['size_kw'], y=residential_sub['cost'], s=residential_sub['size_kw']/10, alpha=0.5)
plt.xlabel('Size (kW)')
plt.ylabel('Cost ($)')
plt.title('Cost vs Size')
plt.xlim([0, 750])
plt.ylim([0, 3000000])
plt.show()

# CORRELATION
# =============================================================================
# drop NA before computing correlation
clean_pv = residential_sub.dropna(how='any').reset_index(drop=True)
pearson_r(clean_pv['size_kw'],clean_pv['cost'])

test_ls = ['cost', 'size_kw', 'rebate', 'annual_insolation', 'reported_annual_energy_prod',
                       'annual_pv_prod']
corrs = []
for item in test_ls:
  n = pearson_r(clean_pv[item],clean_pv['cost'])
  print('cost-' + str(item) + str(': ') + str(n))
  corrs.append(n)
print(corrs)

# joint plot
sns.jointplot(x='size_kw', y='cost', data=clean_pv)
plt.show()



#
# =============================================================================
# pair plot
sns.pairplot(clean_pv[['cost', 'size_kw', 'rebate', 'annual_insolation', 'reported_annual_energy_prod',
                       'annual_pv_prod', 'sales_tax_cost', 'cost_per_watt']])
plt.show()
#
# =============================================================================




# linear regression
sns.lmplot(x='size_kw', y='cost', data=residential_sub, markers='.')
plt.xlabel('Capacity (kW)')
plt.ylabel('Cost')
plt.margins(0.02)
#plt.xlim([0, 750])
#plt.ylim([0, 3000000])
plt.show()



# =============================================================================
# Seaborn Scatter plots
# =============================================================================
# jointplot
# kind='scatter', 'reg', 'kde', 'hex', 'resid'
sns.set(style="darkgrid", color_codes=True) #  white, dark, whitegrid, darkgrid, ticks
sns.jointplot(x='size_kw', y='cost',  data=clean_pv, kind='reg')
sns.jointplot(x='size_kw', y='cost',  data=clean_pv, kind='scatter')
sns.jointplot(x='size_kw', y='cost',  data=clean_pv, kind='scatter')
plt.xlim([0, 200])
plt.ylim([0, 150000])
plt.show()



# scatter plot
n = pv_pop.state.value_counts()
plt.scatter(x=residential_sub['size_kw'],
            y=residential_sub['cost'],
            s=residential_sub['size_kw']/10,
            alpha=0.5,
            marker='.')
plt.xlabel('Size (kW)')
plt.ylabel('Cost ($)')
plt.title('Cost vs Size')
plt.margins(0.02)
plt.xlim([0, 750])
plt.ylim([0, 3000000])
plt.show()

# regression plots
sns.regplot(x='size_kw', y='cost', data=residential_sub, scatter=None, label='order 2', color='green')

# residuals plot
sns.residplot(x='size_kw', y='cost', data=residential_sub, color='green')
plt.show()


# =============================================================================
# COST and REBATE by state
# =============================================================================
plt.figure(figsize=(10,6))
residential_sub['after_rebate'] = residential_sub['cost'] - residential_sub['rebate']
residential_cost = residential_sub.groupby('state')[['rebate', 'cost']].mean().sort_values(by='cost')

residential_cost['michigan':'new jersey']
residential_cost['michigan':'new jersey'].plot(kind='bar')
plt.title('Average Cost and Rebate')
plt.ylabel('Cost ($)')
plt.xlabel('State')
plt.xticks(rotation=70)
plt.tight_layout()
plt.show()

# =============================================================================
# BREAK EVEN
# =============================================================================
residential_pv = pd.read_csv('residential_pv.csv', sep=',', low_memory=False)

# =============================================================================
# DATETIME CONVERSION
date_time = '%m/%d/%Y'
datetime = pd.to_datetime(residential_pv['date_installed'], format=date_time)
pv_pop_date = residential_pv.set_index(datetime)
# =============================================================================
pv_after = pv_pop_date.loc['2017':'2018', :]
pv_before = pv_pop_date.loc['1998':'2009', :]

# Compute breakven for different time periods
# =============================================================================
pv_after['rates'].mean()
pv_after['annual_pv_prod'].mean()
pv_after['cost'].mean()
pv_after['rebate'].mean()

# Federal tax credit = 30%
pv_after.loc[:, 'fed_tax_cred'] = 0.30 * pv_after['cost']
pv_after.loc[:, 'annual_cost'] = (pv_after['rates'] * pv_after['annual_pv_prod'])
pv_after.loc[:, 'combined_cost'] = pv_after['cost'] + pv_after['sales_tax_cost'] - pv_after['rebate'] - pv_after['fed_tax_cred']
# if no rebates and or tax data available
pv_after.loc[:, 'combined_cost']  = pv_after['cost'] - pv_after['fed_tax_cred']
pv_after.loc[:, 'break_even_yrs'] = pv_after['combined_cost'] / pv_after['annual_cost']
pv_after.loc[:, 'break_even_yrs'].describe()

pv_after[pv_after['break_even_yrs'] <= 0][['break_even_yrs', 'cost', 'rebate', 'combined_cost', 'fed_tax_cred', 'break_even_yrs']]


#
# =============================================================================
pv_before = pv_pop_date.loc['1998':'2009', :]
pv_before.loc[:, 'fed_tax_cred'] = 0.30 * pv_before['cost']
pv_before.loc[:, 'annual_cost'] = (pv_before['rates'] * pv_before['annual_pv_prod'])
pv_before.loc[:, 'combined_cost'] = pv_before['cost'] + pv_before['sales_tax_cost'] - pv_before['rebate'] - pv_before['fed_tax_cred']
pv_before.loc[:, 'break_even_yrs'] = pv_before['combined_cost'] / pv_before['annual_cost']
pv_before.loc[:, 'break_even_yrs'].describe()



# PLT BAR PLOT - COST - REBATE
# =============================================================================
residential_pv = pd.read_csv('residential_pv.csv', sep=',', low_memory=False)
residential_pv['rates'].mean()
residential_pv['annual_pv_prod'].mean()
residential_pv['cost'].mean()
residential_pv['rebate'].mean()

#
# =============================================================================
residential_cost = residential_pv.groupby('state')[['cost', 'rebate']].mean().sort_values(by='cost')

ave_cost = round(residential_pv['cost'].mean(), 2)

left_cost = np.arange(len(residential_cost['michigan':'new jersey']))
height_cost = residential_cost['michigan':'new jersey']['cost']

left_reb = np.arange(len(residential_cost['michigan':'new jersey']))
height_reb = residential_cost['michigan':'new jersey']['rebate']

plt.figure(figsize=(12,8))
rect1 = plt.bar(left_cost, height_cost, width=0.8, color='green', alpha=0.6)
rect2 = plt.bar(left_reb, height_reb, width=0.8, color='blue', alpha=0.6)

plt.xticks(left_cost - 0.5, residential_cost['michigan':'new jersey'].index, rotation = 75)
plt.title('Installation Average Cost and Rebate')
plt.ylabel('Cost ($)')
plt.xlabel('State')
plt.axhline(y=ave_cost, color='red')
plt.text(0, 28000, s= ('Average cost: ' + str('$') + str(ave_cost)))
plt.legend((rect1[0], rect2[0]) ,['cost', 'rebate'], loc='upper left')
plt.show()

residential_pv[residential_pv.state == 'indiana']
# =============================================================================
# Varies Size Systems - Bar Plot
# =============================================================================
small_sys = residential_pv[residential_pv['size_catg'] == 'small']
normal_sys = residential_pv[residential_pv['size_catg'] == 'normal']
large_sys = residential_pv[residential_pv['size_catg'] == 'large']

small_sys_cost = small_sys.groupby('state')['cost'].mean().sort_values()
normal_sys_cost = normal_sys.groupby('state')['cost'].mean().sort_values()
large_sys_cost = large_sys.groupby('state')['cost'].mean().sort_values()


ave_cost = round(residential_pv['cost'].mean(), 2)
htg = residential_pv.cost.values
# small cap system
sm_cost = np.arange(len(small_sys_cost['virginia':'new jersey']))
sm_height_cost = small_sys_cost['virginia':'new jersey'].values
# normal cap system
norm_cost = np.arange(len(normal_sys_cost['michigan':'new jersey']))
norm_height_cost = normal_sys_cost['michigan':'new jersey'].values
# large cap system
lrg_cost = np.arange(len(large_sys_cost['idaho':'new jersey']))
lrg_height_cost= large_sys_cost['idaho':'new jersey'].values


plt.figure(figsize=(12,8))
rect1 = plt.bar(sm_cost, sm_height_cost, width=0.4, color='red', alpha=0.7) # small
rect2 = plt.bar(norm_cost, norm_height_cost, width=0.6, color='blue', alpha=0.4) # normal
rect3 = plt.bar(lrg_cost, lrg_height_cost, width=0.8, color='gray', alpha=0.4) # large

plt.xticks(lrg_cost - 0.25, large_sys_cost.index, rotation = 75)
plt.title('System Cap Average Cost and Rebate')
plt.ylabel('Cost ($)')
plt.xlabel('State')
#plt.axhline(y=ave_cost, color='red')
#plt.text(0, 28000, s= ('Average cost: ' + str('$') + str(ave_cost)))
plt.legend(('small', 'normal', 'large'), loc='upper left')
plt.show()

#
# =============================================================================

print('Small capacity average: ' + str(round(small_sys['size_kw'].mean(), 2)))
print('Normal capacity average: ' + str(round(normal_sys['size_kw'].mean(), 2)))
print('Large capacity average: ' + str(round(large_sys['size_kw'].mean(), 2)))

print('Small capacity system cost between 2016 and 2018: ' + str('$') + str(round(small_sys_cost['2016':'2018'].mean(), 2)))
print('Normal capacity system cost between 2016 and 2018: ' + str('$') + str(round(normal_sys_cost['2016':'2018'].mean(), 2)))
print('Large capacity system cost between 2016 and 2018: ' + str('$') + str(round(large_sys_cost['2016':'2018'].mean(), 2)))


# =============================================================================
# COST
# =============================================================================
residential_cost = residential_sub.groupby('state')[['cost', 'after_rebate']].mean().sort_values(by='cost')
residential_cost.plot(kind='bar')
plt.title('Median Cost and Rebate')
plt.ylabel('Cost ($)')
plt.xlabel('State')
plt.xticks(rotation=70)
plt.tight_layout()
plt.show()

# =============================================================================
# REBATE
# =============================================================================
# median residential incentive rebate
plt.figure(figsize=(11,8))
residential_rebate = residential_sub.groupby('state')['rebate'].mean().sort_values()
residential_rebate['maryland':'florida'].plot(kind='bar')
plt.title('Average Rebate')
plt.ylabel('Rebate')
plt.xlabel('State')
plt.axhline(y=residential_sub.rebate.mean(), color='red')
plt.xticks(rotation=70)
plt.show()

# =============================================================================
# AFTER REBATE
# =============================================================================
residential_sub['after_rebate'] = residential_sub['cost'] - residential_sub['rebate']
plt.figure(figsize=(10,6))
residential_AFTER_rebate = residential_sub.groupby('state')['after_rebate'].mean().sort_values()
residential_AFTER_rebate['texas':'new jersey'].plot(kind='bar')
plt.title('After Rebate Cost')
plt.ylabel('Cost')
plt.xlabel('State')
plt.axhline(y=residential_sub.after_rebate.mean(), color='red')
plt.xticks(rotation=70)
plt.show()

# =============================================================================
# OLD
# =============================================================================
plt.figure(figsize=(10,6))
residential_sub['after_rebate'] = residential_sub['cost'] - residential_sub['rebate']
residential_cost = residential_sub.groupby('state')[['rebate', 'cost']].mean().sort_values(by='cost')
residential_cost['michigan':'new jersey']
residential_cost['michigan':'new jersey'].plot(kind='bar')
plt.title('Average Cost and Rebate')
plt.ylabel('Cost ($)')
plt.xlabel('State')
plt.xticks(rotation=70)
plt.axhline(y=residential_sub.cost.mean(), color='red')
plt.text(0, 28000, s='mean cost')
plt.show()



# =============================================================================
# YEARLY COST
# convert to datetime
# =============================================================================

date_time = '%m/%d/%Y'
datetime = pd.to_datetime(residential_sub['date_installed'], format=date_time)
pv_pop_date = residential_sub.set_index(datetime)
pv_pop_date.info()


# =============================================================================
# CUT INTERVAL FOR SIZE
# =============================================================================
size = pv_pop_date['size_kw']
size.describe()

pv_pop_date['size_catg'] = pd.qcut(size.values, 3, labels=['small', 'normal', 'large'])
pv_pop_date.head()


# =============================================================================
# Cost Time-Series
# =============================================================================
# resample by 6 months and compute mean
yearly_mean_cost = pv_pop_date['cost'].resample('6M').agg('mean').dropna(axis=0, how='all')

dates_cost = yearly_mean_cost.index[::2]
labels_cost = dates_cost.strftime('%b-%Y')

plt.figure(figsize=(10,6))
plt.plot(yearly_mean_cost)
plt.xticks(rotation=60)
plt.xlabel('Year')
plt.ylabel('Cost($)')
plt.show()

percent_change = (yearly_mean_cost['2008'].mean() - yearly_mean_cost['2017'].mean()) / yearly_mean_cost['2008'].mean()
print('Average price change of', round(percent_change*100, 2), 'percent since 2008 peak.')



# =============================================================================
# Small Systems
# =============================================================================

def cost_time_series(df, col, X):
  name = str(X) + str('_sys')
  name = df[df[col] == X]
  name_cost = name['cost'].resample('6M').agg('mean').dropna(axis=0, how='all')

#  # dates and labels
#  dates_ = name_cost.index[::2]
#  labels_ = dates_.strftime('%b-%Y')

  # plot
  plt.figure(figsize=(10,6))
  plt.plot(name_cost)
  plt.xticks(rotation=60)
  plt.xlabel('Year')
  plt.ylabel('Cost($)')
  title = (str(X) + ' system installation cost').upper()
  plt.title(title)
  plt.show()

cost_time_series(pv_pop_date, 'size_catg', 'small')
cost_time_series(pv_pop_date, 'size_catg', 'normal')
cost_time_series(pv_pop_date, 'size_catg', 'large')


# =============================================================================
# Summary Stats
# Small, Normal, Large Systems
# =============================================================================

small_sys = pv_pop_date[pv_pop_date['size_catg'] == 'small']
normal_sys = pv_pop_date[pv_pop_date['size_catg'] == 'normal']
large_sys = pv_pop_date[pv_pop_date['size_catg'] == 'large']

# subplot boxplot
# =============================================================================
plt.figure(figsize=(10,6))
plt.subplot(1,1,1)
sns.boxplot(small_sys['size_kw'], palette='Set3')
plt.show()

plt.figure(figsize=(10,6))
plt.subplot(2,1,2)
sns.boxplot(normal_sys['size_kw'], palette='Set3')
plt.show()

plt.figure(figsize=(10,6))
plt.subplot(2,1,2)
sns.boxplot(large_sys['size_kw'], palette='Set3')
plt.show()






# =============================================================================
# Cost Time-Series
# Small, Normal, Large Systems
# =============================================================================
# extract cost and resample
small_sys_cost = small_sys['cost'].resample('6M').agg('mean').dropna(axis=0, how='all')
normal_sys_cost = normal_sys['cost'].resample('6M').agg('mean').dropna(axis=0, how='all')
large_sys_cost = large_sys['cost'].resample('6M').agg('mean').dropna(axis=0, how='all')

# plot size
plt.figure(figsize=(10,6))
# plot TS
plt.plot(small_sys_cost)
plt.plot(normal_sys_cost)
plt.plot(large_sys_cost)
# label
plt.xlabel('Year')
plt.ylabel('Cost($)')
plt.title('Installation Cost')
plt.legend(('small', 'normal', 'large'), loc='upper right')
plt.show()


# =============================================================================
#
# =============================================================================
# resample by 6 months and compute mean
sm_cost_per_watt = small_sys['cost_per_watt'].resample('6M').agg('mean').dropna(axis=0, how='all')
nr_cost_per_watt = normal_sys['cost_per_watt'].resample('6M').agg('mean').dropna(axis=0, how='all')
lg_cost_per_watt = large_sys['cost_per_watt'].resample('6M').agg('mean').dropna(axis=0, how='all')


plt.figure(figsize=(10,6))
# plot TS
plt.plot(sm_cost_per_watt)
plt.plot(nr_cost_per_watt)
plt.plot(lg_cost_per_watt)
# label
plt.xlabel('Year')
plt.ylabel('Cost per Watt ($)')
plt.title('Cost per Watt')
plt.legend(('small', 'normal', 'large'), loc='upper right')
plt.margins(0.02)
plt.show()

# =============================================================================
# Power Generation
# Small, Normal, Large Systems
# =============================================================================

sm_mean_kwh_prod = small_sys['annual_pv_prod'].resample('6M').agg('mean').dropna(axis=0, how='all')
nr_mean_kwh_prod = normal_sys['annual_pv_prod'].resample('6M').agg('mean').dropna(axis=0, how='all')
lg_mean_kwh_prod = large_sys['annual_pv_prod'].resample('6M').agg('mean').dropna(axis=0, how='all')


plt.figure(figsize=(10,6))
plt.plot(sm_mean_kwh_prod['1998':])
plt.plot(nr_mean_kwh_prod['1998':])
plt.plot(lg_mean_kwh_prod['1998':])

plt.xticks(rotation=60)
plt.xlabel('Year')
plt.ylabel('Power (kWh)')
plt.title('Power Generation')
plt.ylim([0, 13000])
plt.legend(('small', 'normal', 'large'), loc='lower center')
plt.margins(0.02)
plt.show()


# STATES FUNCTION
# =============================================================================
# For each state input generate:
# TS:
#  Cost for small, normal, large systems
# Bar Graphs:
#   Average Cost and Rebate
# =============================================================================
new_york = pv_pop_date[pv_pop_date['state'] == 'new york']
new_york.head()

cost_plt = new_york['cost'].resample('6M').agg('mean').dropna(axis=0, how='all')
rebate_plt = new_york['rebate'].resample('6M').agg('mean').dropna(axis=0, how='all')
cost_watt_plt = new_york['cost_per_watt'].resample('6M').agg('mean').dropna(axis=0, how='all')
size_plt = new_york['size_kw'].resample('6M').agg('mean').dropna(axis=0, how='all')
power1_plt = new_york['annual_pv_prod'].resample('6M').agg('mean').dropna(axis=0, how='all')
power2_plt = new_york['reported_annual_energy_prod'].resample('6M').agg('mean').dropna(axis=0, how='all')


# national average installation cost
mean_cost = round(pv_pop_date['cost'].mean(), 2)

plt.subplot(2,1,1)
plt.figure(figsize=(10,6))
plt.plot(cost_plt)
plt.plot(rebate_plt)
plt.xticks(rotation=60)
plt.margins(0.02)
plt.xlabel('Year')
plt.title('New York - Average Cost and Rebate')
plt.ylabel('Average Cost ($)')
plt.legend(('Cost', 'Rebate amount'), loc='upper left')
#plt.axhline(y=mean_cost, color='red')
#plt.text(x=0, y=30000, s=('Average National Cost: ' + str('$') + str(mean_cost)))
plt.show()

plt.subplot(2,1,2)
plt.figure(figsize=(10,6))
plt.plot(cost_watt_plt)
plt.plot(size_plt)
plt.xticks(rotation=60)
plt.margins(0.02)
plt.xlabel('Year')
plt.title('New York - Average Cost per Watt')
plt.ylabel('Average Cost ($)')
plt.show()

plt.subplot(2,2,1)
plt.figure(figsize=(10,6))
plt.plot(power1_plt, color='black') # estimate
plt.plot(power2_plt, color='green') # actual
plt.xticks(rotation=60)
plt.margins(0.02)
plt.xlabel('Year')
plt.ylabel('Power (kWh)')
plt.legend(('Estimated', 'Actual'), loc='upper left')
plt.title('Annual Energy Production (Estimate vs Actual)')
plt.show()

# scatter plot
# Cost vs Size
mean_cost = round(residential_sub['cost'].mean(), 2)
mean_size = round(new_york['size_kw'].mean(), 2)

plt.figure(figsize=(10,6))
plt.scatter(x=new_york['size_kw'], y=new_york['cost'], marker='.', alpha=0.5)
plt.xlabel('Size (kW)')
plt.ylabel('Cost ($)')
plt.title('Capacity and Price')

plt.axvline(x=residential_sub['size_kw'].mean(), color='red')
plt.axhline(y=residential_sub['cost'].mean(), color='red')
plt.margins(0.02)

plt.text(x=9, y=25000, s=('National Average Cost: ' + str('$') + str(mean_cost)))
plt.text(x=5.1, y=80000, s=('Average State Capacity: ' +  str(mean_size) + str('kW')))

plt.show()





# =============================================================================
# Cost per Watt
# =============================================================================
# resample by 6 months and compute mean
yearly_cost_per_watt = pv_pop_date['cost_per_watt'].resample('6M').agg('mean').dropna(axis=0, how='all')
dates_cost_watt = yearly_cost_per_watt.index[::4]
labels_cost_watt = dates_cost_watt.strftime('%b-%Y')

plt.figure(figsize=(10,6))
plt.plot(yearly_cost_per_watt)
plt.xticks(rotation=60)
plt.xlabel('Year')
plt.ylabel('Cost($)')
plt.title('Cost per Watt')
plt.show()

percent_change = (yearly_cost_per_watt['2008'].mean() - yearly_cost_per_watt['2017'].mean()) / yearly_cost_per_watt['2008'].mean()
print('Average cost per watt change of', round(percent_change*100, 2), 'percent.')


# =============================================================================
# 2009 to 2017
# =============================================================================
plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(yearly_mean_cost['2007':'2017'])
plt.xlabel('Year')
plt.ylabel('Average Cost($)')
plt.title('2008 to 2017')
plt.show()


# =============================================================================
# 1998 to 2009
# =============================================================================
plt.figure(figsize=(10,6))
plt.subplot(2,1,2)
plt.plot(yearly_mean_cost['1998':'2008'])
plt.xlabel('Year')
plt.ylabel('Average Cost($)')
plt.title('1998 to 2009')
plt.show()


# =============================================================================
# Mean cost after 2009
# =============================================================================
before = pv_pop_date.loc['2009':'2017', 'cost'].mean()
after = pv_pop_date.loc['1998':'2009', 'cost'].mean()



# =============================================================================
# Count Incentives and Rebates
# =============================================================================
yearly_incent_count = pv_pop_date['incentive_prog_names'].resample('6M').agg('count').dropna(axis=0, how='all')
dates_incent = yearly_incent_count['1998':'Jan-2016'].index[::2]
labels_ncent = dates_incent.strftime('%b-%Y')

plt.figure(figsize=(10,6))
plt.plot(yearly_incent_count['1998':'Jan-2016'])
plt.xticks(dates_incent, labels_ncent, rotation=60)
plt.xlabel('Year')
plt.ylabel('Count')
plt.title('Number of Incentives')
plt.show()


# Rebates
# =============================================================================
yearly_rebate = pv_pop_date['rebate'].resample('6M').agg('mean').dropna(axis=0, how='all')
dates_re = yearly_rebate['1998':].index[::2]
labels_re = dates_re.strftime('%b-%Y')

plt.figure(figsize=(10,6))
plt.plot(yearly_rebate['1998':])
plt.plot(yearly_mean_cost[:dates_re[-1]])
plt.xticks(dates_re, labels_re, rotation=60)
plt.xlabel('Year')
plt.ylabel('Rebate Amount ($)')
plt.legend(('rebate amount', 'cost'), loc='upper left')
plt.show()


# =============================================================================
# Production

# Estimated Production
# =============================================================================
yearly_mean_kwh_prod = pv_pop_date['annual_pv_prod'].resample('6M').agg('mean').dropna(axis=0, how='all')
dates_ = yearly_mean_kwh_prod['1998':].index[::2]
labels_ = dates_.strftime('%b-%Y')

plt.figure(figsize=(10,6))
plt.plot(yearly_mean_kwh_prod['1998':])
plt.xticks(rotation=60)
plt.xlabel('Year')
plt.ylabel('kWh')
plt.show()

# Actual/Reported Production
# =============================================================================
yearly_mean_kwh_act = pv_pop_date['reported_annual_energy_prod'].resample('6M').agg('mean').dropna(axis=0, how='all')
dates = yearly_mean_kwh_act['1998':].index[::2]
labels = dates.strftime('%b-%Y')

plt.figure(figsize=(10,6))
plt.plot(yearly_mean_kwh_prod['1998':], color='black') # estimated/theoretical
plt.plot(yearly_mean_kwh_act['1998':], color='green') # actual energy output
plt.xticks(dates, labels, rotation=60)
plt.xlabel('Year')
plt.ylabel('kWh')
plt.legend(('estimated', 'reported'), loc='upper left')
plt.show()


#
# =============================================================================
yearly_mean_cap = pv_pop_date['size_kw'].resample('6M').agg('mean').dropna(axis=0, how='all')
dates_size = yearly_mean_cost.index[::2] # select every 2nd date
labels_size = dates.strftime('%b-%Y')

plt.figure(figsize=(10,6))
plt.plot(yearly_mean_cap['1998':])
#plt.plot(yearly_mean_cost['1998':])
plt.xticks(dates_size, labels_size, rotation=60)
plt.xlabel('Year')
plt.ylabel('Capacity (kW)')
plt.show()






# =============================================================================
# Number of Installations
# =============================================================================
# count installations per year
install_count = pv_pop_date['cost'].resample('6M').agg('count').dropna(axis=0, how='all')
dates_n= install_count['1998':'2016'].index[::2] # select every 2nd date
labels_n = dates.strftime('%b-%Y')

plt.figure(figsize=(10,6))
plt.plot(install_count['1998':'Jan 2016'])
plt.xticks(dates_n, labels_n, rotation=60)
plt.xlabel('Year')
plt.ylabel('Count')
plt.title('Number of Installations')
plt.show()


pv_pop_date.loc['2016':'2017', :].info()
# =============================================================================
#




# =============================================================================
# Group Cities
# =============================================================================




install_type_size = pv_pop.groupby('install_type')['size_kw'].median().sort_values()
install_type_size[:'government'].plot(kind='bar')
type_group['residential':'government'].plot(kind='bar')
plt.title('Median Size')
plt.ylabel('Size')
plt.xlabel('Type of Installation')
plt.xticks(rotation=60)
plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

N = 5
men_means = (20, 35, 30, 35, 27)
men_std = (2, 3, 4, 1, 2)

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, men_means, width, color='r')
women_means = (25, 32, 34, 20, 25)
women_std = (3, 5, 2, 3, 3)
rects2 = ax.bar(ind + width, women_means, width, color='y')


left = np.arange(len(install_type_size[:'government']))
height = type_group['residential':'government'].values
width = 0.5

plt.bar(left, height, width)
plt.xticks(range(len(type_group)), type_group['residential':'government'].keys(), rotation = 45)
plt.title('Median Size')
plt.ylabel('Size')
plt.xlabel('Type of Installation')
plt.tight_layout()
plt.show()











