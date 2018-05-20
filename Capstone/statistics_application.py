# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 15:22:18 2018

@author: mouz
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
sns.set()

# read
residential_pv = pd.read_csv('residential_pv.csv', sep=',', low_memory=False)

residential_pv.info()


# Functions
# =============================================================================
np.random.seed(10)

def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)
    # x-data for the ECDF: x
    x = np.sort(data)
    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n
    return x, y

def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    # Compute correlation matrix: corr_mat
    corr_mat = np.corrcoef(x,y)

    # Return entry [0,1]
    return corr_mat[0,1]

def bootstrap_replicate_1d(data, func):
  '''Generate boostrap replicate of one-dim data '''
  bs_sample = np.random.choice(data, len(data))
  return func(bs_sample)

# generate many bootstrap replicates from the data set
def draw_bs_reps(data, func, size=1):
    """Draw bootstrap replicates."""
    # Initialize array of replicates: bs_replicates
    bs_replicates = np.empty(size)
    # Generate replicates
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data, func)
    return bs_replicates

def diff_of_means(data_1, data_2):
    """Difference in means of two arrays."""
    # The difference of means of data_1, data_2: diff
    diff = np.mean(data_1) - np.mean(data_2)
    return diff

def permutation_sample(data1, data2):
    """Generate a permutation sample from two data sets."""
    # Concatenate the data sets: data
    data = np.concatenate((data1, data2))
    # Permute the concatenated array: permuted_data
    permuted_data = np.random.permutation(data)
    # Split the permuted array into two: perm_sample_1, perm_sample_2
    perm_sample_1 = permuted_data[:len(data1)]
    perm_sample_2 = permuted_data[len(data1):]
    return perm_sample_1, perm_sample_2

def draw_perm_reps(data_1, data_2, func, size=1):
    """Generate multiple permutation replicates."""
    # Initialize array of replicates: perm_replicates
    perm_replicates = np.empty(size)
    for i in range(size):
        # Generate permutation sample
        perm_sample_1, perm_sample_2 = permutation_sample(data_1, data_2)
        # Compute the test statistic
        perm_replicates[i] = func(perm_sample_1, perm_sample_2)
    return perm_replicates



# =============================================================================
# Correlation of Cost and Size
# =============================================================================
# explore the relationship, did this correlation occur PURELY BY CHANCE?

# scatter plot
plt.scatter(x=residential_pv['size_kw'].values, y=residential_pv['cost'].values, marker='.', alpha=0.5)
plt.xlabel('Size (kW)')
plt.ylabel('Cost ($)')
plt.title('Cost vs Size')
plt.show()

# remove all missing values before computing pearson r
clean_pv = residential_pv
clean_pv[clean_pv['cost'].isnull().values] = np.nan
clean_pv = residential_pv.dropna(axis=0, how='all').reset_index(drop=True)
pearson_r(clean_pv['size_kw'].values, clean_pv['cost'])
pearson_r(clean_pv['annual_pv_prod'].values, clean_pv['cost'])



# =============================================================================
# Tracking Type
# =============================================================================
clean_pv.groupby('tracking_type')['annual_pv_prod'].mean().sort_values()

state = clean_pv[clean_pv['state'] == 'arizona']
print(len(state))
print(pearson_r(state['annual_pv_prod'], state['cost']))

# Plot
plt.bar(clean_pv['tracking_type'], clean_pv['annual_pv_prod'])
#plt.scatter(state['size_kw'], state['rebate'], marker='.',  alpha=0.5)
plt.margins(0.02)
plt.xlabel('azimuth1')
plt.ylabel('Cost ($)')
plt.show()
print(pearson_r(clean_pv['tilt1'], clean_pv['cost']))




# =============================================================================
# Local Correlations
#
# =============================================================================
# expand function to show amount of generated energy, average state rebate

def state_cost(state_name, sys_cap):
  state = clean_pv[clean_pv['state'] == state_name]
  print('Amount of data:', len(state))
  print('Correlation between system capacity and cost:', pearson_r(state['size_kw'], state['cost']))

  a, b = np.polyfit(state['size_kw'], state['cost'], deg=1)
  # Print the results to the screen
  print('slope =', a, 'change in price per kilowatt')
  print('intercept =', b, 'dollars')
  # Make theoretical line to plot
  x = np.array([0,state['size_kw'].max()])
  y = a * x + b
  # Plot

  plt.scatter(state['size_kw'], state['cost'], marker='.',  alpha=0.5)
  plt.plot(x, y, color='black')
  plt.margins(0.02)
  plt.xlabel('Cap')
  plt.ylabel('Cost ($)')
  plt.show()
  # when x = 10 kW
  x = sys_cap
  y = a * x + b
  print('Gross cost of a ' + str(x) + ' kilowatt system is ' + str('$') + str(round(y, 2)))
state_cost('new jersey', 6)


# =============================================================================
# Cost and Annual Production
# =============================================================================


a1, b1 = np.polyfit(clean_pv['annual_pv_prod'], clean_pv['cost'], deg=1)
# Print the results to the screen
print('slope =', a1, 'change in price per kilowatt-hour')
print('intercept =', b1, 'dollars')
print('Correlation between power generated and cost:', pearson_r(clean_pv['annual_pv_prod'], clean_pv['cost']))


# Make theoretical line to plot
x1 = np.array([0, clean_pv['annual_pv_prod'].max()])
y1 = a1 * x1 + b1

plt.scatter(clean_pv['annual_pv_prod'], clean_pv['cost'], marker='.',  alpha=0.7)
plt.plot(x1,y1, color='black')
plt.margins(0.02)
plt.xlabel('Power (kWh)')
plt.ylabel('Cost ($)')
plt.show()



# =============================================================================
# Linear Regression
# =============================================================================

size = clean_pv['size_kw'].values
cost = clean_pv['cost'].values

# Plot
plt.plot(size, cost, marker='.', linestyle='none', alpha=0.5)
plt.margins(0.02)
plt.xlabel('Capacity (kW)')
plt.ylabel('Cost ($)')
plt.show()

# calculate slope and intercept
# Perform a linear regression using np.polyfit(): a, b
a, b = np.polyfit(size, cost, deg=1)

# Print the results to the screen
print('slope =', a, 'change in price per kilowatt')
print('intercept =', b, 'dollars')

# Make theoretical line to plot
x = np.array([0,18])
y = a * x + b

# Add regression line to plot
plt.plot(x, y, color='black')
plt.plot(size, cost, marker='.', linestyle='none', alpha=0.05)
plt.margins(0.02)
plt.xlabel('Capacity (kW)')
plt.ylabel('Cost ($)')
plt.title('Cost vs Capacity Regression')
plt.margins(0.5)
plt.xlim([0,20])
plt.ylim([0,100000])
plt.show()


# when x = 10 kW
x = 10
y = a * x + b
print('For a ' + str(x) + ' kilowatt system is ' + str('$') + str(y))


# =============================================================================
# Normality Test
# =============================================================================
# test for normality of size and cost

# Cost - ECDF plot
x,y = ecdf(residential_pv['cost'])
plt.plot(x,y, linestyle='none', marker='.', alpha=0.3)
plt.margins(0.02)
plt.show()

# Size - ECDF plot
x1,y1 = ecdf(residential_pv['size_kw'])
plt.plot(x1,y1, linestyle='none', marker='.', alpha=0.3)
plt.margins(0.02)
plt.show()

# Cost per Watt - ECDF plot
x2,y2 = ecdf(residential_pv['cost_per_watt'])
plt.plot(x2,y2, linestyle='none', marker='.', alpha=0.3)
plt.margins(0.02)
plt.show()


# Boxplots
# =============================================================================
plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
sns.boxplot(x=residential_pv['cost'])
plt.show()

plt.figure(figsize=(10,6))
plt.subplot(2,1,2)
sns.boxplot(x=residential_pv['size_kw'])
plt.show()


# Distributions
# =============================================================================

plt.figure(figsize=(10,6))
plt.subplot(211)
plt.hist(residential_pv['size_kw'].values, bins=50)
plt.xlabel('System Capacity (kW)')
plt.ylabel('Count')
plt.margins(0.02)
plt.axvline(x=residential_pv['size_kw'].mean(), color='red')
plt.axvline(x=residential_pv['size_kw'].median(), color='black')
plt.title('System Size Distribution')
plt.legend(('mean', 'median'), loc='upper left')
plt.show()

plt.figure(figsize=(10,6))
plt.subplot(212)
plt.hist(residential_pv['cost'].dropna(), bins=50)
plt.xlabel('Cost ($)')
plt.ylabel('Count')
plt.margins(0.02)
plt.axvline(x=residential_pv['cost'].mean(), color='red')
plt.axvline(x=residential_pv['cost'].median(), color='black')
plt.title('Cost Distribution')
plt.legend(('mean', 'median'), loc='upper right')
plt.show()


# quantile plot
# =============================================================================
import scipy.stats as stats
import pylab
stats.probplot(residential_pv['cost'], dist="norm", plot=pylab)
plt.show()

stats.probplot(['size_kw'], dist="norm", plot=pylab)
plt.show()




# =============================================================================
# Cost - Confidence Interval
# =============================================================================
# Take 10,000 bootstrap replicates of the mean: bs_replicates
bs_reps = draw_bs_reps(cost, np.mean, size=10000)
np.mean(cost)
np.mean(bs_reps)
np.std(bs_reps)

# SEM
sem = np.std(cost) / np.sqrt(len(cost))
print('Standard error of mean:', sem)

# Compute and print standard deviation of bootstrap replicates
bs_std = np.std(bs_reps)
print(bs_std)

# Make a histogram of the results
n_bins = int(np.cbrt(len(bs_reps)))
plt.hist(bs_reps, bins=30, normed=True, alpha=0.5, color='green')
plt.xlabel('Mean Cost ($)')
plt.ylabel('PDF')
plt.show()

# confidence interval
conf_int = np.percentile(bs_reps, [0.5, 99.5])
print('Confidence interval:', conf_int)
print(np.mean(bs_reps))
print(np.std(bs_reps))
# With 99% confidence we can say that the mean cost lies within the interval of $27,105.29 and $27,232.25


# =============================================================================
# FUNCTION:
# generate bs replicates for a given variable and plot it
# print confidence interval, mean, std
# =============================================================================
size = clean_pv['size_kw'].values
cost = clean_pv['cost'].values
cost_per_watt = clean_pv['cost_per_watt'].values

def bs_reps_plot(data, func, c_int, size=100, label='data'):
  data_bs_reps = draw_bs_reps(data, func, size)
  n_bins = int(np.cbrt(len(data)))
  plt.hist(data_bs_reps, bins=n_bins, normed=True, alpha=0.5, color='green')
  plt.xlabel(label)
  plt.ylabel('PDF')
  plt.show()
  conf_int = np.percentile(data_bs_reps, c_int)
  print('Confidence interval:', conf_int)
  print('Mean of bootstrap reps:', np.mean(data_bs_reps))
  print('Standard Deviation of bootstrap reps:', np.std(data_bs_reps))

bs_reps_plot(size, np.mean, [2.5, 97.5], 1000, label='capacity (kW)')
bs_reps_plot(cost_per_watt, np.mean, [2.5, 97.5], 1000, label='cost per watt ($ / watt)')
bs_reps_plot(cost, np.mean, [2.5, 97.5], 1000, label='Cost ($)')



# =============================================================================
# Hypothesis test on Pearson correlation
# =============================================================================


# HYPOTHESIS:
# H0: The observed correlation between system capacity and installation cost may have occurred by chance
# The two variables may be completely indepedent of each other

# APPROACH
# simulate hypothesis that they are completely independent of each other
# permute size/capacity; leave cost fixed
# for each permutation(shuffling data), compute Pearson correlation
# asses how many replicates have Pearson corr greater than observed

# Compute observed correlation: r_obs
r_obs = pearson_r(size, cost)

# Initialize permutation replicates: perm_replicates
permutation_reps = np.empty(10000)

# Draw replicates
for i in range(10000):
    # Permute illiteracy measurments: illiteracy_permuted
    size_permuted = np.random.permutation(size)
    # Compute Pearson correlation
    permutation_reps[i] = pearson_r(size_permuted, cost)

# Compute p-value: p
p = np.sum(permutation_reps > r_obs)/len(permutation_reps)
print('p-val =', p)
# out of 10000 samples, there were ZERO replicates with correlation being greater than 0.80
# if correlation had occured PURELY BY CHANCE, we would expect very few number of values of positive correlations

# compute number of values which contains positive correlation
# if correlation had occured PURELY BY CHANCE, this value would be small
np.sum(permutation_reps > r_obs)















# =============================================================================
# Bootstrap Pairs
# =============================================================================

# x, y - slope and intercept
bs_slope_reps, bs_intercept_reps = draw_bs_pairs_linreg(size, cost, size=1000)

# Compute and print 95% CI for slope
print(np.percentile(bs_slope_reps, [2.5, 97.5]))

# Plot the histogram
plt.hist(bs_slope_reps, bins=50, normed=True)
plt.xlabel('slope')
plt.ylabel('PDF')
plt.show()

# visualize the variability in linear regression
# Generate array of x-values for bootstrap lines: x
x = np.array([0,100])

# Plot the bootstrap lines
for i in range(100):
     plt.plot(x, bs_slope_reps[i]*x + bs_intercept_reps[i],
                 linewidth=1, color='black')

# Plot the data
plt.plot(size, cost, marker='.', linestyle='none', alpha=0.6)

# Label axes, set the margins, and show the plot
plt.xlabel('size (kW)')
plt.ylabel('Cost ($)')
plt.xlim([0,size.max()+size.std()])
plt.ylim([0, cost.max()+cost.std()])
plt.margins(0.06)
plt.show()




# =============================================================================
# Bootstrap Resample
# =============================================================================
size = clean_pv['size_kw'].values
cost = clean_pv['cost'].values
cost_per_watt = clean_pv['cost_per_watt'].values


# generate sample data and plot ECDF
for _ in range(50):
    # Generate bootstrap sample: bs_sample
    bs_sample = np.random.choice(cost, size=len(cost))
    # Compute and plot ECDF from bootstrap sample
    x, y = ecdf(bs_sample)
    _ = plt.plot(x, y, marker='.', linestyle='none',
                 color='gray', alpha=0.1)

x, y = ecdf(cost)
plt.plot(x, y, marker='.')
plt.margins(0.02)
plt.xlabel('Cost ($)')
plt.ylabel('ECDF')
plt.show()

























# =============================================================================
# Normality test
# =============================================================================
# normality test
import pylab
# quantile plot
stats.probplot(small_ex, dist="norm", plot=pylab)
plt.show()

















# =============================================================================
# Difference in Means, Before and After 2008/2009
# =============================================================================
# =============================================================================
# YEARLY COST
# =============================================================================

date_time = '%m/%d/%Y'
datetime = pd.to_datetime(clean_pv['date_installed'], format=date_time)
pv_pop_date = clean_pv.set_index(datetime)
pv_pop_date.info()


# =============================================================================
# Mean cost before and after 2008-2009
# =============================================================================
pv_pop_date.loc['2009':'2017', 'cost'].mean()
pv_pop_date.loc['1998':'2009', 'cost'].mean()


pv_pop_date.loc['2015':'2017', 'cost_per_watt'].mean()
pv_pop_date.loc['1998':'2009', 'cost_per_watt'].mean()

def state_cost_cap(df, state_name, sys_cap):
  state = df[df['state'] == state_name]
  if len(state) < 1:
    print('No data available for state.')
  else:
    print('Amount of data:', len(state))
    print('Correlation between system capacity and cost:', pearson_r(state['size_kw'], state['cost']))
    a, b = np.polyfit(state['size_kw'], state['cost'], deg=1)
    x = np.array([0,state['size_kw'].max()])
    y = a * x + b
    plt.scatter(state['size_kw'], state['cost'], marker='.',  alpha=0.5)
    plt.plot(x, y, color='black')
    plt.margins(0.02)
    plt.xlabel('Cap')
    plt.ylabel('Cost ($)')
    plt.show()
    x = sys_cap
    y = a * x + b
    print('Gross cost of a ' + str(x) + ' kilowatt system is ' + str('$') + str(round(y, 2)))
    print('slope =', a, 'change in price per kilowatt')
    print('intercept =', b, 'dollars')


state_cost_cap(pv_pop_date['2009':'2017'], 'new york', 6)
state_cost_cap(pv_pop_date['1998':'2009'], 'california', 6)













# =============================================================================
# Cost over the years
# =============================================================================
# explore correlation over the years, how does the mean cost changed over the years?

















