import numpy as np
import pandas as pd

import scripts.helper
from scripts.fama_macbeth import stage_one_fama_macbeth, stage_two_fama_macbeth
from scripts.plotting import plot_second_stage_result_from_values
from scripts.pruning_animated import generate_pruning_animation
from scripts.ransac_attempt import apply_ransac

# read csv-data and collect them in dataframes
df_consumption = pd.read_csv('data/Nondurables_Services_Consumption.csv')  # consumption data
df_population = pd.read_csv('data/Population.csv')                         # population data
df_price_index = pd.read_csv('data/price_index.csv')                       # price index
df_test_assets = pd.read_csv('data/test_assets.csv')

#############################################################################
# Task 2: Calculate yearly (filtered) consumption per capita price adjusted #
#############################################################################

# check if years match
assert (df_consumption['year'].values == df_population['year'].values).all()
# calculate price adjusted non-durables and services
df_real_consumption = df_consumption.copy()
df_real_consumption['nondurables'] = df_consumption['nondurables'] / df_price_index['prc_index_nondurables']
df_real_consumption['services'] = df_consumption['services'] / df_price_index['prc_index_services']

# calculated real total consumption of each year
df_real_consumption['total_real_consumption'] = df_real_consumption['nondurables'] + df_real_consumption['services']

# merge total consumption and population of a given year
df_solution = df_real_consumption[['year', 'total_real_consumption']].merge(df_population, on='year')
# calculate per capita real consumption
# adjust population by multiplying with 1000 and adjust total_consumption by 1Mio
# This step is optional since units cancel out upon division
df_solution['total_real_consumption_per_capita'] = (
        df_solution['total_real_consumption'] * (10 ** 6) / (df_solution['pop'] * (10 ** 3))
)
# calculate the growth rate (using log differences of consecutive years)
# we shift the data to ensure that we divide by the previous year
# this results in the growth rate of the first year to be NaN
df_solution['filtered_growth_rate'] = np.log(
    (df_solution['total_real_consumption_per_capita'] /
     df_solution['total_real_consumption_per_capita'].shift(1))
)
# the average growth rate, ignoring the NaN entries
average_growth_rate = df_solution['filtered_growth_rate'].mean(skipna=True)

###############################################################################
# Task 3: Calculate yearly (unfiltered) consumption per capita price adjusted #
###############################################################################

# Omega is the filter parameter as specified in the course material/paper
omega = 0.46
# Function to compute the unfiltered log-level of consumption for each row
def unfiltering(df):
    """
    Reconstructs the unfiltered log consumption level (ŷₜ) from filtered growth data,
    using the formula:  ŷₜ = [ĉₜ − (1 − Ω) * Δĉₜ₋₁] / Ω
    Args:
        df (pd.Series): A row of the DataFrame, including 'total_consumption_per_capita'
                        and a lagged value of the filtered growth rate.
    Returns:
        float: Unfiltered log consumption level at time t.
    """
    return (np.log(df['total_real_consumption_per_capita']) - (1 - omega) * df['growth_rate_shifted']) / omega


# Shift the filtered growth rate by one period to obtain the c_{t-1}
df_solution['growth_rate_shifted'] = df_solution['filtered_growth_rate'].shift(1)
# Apply the unfiltering formula row-wise to compute the unfiltered log consumption level
df_solution['unfiltered_log_consumption_level'] = df_solution.apply(unfiltering, axis=1)
# Drop the temporary shifted column as it's no longer needed
df_solution.drop('growth_rate_shifted', axis=1, inplace=True)
# Compute the growth rate of the unfiltered log consumption level
df_solution['unfiltered_growth_rate'] = (
        df_solution['unfiltered_log_consumption_level'] -
        df_solution['unfiltered_log_consumption_level'].shift(1)
)
# Optional: Inspect the filtered vs. reconstructed unfiltered growth rates
print()
print(df_solution[['year', 'filtered_growth_rate', 'unfiltered_growth_rate']].head(8))
print()
df_solution.to_csv('results/solution_consumption.csv', index=False)

####################################################################################################
# Task 5: Calculate and compare fama and macbeth by using filtered and unfiltered consumption data #
####################################################################################################

# first stage fama and mcbeth with filtered NIPA data
fil_alpha_values, fil_beta_values = stage_one_fama_macbeth(
    df_solution['filtered_growth_rate'],
    df_test_assets,
    3,
    1)

# second stage fama and mcbeth with filtered NIPA data
fil_exposure, fil_lambda0 = stage_two_fama_macbeth(df_test_assets,fil_beta_values)


# first stage fama and mcbeth with unfiltered NIPA data
unfil_alpha_values, unfil_beta_values = stage_one_fama_macbeth(
    df_solution['unfiltered_growth_rate'],
    df_test_assets,
    5,
    3)

# second stage fama and mcbeth with unfiltered NIPA data

unfil_exposure, unfil_lambda0 = stage_two_fama_macbeth(df_test_assets, unfil_beta_values)

###################################################################################
######################### Optional: visualize the results:#########################
###################################################################################
print("\n### RESULTS ###")
print("Filtered Consumption Data:")
print("Price of Risk (slope/lambda):")
print(fil_exposure)
print("Return Unexplained by Factor Model (Intercept):")
print(fil_lambda0)
print("\nUnfiltered Consumption Data:")
print("Price of Risk (slope/lambda):")
print(unfil_exposure)
print("Return Unexplained by Factor Model (Intercept):")
print(unfil_lambda0)
print()
# filtered
plot_second_stage_result_from_values(
    fil_beta_values,
    df_test_assets,
    fil_exposure,
    fil_lambda0,
    "filtered"
)
# unfiltered
plot_second_stage_result_from_values(
    unfil_beta_values,
    df_test_assets,
    unfil_exposure,
    unfil_lambda0,
    "unfiltered"
)
print()
###################################################################################
################################### Optional 2: ###################################
###################################################################################

# it seems outlier skew the line (high lambda_0)
# idea, drop n largest and smallest value

# find n smallest and n_largest
def prune_n_from_betas_and_assets(betas:pd.Series, assets:pd.DataFrame, n):
    sorted_beta = unfil_beta_values.copy(deep=True).to_numpy()
    sorted_beta.sort()
    n_smallest = sorted_beta[:n]
    n_largest = sorted_beta[-n:]
    # get assets names corresponding to of n_smallest and n_largest
    smallest_assets = scripts.helper.get_elements_of_series(unfil_beta_values, n_smallest)
    largest_assets = scripts.helper.get_elements_of_series(unfil_beta_values, n_largest)

    pruned_beta_values = unfil_beta_values.copy(deep=True)
    scripts.helper.delete_elements_from_series(pruned_beta_values, smallest_assets, inplace=True)
    scripts.helper.delete_elements_from_series(pruned_beta_values, largest_assets, inplace=True)

    pruned_test_assets = df_test_assets.copy(deep=True)
    scripts.helper.delete_columns_from_dataframe(pruned_test_assets, smallest_assets, inplace=True)
    scripts.helper.delete_columns_from_dataframe(pruned_test_assets, largest_assets, inplace=True)
    return pruned_beta_values, pruned_test_assets

pruned_betas_list = []
pruned_assets_list = []
for i in range(1,4):
    b, a = prune_n_from_betas_and_assets(unfil_beta_values, df_test_assets, i)
    pruned_betas_list.append(b)
    pruned_assets_list.append(a)

pruned_exposure, pruned_lambda0 = stage_two_fama_macbeth(pruned_assets_list[2], pruned_betas_list[2])
generate_pruning_animation(pruned_betas_list, pruned_assets_list)

plot_second_stage_result_from_values(
    pruned_betas_list[2],
    pruned_assets_list[2],
    pruned_exposure,
    pruned_lambda0,
    "unfiltered_outlier_pruning"
)

##################################################################################
################################### Optional 3:###################################
##################################################################################
# A more sophisticated approach to outlier reduction is RANSAC
avg_returns = df_test_assets.iloc[:, 1:].mean()
asset_names = avg_returns.index

x = unfil_beta_values[asset_names].values
y = avg_returns.values
apply_ransac(x, y, "outlier_removal_RANSAC")