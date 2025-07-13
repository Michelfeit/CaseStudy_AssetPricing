import numpy as np
import pandas as pd
from scipy.stats import linregress


def stage_one_fama_macbeth(df_risk_factor: pd.DataFrame,
                           df_assets: pd.DataFrame,
                           startIdx_asset,
                           startIdx_risk_factor):
    """
    Performs the first stage of the Fama-MacBeth two-stage regression procedure.

    We estimate time-series regressions of individual asset returns on a single risk factor.
    It returns the estimated slope coefficients (betas) and intercepts (alphas) for each asset.

    Args:
        df_risk_factor (pd.DataFrame): DataFrame containing the risk factor time series (e.g., macro variables).
        df_assets (pd.DataFrame): DataFrame of asset returns. Columns are assets; rows are time periods.
        startIdx_asset (int): Starting index to slice asset return time series for alignment.
        startIdx_risk_factor (int): Starting index to slice risk factor time series for alignment.

    Returns:
        alpha_values (pd.Series): A Series of intercepts (alphas) for each asset.
        beta_values (pd.Series): A Series of slope coefficients (betas) for each asset.

    Notes:
        - Assets with missing values in either their return series or the risk factor are skipped.
        - Assumes a single-column risk factor DataFrame.
        - Assumes that after slicing, each asset's return series and the risk factor have the same length.
    """
    # Dictionaries to store regression results for each asset
    betas = {}
    alphas = {}

    # Loop through each asset column (excluding the first if it's a date or time column)
    for asset in df_assets.columns[1:]:
        # Slice asset returns and risk factor to align time series
        y = df_assets[asset].values[startIdx_asset:]  # asset returns (dependent variable)
        x = df_risk_factor.values[startIdx_risk_factor:]  # risk factor values (independent variable)

        # Ensure x and y are of the same length
        assert len(x) == len(y), f"Length mismatch: {len(x)} vs {len(y)} for asset {asset}"

        # Only run regression if both series are free of NaNs
        if not np.any(pd.isnull(y)) and not np.any(pd.isnull(x)):
            # Perform linear regression: y = alpha + beta * x + error
            slope, intercept, r_value, p_value, std_err = linregress(x, y)

            # Store results
            # we only require the intercepts and the slope as this corresponds to the alpha and beta values
            betas[asset] = slope
            alphas[asset] = intercept

    # Convert dictionaries to Pandas Series for easy use in stage two
    alpha_values = pd.Series(alphas)
    beta_values = pd.Series(betas)

    return alpha_values, beta_values


def stage_two_fama_macbeth(df_assets: pd.DataFrame, betas: pd.Series):
    """
    Correct Fama-MacBeth second-stage regression.

    Parameters:
        df_assets: pd.DataFrame — asset returns (rows: time, columns: assets)
        betas: pd.Series — factor loadings from stage 1, indexed by asset name

    Returns:
        lambda_mean: float — average price of risk across time
        lambda0_mean: float — average return for assets not exposed to factor
    """
    returns = df_assets.iloc[:, 1:]  # remove first column if non-return
    betas = betas[returns.columns]

    intercepts = []
    slopes = []
    for t in range(len(returns)):
        y_t = returns.iloc[t].values  # returns at time t
        x = betas.values  # constant across t
        if not np.any(np.isnan(y_t)):
            slope, intercept, _, _, _ = linregress(x, y_t)
            slopes.append(slope)
            intercepts.append(intercept)

    lambda_mean = np.mean(slopes)
    lambda0_mean = np.mean(intercepts)

    return lambda_mean, lambda0_mean