import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

def plot_second_stage_result_from_values(betas: pd.Series,
                                         df_assets: pd.DataFrame,
                                         slope: float,
                                         intercept: float,
                                         model_label: str,
                                         save_dir: str = "results"):
    """
    Plots and saves the second-stage Fama-MacBeth regression result using pre-computed values.

    Args:
        betas (pd.Series): First-stage beta estimates (indexed by asset name).
        df_assets (pd.DataFrame): Asset returns (columns are asset names).
        slope (float): Pre-computed slope from second-stage regression.
        intercept (float): Pre-computed intercept from second-stage regression.
        model_label (str): Label for the model (e.g., 'filtered' or 'unfiltered').
        save_dir (str): Directory to save the plot. Default is 'results'.
    """
    # Ensure output directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Compute average returns (excluding the first column if not an asset)
    avg_returns = df_assets.iloc[:, 1:].mean()
    asset_names = avg_returns.index

    x = betas[asset_names].values
    y = avg_returns.values

    x_limit = (-100, 300)
    y_limit = (-2, 22)
    y_pred = [intercept + slope * x_val for x_val in x_limit]

    # Create plot
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, color='black', s=30, label='Assets')
    ax.plot(x_limit, y_pred, color='crimson', linestyle='-', label='Fitted line')

    # Set axis limits
    ax.set_xlim(x_limit)
    ax.set_ylim(y_limit)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Move bottom and left spines to zero
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    # Hide ticks on top/right
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    # Add arrows at ends of axes
    ax.plot([x_limit[1]], [0], marker=r'>', color='black', markersize= 10, transform=ax.transData, clip_on=False)
    ax.plot([0], [y_limit[1]], marker=r'^', color='black', markersize= 10, transform=ax.transData, clip_on=False)

    # Add labels and title
    ax.set_xlabel('Beta (Factor Exposure)', loc='right')
    ax.set_ylabel('Average Return', loc='top')
    ax.set_title(f'Second-Stage Fama-MacBeth: {model_label}', pad=15)
    ax.legend()
    ax.legend(loc='center', bbox_to_anchor=(.365, .922))
    ax.grid(False)

    # Save figure
    filename = os.path.join(save_dir, f'{model_label.lower()}_second_stage.png')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {filename}")
