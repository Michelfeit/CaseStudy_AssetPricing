import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation

from scripts.fama_macbeth import stage_two_fama_macbeth


def generate_pruning_animation(pruned_betas_list, pruned_assets_list):
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter([], [], color='black', s=30)
    line, = ax.plot([], [], color='crimson', linestyle='-')
    title = ax.set_title("")

    # Fixed axis limits (optional)
    x_limit = (-100, 300)
    y_limit = (0, 22)
    ax.set_xlim(x_limit)
    ax.set_ylim(y_limit)

    # Zero axes
    ax.axhline(0, color='black', linewidth=1.5)
    ax.axvline(0, color='black', linewidth=1.5)
    ax.set_xlabel("Beta (Factor Exposure)")
    ax.set_ylabel("Average Return")

    def update(frame):
        betas = pruned_betas_list[frame]
        assets = pruned_assets_list[frame]

        # Regression
        pruned_exposure, pruned_lambda0  = stage_two_fama_macbeth(assets, betas)

        # Compute average returns (excluding the first column if not an asset)
        avg_returns = assets.iloc[:, 1:].mean()
        asset_names = avg_returns.index

        x = betas[asset_names].values
        y = avg_returns.values

        # Update scatter
        scatter.set_offsets(np.column_stack((x, y)))

        # Update regression line
        x_line = np.array(x_limit)
        y_line = pruned_lambda0 + pruned_exposure * x_line
        line.set_data(x_line, y_line)

        # Update title
        title.set_text(f"Pruning {frame + 1} Outlier{'s' if frame > 0 else ''}")

        return scatter, line, title

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
    ax.plot([x_limit[1]], [0], marker=r'>', color='black', markersize=10, transform=ax.transData, clip_on=False)
    ax.plot([0], [y_limit[1]], marker=r'^', color='black', markersize=10, transform=ax.transData, clip_on=False)

    # Add labels and title
    ax.set_xlabel('Beta (Factor Exposure)', loc='right')
    ax.set_ylabel('Average Return', loc='top')
    ax.set_title(f'Second-Stage Fama-MacBeth: pruning', pad=15)

    ax.grid(False)
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(pruned_betas_list), interval=2000, blit=True)

    # Or save to GIF
    anim.save("results/regression_pruning.gif", writer='pillow', fps=1)
    print("GIF saved as regression_pruning.gif")