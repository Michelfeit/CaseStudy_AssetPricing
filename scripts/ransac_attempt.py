import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor, LinearRegression


def apply_ransac(x, y, name):
    # Ensure inputs are NumPy arrays
    x = np.asarray(x).reshape(-1, 1)  # Convert x to shape (n_samples, 1)
    y = np.asarray(y)  # y can remain 1D

    # Fit RANSAC
    ransac = RANSACRegressor(
        estimator=LinearRegression(),
        residual_threshold=2,  # try increasing or decreasing this
        random_state=42
    )
    ransac.fit(x, y)
    # Get the inlier mask
    inlier_mask = ransac.inlier_mask_
    outlier_mask = ~inlier_mask

    x_limit = (-100, 300)
    y_limit = (-2, 22)
    # Get line for plotting
    line_x = np.linspace(x_limit[0], x_limit[1], 100).reshape(-1, 1)
    line_y = ransac.predict(line_x)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x[inlier_mask], y[inlier_mask], color='black', label='Inliers')
    ax.scatter(x[outlier_mask], y[outlier_mask], color='lightgrey', label='Outliers')

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
    ax.legend()
    ax.legend(loc='center', bbox_to_anchor=(.365, .922))
    ax.grid(False)
    ax.plot(line_x, line_y, color='crimson', linestyle= '-', label='RANSAC Fit')
    plt.title("RANSAC Regression")
    ax.legend()
    filename = os.path.join("results", name)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()