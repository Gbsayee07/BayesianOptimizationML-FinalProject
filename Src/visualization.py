# Src/visualization.py
import os
from typing import Callable, Optional

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns

from scipy.stats import gaussian_kde
from Data.branin import branin
from Src.gp_utils import gp_posterior

from Data.branin import branin


def plot_branin_heatmap(
    n_per_dim: int = 1000,
    transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    save_path: str = "Results/Figures/branin_heatmap.png",
) -> None:
    """
    Plot a heatmap of the Branin function over its standard domain.

    Domain:
        x1 in [-5, 10]
        x2 in [0, 15]

    Parameters
    ----------
    n_per_dim : int
        Number of grid points per dimension (default 1000 → 1000x1000 image).
    transform : callable or None
        Optional transform applied to the Branin values before plotting
        (e.g. np.log1p for making the function more "stationary").
    save_path : str
        Where to save the heatmap PNG.
    """
    # Create grid
    x1 = np.linspace(-5.0, 10.0, n_per_dim)
    x2 = np.linspace(0.0, 15.0, n_per_dim)
    X1, X2 = np.meshgrid(x1, x2, indexing="xy")

    # Stack into points of shape (N, 2)
    X = np.stack([X1.ravel(), X2.ravel()], axis=-1)

    # Evaluate Branin
    Z = branin(X).reshape(n_per_dim, n_per_dim)

    # Optional transform (for stationarity experiments)
    if transform is not None:
        Z = transform(Z)

    # Ensure output directory exists
    out_dir = os.path.dirname(save_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Plot heatmap
    plt.figure(figsize=(6, 5))
    # imshow expects (rows, cols) as (y, x); we align x1 (horizontal), x2 (vertical)
    im = plt.imshow(
        Z,
        origin="lower",
        extent=[x1.min(), x1.max(), x2.min(), x2.max()],
        aspect="auto",
        cmap="viridis",
    )
    plt.colorbar(im, label="Branin value" + (" (transformed)" if transform else ""))

    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title("Branin function heatmap" + (" (transformed)" if transform else ""))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    # Example usage: raw Branin heatmap
    plot_branin_heatmap()

    # Example usage: log-transformed Branin heatmap (for stationarity check)
    # plot_branin_heatmap(
    #     transform=lambda z: np.log1p(z - z.min() + 1e-6),
    #     save_path="Results/figures/branin_heatmap_log.png",
    # )



def load_dataset(path: str) -> np.ndarray:
    """
    Load svm.csv or lda.csv and return the objective values (4th column).
    The datasets have the format:
        col1, col2, col3, objective, <ignored column>
    """
    df = pd.read_csv(path, header=None)
    # 4th column (index 3) is the objective value to minimize
    return df.iloc[:, 3].values


def plot_kde_for_dataset(values: np.ndarray, name: str, save_path: str) -> None:
    """
    Plot a basic KDE of the objective values.

    Parameters
    ----------
    values : np.ndarray
        The objective values (1D array).
    name : str
        'svm' or 'lda' for labeling.
    save_path : str
        Where to save the PNG.
    """
    plt.figure(figsize=(6, 4))
    sns.kdeplot(values, fill=True, color="purple", alpha=0.6)
    plt.title(f"KDE of {name.upper()} objective values")
    plt.xlabel("Objective value")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_kde_transformed(values: np.ndarray, name: str, transform, save_path: str) -> None:
    """
    Apply a transformation (log, sqrt, etc.) and plot the KDE.

    Parameters
    ----------
    values : np.ndarray
        Objective values.
    name : str
        'svm' or 'lda'
    transform : callable
        Function to transform the values.
    save_path : str
        Where to save the PNG.
    """
    transformed = transform(values)

    plt.figure(figsize=(6, 4))
    sns.kdeplot(transformed, fill=True, color="green", alpha=0.6)
    plt.title(f"Transformed KDE of {name.upper()} values")
    plt.xlabel("Transformed value")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_gp_posterior_heatmaps(model, n_per_dim, save_mean, save_std):
    """
    Make heatmaps of GP posterior mean and std over the Branin domain.
    """
    import os

    os.makedirs(os.path.dirname(save_mean), exist_ok=True)

    bounds = np.array([[-5.0, 10.0], [0.0, 15.0]])
    x1 = np.linspace(bounds[0, 0], bounds[0, 1], n_per_dim)
    x2 = np.linspace(bounds[1, 0], bounds[1, 1], n_per_dim)
    X1, X2 = np.meshgrid(x1, x2)
    X_test = np.column_stack([X1.ravel(), X2.ravel()])

    mu, var = gp_posterior(
        model["X_train"],
        model["y_train"],
        X_test,
        model["noise_std"],
        model["lengthscales"],
        model["variance"],
        model["mean"],
    )
    std = np.sqrt(var)

    mu_grid = mu.reshape(n_per_dim, n_per_dim)
    std_grid = std.reshape(n_per_dim, n_per_dim)

    # Posterior mean
    plt.figure(figsize=(7, 6))
    plt.imshow(
        mu_grid,
        origin="lower",
        extent=[bounds[0, 0], bounds[0, 1], bounds[1, 0], bounds[1, 1]],
        aspect="auto",
        cmap="viridis",
    )
    plt.colorbar(label="Posterior mean")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("GP posterior mean for Branin")
    plt.tight_layout()
    plt.savefig(save_mean, dpi=150)
    plt.close()

    # Posterior std
    plt.figure(figsize=(7, 6))
    plt.imshow(
        std_grid,
        origin="lower",
        extent=[bounds[0, 0], bounds[0, 1], bounds[1, 0], bounds[1, 1]],
        aspect="auto",
        cmap="viridis",
    )
    plt.colorbar(label="Posterior std")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("GP posterior standard deviation for Branin")
    plt.tight_layout()
    plt.savefig(save_std, dpi=150)
    plt.close()


def plot_residual_zscore_kde(model, n_points=2000, save_path="Results/figures/branin_residual_z_kde.png"):
    """
    Sample a dense set of points, compute residual z-scores between
    the GP posterior mean and true Branin values, and plot a KDE.

    If the GP is well calibrated, these z-scores should look ~ N(0,1).
    """
    import os

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    bounds = np.array([[-5.0, 10.0], [0.0, 15.0]])
    x1 = np.linspace(bounds[0, 0], bounds[0, 1], int(np.sqrt(n_points)))
    x2 = np.linspace(bounds[1, 0], bounds[1, 1], int(np.sqrt(n_points)))
    X1, X2 = np.meshgrid(x1, x2)
    X = np.column_stack([X1.ravel(), X2.ravel()])

    mu, var = gp_posterior(
        model["X_train"],
        model["y_train"],
        X,
        model["noise_std"],
        model["lengthscales"],
        model["variance"],
        model["mean"],
    )
    std = np.sqrt(var)

    # true values
    y_true = np.array([branin(x) for x in X])

    # z-scores of residuals
    z = (y_true - mu) / std

    kde = gaussian_kde(z)
    z_grid = np.linspace(-4, 4, 400)
    density = kde(z_grid)

    plt.figure(figsize=(6, 4))
    plt.plot(z_grid, density, label="Residual z-score KDE")
    plt.xlabel("z")
    plt.ylabel("Density")
    plt.title("KDE of residual z-scores (Branin GP)")
    plt.axvline(0, color="k", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    return z  # you might want to inspect these later



def plot_learning_curves(gaps_ei: np.ndarray,
                         gaps_rs: np.ndarray,
                         name: str,
                         save_path: str,
                         max_steps: int = 30) -> None:
    """
    Plot average gap vs iteration for EI and random search.

    Parameters
    ----------
    gaps_ei : (n_runs, T_ei)
        Gap values for EI policy.
    gaps_rs : (n_runs, T_rs)
        Gap values for random search.
    name : str
        Dataset name for the title.
    save_path : str
        PNG path.
    max_steps : int
        Number of iterations to display (e.g., 30).
    """
    import matplotlib.pyplot as plt

    T_ei = gaps_ei.shape[1]
    T_rs = gaps_rs.shape[1]
    T = min(max_steps, T_ei, T_rs)

    steps = np.arange(1, T + 1)
    mean_ei = gaps_ei[:, :T].mean(axis=0)
    mean_rs = gaps_rs[:, :T].mean(axis=0)

    plt.figure(figsize=(6, 4))
    plt.plot(steps, mean_ei, label="EI (BayesOpt)")
    plt.plot(steps, mean_rs, label="Random search")
    plt.xlabel("Iteration")
    plt.ylabel("Average gap")
    plt.title(f"Learning curves on {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()