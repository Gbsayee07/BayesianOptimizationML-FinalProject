# Src/model_fitting.py

import numpy as np
from scipy.optimize import minimize

from Data.branin import branin
from Src.sobol_utils import sobol_points
from Src.gp_utils import gp_neg_log_marginal_likelihood, gp_posterior

from typing import Dict, List, Tuple
from Src.gp_utils import squared_exp_kernel, linear_kernel


def generate_branin_training_data(n=32):
    """
    Generate n Sobol points in the Branin domain and evaluate the function.

    Returns
    -------
    X_train : (n, 2)
    y_train : (n,)
    """
    bounds = np.array([[-5.0, 10.0], [0.0, 15.0]])
    X_train = sobol_points(n=n, dim=2, bounds=bounds)
    y_train = np.array([branin(x) for x in X_train])
    return X_train, y_train


def fit_gp_to_branin(X_train, y_train, noise_std=1e-3):
    """
    Fit a GP with RBF kernel and constant mean to (X_train, y_train).

    Returns
    -------
    model : dict
        Contains 'lengthscales', 'variance', 'mean', 'noise_std', 'X_train', 'y_train'.
    """
    X_train = np.atleast_2d(X_train)
    y_train = np.asarray(y_train)
    n, d = X_train.shape

    # Initial guesses
    log_ell0 = np.log(np.ones(d) * 2.0)
    log_sqrt_var0 = np.log(np.std(y_train) if np.std(y_train) > 0 else 1.0)
    mean0 = np.mean(y_train)

    theta0 = np.concatenate([log_ell0, [log_sqrt_var0, mean0]])

    def objective(theta):
        return gp_neg_log_marginal_likelihood(theta, X_train, y_train, noise_std)

    res = minimize(objective, theta0, method="L-BFGS-B")

    theta_opt = res.x
    log_ell_opt = theta_opt[:d]
    log_sqrt_var_opt = theta_opt[d]
    mean_opt = theta_opt[d + 1]

    lengthscales = np.exp(log_ell_opt)
    variance = np.exp(2 * log_sqrt_var_opt)

    # ---- Fix #3: Clip extreme values ----
    lengthscales = np.clip(lengthscales, 1e-3, 1e3)
    variance = np.clip(variance, 1e-6, 1e6)
    mean_opt = np.clip(mean_opt, -1e3, 1e3)

    model = {
        "X_train": X_train,
        "y_train": y_train,
        "lengthscales": lengthscales,
        "variance": variance,
        "mean": mean_opt,
        "noise_std": noise_std,
        "opt_result": res,
    }
    return model


def gp_posterior_on_grid(model, n_per_dim=200, transform=None):
    """
    Compute GP posterior mean & std on a dense grid over the Branin domain.

    Parameters
    ----------
    model : dict
        Output of fit_gp_to_branin.
    n_per_dim : int
        Grid resolution per dimension.
    transform : callable or None
        If provided, apply to the *targets* before fitting,
        and invert when interpreting residuals externally.

    Returns
    -------
    X1, X2 : meshgrid
    mu_grid : (n_per_dim, n_per_dim)
    std_grid : (n_per_dim, n_per_dim)
    """
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

    return X1, X2, mu_grid, std_grid


# ---------------------------------------------------------------------
# Helpers: number of params, build kernel, NLL, etc.
# ---------------------------------------------------------------------

def _num_params(mean_type: str, kernel_type: str, dim: int) -> int:
    """How many hyperparameters does this (mean, kernel) combo have?"""
    mean_params = 1 if mean_type == "const" else 0

    if kernel_type == "se":
        # d lengthscales + log variance
        kernel_params = dim + 1
    elif kernel_type == "se+lin":
        # se: d lengthscales + log var + linear: log var
        kernel_params = dim + 2
    else:
        raise ValueError(f"Unknown kernel_type: {kernel_type}")

    return mean_params + kernel_params


def _build_kernel_from_theta(theta: np.ndarray,
                             X: np.ndarray,
                             mean_type: str,
                             kernel_type: str,
                             noise: float) -> Tuple[np.ndarray, float]:
    """
    Given theta, build K(X,X) + noise^2 I and scalar mean m.
    We parameterize everything in log-space except mean.
    """
    n, d = X.shape
    idx = 0

    # mean
    if mean_type == "const":
        m = theta[0]
        idx += 1
    elif mean_type == "zero":
        m = 0.0
    else:
        raise ValueError(f"Unknown mean_type: {mean_type}")

    # kernel
    if kernel_type == "se":
        # ell_1..ell_d, log sigma2
        ell = np.exp(theta[idx:idx + d])
        sigma2 = np.exp(theta[idx + d])
        K = squared_exp_kernel(X, X, ell, sigma2)
    elif kernel_type == "se+lin":
        ell = np.exp(theta[idx:idx + d])
        sigma2_se = np.exp(theta[idx + d])
        sigma2_lin = np.exp(theta[idx + d + 1])
        K = (squared_exp_kernel(X, X, ell, sigma2_se) +
             linear_kernel(X, X, sigma2_lin))
    else:
        raise ValueError(f"Unknown kernel_type: {kernel_type}")

    K = K + (noise ** 2) * np.eye(n) + 1e-6 * np.eye(n)
    return K, m


def _neg_log_marginal(theta: np.ndarray,
                      X: np.ndarray,
                      y: np.ndarray,
                      mean_type: str,
                      kernel_type: str,
                      noise: float) -> float:
    """Negative log marginal likelihood for GP."""
    n = len(y)
    K, m = _build_kernel_from_theta(theta, X, mean_type, kernel_type, noise)
    y_centered = y - m

    # robust Cholesky with jitter
    jitter = 1e-6
    try:
        L = np.linalg.cholesky(K + jitter * np.eye(n))
    except np.linalg.LinAlgError:
        return 1e25  # punish bad hyperparams

    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_centered))
    log_det = 2.0 * np.sum(np.log(np.diag(L)))

    log_marginal = -0.5 * y_centered @ alpha \
                   - 0.5 * log_det \
                   - 0.5 * n * np.log(2.0 * np.pi)
    return -log_marginal


# ---------------------------------------------------------------------
# Top-level: fit GP with BIC
# ---------------------------------------------------------------------

def fit_gp_with_bic(X: np.ndarray,
                    y: np.ndarray,
                    mean_type: str,
                    kernel_type: str,
                    noise: float) -> Dict:
    """
    Maximize GP marginal likelihood, then compute BIC.

    Returns dict with:
      'mean_type', 'kernel_type', 'theta_opt', 'log_marginal', 'bic', 'num_params'
    """
    n, d = X.shape
    k_params = _num_params(mean_type, kernel_type, d)

    # zeros => ell=1, sigma2=1, mean=0 (if used)
    theta0 = np.zeros(k_params)

    res = minimize(
        _neg_log_marginal,
        theta0,
        args=(X, y, mean_type, kernel_type, noise),
        method="L-BFGS-B",
    )

    theta_opt = res.x

    # interpret parameters
    idx = 0

    # mean
    if mean_type == "const":
        mean = theta_opt[idx]
        mean = np.clip(mean, -1e3, 1e3)
        idx += 1
    else:
        mean = 0.0

    # kernel
    if kernel_type == "se":
        ell = np.exp(theta_opt[idx:idx + d])
        ell = np.clip(ell, 1e-3, 1e3)
        sigma2 = np.exp(theta_opt[idx + d])
        sigma2 = np.clip(sigma2, 1e-6, 1e6)

    elif kernel_type == "se+lin":
        ell = np.exp(theta_opt[idx:idx + d])
        ell = np.clip(ell, 1e-3, 1e3)
        sigma2_se = np.exp(theta_opt[idx + d])
        sigma2_se = np.clip(sigma2_se, 1e-6, 1e6)
        sigma2_lin = np.exp(theta_opt[idx + d + 1])
        sigma2_lin = np.clip(sigma2_lin, 1e-6, 1e6)


    log_marginal = -_neg_log_marginal(theta_opt, X, y, mean_type, kernel_type, noise)
    bic = k_params * np.log(n) - 2.0 * log_marginal

    return {
        "mean_type": mean_type,
        "kernel_type": kernel_type,
        "theta_opt": theta_opt,
        "log_marginal": log_marginal,
        "bic": bic,
        "num_params": k_params,
    }


# ---------------------------------------------------------------------
#  Convenience wrappers for this project
# ---------------------------------------------------------------------

def bic_model_search_branin(X_train: np.ndarray,
                            y_train: np.ndarray,
                            noise: float = 1e-3) -> List[Dict]:
    """
    Small model search on Branin.
    Returns list of models sorted by BIC (ascending).
    """
    model_specs = [
        ("zero", "se"),
        ("const", "se"),
        ("const", "se+lin"),
    ]

    results = []
    print("\nBIC model search on BRANIN:")
    for mean_type, kernel_type in model_specs:
        print(f"  Fitting mean={mean_type}, kernel={kernel_type}")
        res = fit_gp_with_bic(X_train, y_train, mean_type, kernel_type, noise)
        print(f"    log p(y|X,θ) = {res['log_marginal']:.3f}, BIC = {res['bic']:.3f}")
        results.append(res)

    results_sorted = sorted(results, key=lambda r: r["bic"])
    best = results_sorted[0]
    print("\n  >> Best Branin model by BIC:")
    print(f"     mean={best['mean_type']}, kernel={best['kernel_type']}, "
          f"BIC={best['bic']:.3f}, log-evidence={best['log_marginal']:.3f}\n")
    return results_sorted


def bic_model_search_csv(path: str,
                         name: str,
                         n_points: int = 32,
                         noise_frac: float = 0.05) -> List[Dict]:
    """
    Randomly subsample n_points from svm/lda CSV and run same model search.

    CSV format: 3 hyperparameters (cols 0–2), objective (col 3), ignore others.
    """
    import pandas as pd

    df = pd.read_csv(path, header=None)
    X_full = df.iloc[:, :3].values
    y_full = df.iloc[:, 3].values

    rng = np.random.default_rng(123)
    idx = rng.choice(len(y_full), size=n_points, replace=False)
    X = X_full[idx]
    y = y_full[idx]

    noise = noise_frac * np.std(y)

    print(f"\nBIC model search on {name.upper()} (n={n_points}, noise≈{noise:.3g}):")
    model_specs = [
        ("zero", "se"),
        ("const", "se"),
        ("const", "se+lin"),
    ]

    results = []
    for mean_type, kernel_type in model_specs:
        print(f"  Fitting mean={mean_type}, kernel={kernel_type}")
        res = fit_gp_with_bic(X, y, mean_type, kernel_type, noise)
        print(f"    log p(y|X,θ) = {res['log_marginal']:.3f}, BIC = {res['bic']:.3f}")
        results.append(res)

    results_sorted = sorted(results, key=lambda r: r["bic"])
    best = results_sorted[0]
    print(f"\n  >> Best {name.upper()} model by BIC:")
    print(f"     mean={best['mean_type']}, kernel={best['kernel_type']}, "
          f"BIC={best['bic']:.3f}, log-evidence={best['log_marginal']:.3f}\n")
    return results_sorted


