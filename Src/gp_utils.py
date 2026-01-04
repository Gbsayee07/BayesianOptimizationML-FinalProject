# Src/gp_utils.py

import numpy as np
from numpy.linalg import cholesky, solve, slogdet


def rbf_kernel(X1, X2, lengthscales, variance):
    """
    Squared exponential (RBF) kernel.

    k(x,x') = variance * exp(-0.5 * sum(((x - x') / ell)^2))

    Parameters
    ----------
    X1 : (n1, d)
    X2 : (n2, d)
    lengthscales : float or (d,) array
    variance : float

    Returns
    -------
    K : (n1, n2) ndarray
    """
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    ell = np.asarray(lengthscales)
    if ell.ndim == 0:
        ell = np.full(X1.shape[1], ell)

    X1_scaled = X1 / ell
    X2_scaled = X2 / ell

    # ||x - x'||^2 = ||x||^2 + ||x'||^2 - 2 x·x'
    sq_norms1 = np.sum(X1_scaled**2, axis=1)[:, None]
    sq_norms2 = np.sum(X2_scaled**2, axis=1)[None, :]
    dists2 = sq_norms1 + sq_norms2 - 2 * X1_scaled @ X2_scaled.T

    return variance * np.exp(-0.5 * dists2)
# Alias: Step 3 expects squared_exp_kernel()


def squared_exp_kernel(X1, X2, lengthscales, variance):
    return rbf_kernel(X1, X2, lengthscales, variance)

def gp_neg_log_marginal_likelihood(theta, X, y, noise_std, mean_is_constant=True):
    """
    Negative log marginal likelihood for a GP with RBF kernel and constant mean.

    theta = [log_lengthscale_1, ..., log_lengthscale_d, log_sqrt_variance, mean]
    or if mean_is_constant=False, theta excludes the mean term.
    """
    X = np.atleast_2d(X)
    y = np.asarray(y)

    d = X.shape[1]
    log_ell = theta[:d]
    log_sqrt_var = theta[d]
    if mean_is_constant:
        m = theta[d + 1]
    else:
        m = 0.0

    ell = np.exp(log_ell)
    var = np.exp(2 * log_sqrt_var)

    K = rbf_kernel(X, X, ell, var)
    K[np.diag_indices_from(K)] += noise_std**2

    n = X.shape[0]
    y_centered = y - m

    try:
        L = cholesky(K)
    except np.linalg.LinAlgError:
        # If K is not PSD due to numerical issues, penalize heavily
        return 1e10

    alpha = solve(L.T, solve(L, y_centered))

    # log |K|
    sign, logdet = slogdet(K)
    if sign <= 0:
        return 1e10

    nll = 0.5 * y_centered @ alpha + 0.5 * logdet + 0.5 * n * np.log(2 * np.pi)
    return nll


def gp_posterior(X_train, y_train, X_test, noise_std, lengthscales, variance, mean):
    """
    Compute GP posterior mean and variance at X_test.
    """
    X_train = np.atleast_2d(X_train)
    X_test  = np.atleast_2d(X_test)
    y_train = np.asarray(y_train)

    # Training covariance
    K = rbf_kernel(X_train, X_train, lengthscales, variance)
    K[np.diag_indices_from(K)] += noise_std**2

    # Cross covariance
    K_s = rbf_kernel(X_train, X_test, lengthscales, variance)

    # Test covariance diagonal (safer form)
    K_ss = rbf_kernel(X_test, X_test, lengthscales, variance)
    K_ss_diag = np.maximum(np.diag(K_ss), 1e-12)

    # Center target outputs
    y_centered = y_train - mean

    # Robust Cholesky with fallback
    jitter = 1e-8
    for _ in range(6):
        try:
            L = cholesky(K + jitter * np.eye(len(K)))
            break
        except np.linalg.LinAlgError:
            jitter *= 10
    else:
        L = cholesky(K + 1e-3 * np.eye(len(K)))

    # Posterior mean
    alpha = solve(L.T, solve(L, y_centered))
    mu_post = mean + K_s.T @ alpha

    # Posterior variance
    v = solve(L, K_s)
    var_post = K_ss_diag - np.sum(v**2, axis=0)
    var_post = np.maximum(var_post, 1e-12)

    return mu_post, var_post


# --- Extra kernels for model selection (Step 3) ---

def linear_kernel(X1: np.ndarray,
                  X2: np.ndarray,
                  variance: float) -> np.ndarray:
    """
    Simple linear kernel: k(x, x') = variance * x^T x'
    """
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    return variance * (X1 @ X2.T)