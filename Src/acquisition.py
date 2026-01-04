# Src/acquisition.py

import numpy as np
from scipy.stats import norm


def expected_improvement(mu: np.ndarray,
                         sigma: np.ndarray,
                         y_best: float,
                         xi: float = 0.0) -> np.ndarray:
    """
    Expected Improvement acquisition function for *minimization*.

    EI(x) = E[max(0, y_best - f(x) - xi)]
          = (y_best - mu - xi) * Φ(Z) + σ * φ(Z),
    where Z = (y_best - mu - xi) / σ.

    Parameters
    ----------
    mu : (n,) array
        Posterior mean at candidate points.
    sigma : (n,) array
        Posterior std dev at candidate points.
    y_best : float
        Best (smallest) objective value seen so far.
    xi : float
        Exploration parameter (usually small, e.g., 0.0 or 0.01).

    Returns
    -------
    ei : (n,) array
        EI at each candidate point.
    """
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)

    # Guard against zero variance
    sigma_clipped = np.maximum(sigma, 1e-12)

    improvement = y_best - mu - xi
    Z = improvement / sigma_clipped

    ei = improvement * norm.cdf(Z) + sigma_clipped * norm.pdf(Z)
    ei[sigma <= 0.0] = 0.0  # if variance is numerically 0, no improvement

    return ei