# Src/sobol_utils.py

import numpy as np
from scipy.stats import qmc


def sobol_points(n: int, dim: int, bounds):
    """
    Generate Sobol quasi-random points in a box-bounded domain.

    Parameters
    ----------
    n : int
        Number of points.
    dim : int
        Dimensionality of the space.
    bounds : list or array-like of shape (dim, 2)
        [(lower_1, upper_1), ..., (lower_d, upper_d)].

    Returns
    -------
    X : (n, dim) ndarray
        Points in the given bounds.
    """
    bounds = np.asarray(bounds)
    assert bounds.shape == (dim, 2)

    sampler = qmc.Sobol(d=dim, scramble=True)
    # base points in [0,1]^dim
    u = sampler.random(n)
    # scale to bounds
    lower, upper = bounds[:, 0], bounds[:, 1]
    X = lower + u * (upper - lower)
    return X