# Data/branin.py
import numpy as np
import pandas as pd

def branin(x: np.ndarray) -> np.ndarray:
    """
    Compute the Branin (Branin–Hoo) function.

    Parameters
    ----------
    x : np.ndarray
        Array of shape (..., 2), where the last dimension is (x1, x2).
        Domain: x1 in [-5, 10], x2 in [0, 15].

    Returns
    -------
    np.ndarray
        Branin function values with shape x.shape[:-1].
    """
    x = np.asarray(x)
    x1 = x[..., 0]
    x2 = x[..., 1]

    # Standard Branin constants (matching SFU definition)
    a = 1.0
    b = 5.1 / (4.0 * np.pi**2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * np.pi)

    return a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1.0 - t) * np.cos(x1) + s