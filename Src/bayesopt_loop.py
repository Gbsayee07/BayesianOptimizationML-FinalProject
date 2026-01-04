# Src/bayesopt_loop.py

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from Data.branin import branin
from Src.gp_utils import gp_neg_log_marginal_likelihood, gp_posterior
from Src.acquisition import expected_improvement

from Src.sobol_utils import sobol_points
from Src.gp_utils import gp_posterior


# ---------------------------------------------------------------------
# Generic GP fit: constant mean + SE kernel (same as fit_gp_to_branin)
# ---------------------------------------------------------------------

def fit_gp_const_se(X, y, noise_std=1e-2):
    """
    GP with constant mean and squared-exponential kernel.
    We optimize hyperparameters by marginal likelihood,
    then clip them to a safe range for stability.
    """
    X = np.atleast_2d(X)
    y = np.asarray(y)
    n, d = X.shape

    # Initial guess
    log_ell0 = np.zeros(d)  # ell ~ exp(0) = 1
    log_sqrt_var0 = np.log(np.std(y) if np.std(y) > 0 else 1.0)
    mean0 = float(np.mean(y))
    theta0 = np.concatenate([log_ell0, [log_sqrt_var0, mean0]])

    def nll(theta):
        return gp_neg_log_marginal_likelihood(theta, X, y, noise_std)

    res = minimize(nll, theta0, method="L-BFGS-B")

    theta_opt = res.x
    # 🔹 Clip logs so ell ∈ [0.1, 5], var ∈ [1e-6, 5]
    log_ell = np.clip(theta_opt[:d], -2.3, 1.6)      # exp(-2.3)≈0.1, exp(1.6)≈5
    log_sqrt_var = np.clip(theta_opt[d], -3.0, 1.6)  # var = exp(2 * log_sqrt_var)

    m = theta_opt[d + 1]

    ell = np.exp(log_ell)
    var = np.exp(2 * log_sqrt_var)

    return {
        "X_train": X,
        "y_train": y,
        "lengthscales": np.clip(ell, 1e-3, 5.0),
        "variance": float(np.clip(var, 1e-6, 5.0)),
        "mean": float(m),
        "noise_std": float(noise_std),
    }


# ---------------------------------------------------------------------
# BO on Branin (continuous 2D domain)
# ---------------------------------------------------------------------

def bo_experiment_branin(n_runs: int = 20,
                         n_init: int = 5,
                         n_iter: int = 30,
                         noise_std: float = 1e-3,
                         grid_size: int = 50,
                         max_evals_random: int = 150,
                         random_state: int = 0) -> dict:
    """
    Run Bayesian optimization and random search on the Branin function.

    Returns a dict with:
      - 'gaps_ei' : (n_runs, n_iter) gaps for EI policy
      - 'gaps_rs' : (n_runs, max_evals_random) gaps for random search
      - 'global_min' : approximate global minimum of Branin
    """
    rng_global = np.random.default_rng(random_state)

    # Branin domain
    bounds = np.array([[-5.0, 10.0], [0.0, 15.0]])

    # Approximate global minimum on a fine grid
    fine_n = 200
    x1_fine = np.linspace(bounds[0, 0], bounds[0, 1], fine_n)
    x2_fine = np.linspace(bounds[1, 0], bounds[1, 1], fine_n)
    X1_fine, X2_fine = np.meshgrid(x1_fine, x2_fine)
    X_fine = np.column_stack([X1_fine.ravel(), X2_fine.ravel()])
    y_fine = np.array([branin(x) for x in X_fine])
    global_min = float(y_fine.min())

    # Candidate grid for EI acquisition
    x1_c = np.linspace(bounds[0, 0], bounds[0, 1], grid_size)
    x2_c = np.linspace(bounds[1, 0], bounds[1, 1], grid_size)
    X1_c, X2_c = np.meshgrid(x1_c, x2_c)
    X_cand = np.column_stack([X1_c.ravel(), X2_c.ravel()])

    gaps_ei = np.zeros((n_runs, n_iter))
    gaps_rs = np.zeros((n_runs, max_evals_random))

    for run in range(n_runs):
        rng = np.random.default_rng(random_state + run)

        # ---------- Initial design ----------
        X_init = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_init, 2))
        y_init = np.array([branin(x) for x in X_init])
        best_init = float(y_init.min())

        # ---------- EI (Bayesian optimization) ----------
        X_ei = X_init.copy()
        y_ei = y_init.copy()
        best_so_far = best_init

        for t in range(n_iter):
            # Fit GP on current data
            model = fit_gp_const_se(X_ei, y_ei, noise_std=noise_std)

            mu_cand, var_cand = gp_posterior(
                model["X_train"],
                model["y_train"],
                X_cand,
                model["noise_std"],
                model["lengthscales"],
                model["variance"],
                model["mean"],
            )
            sigma_cand = np.sqrt(var_cand)

            ei = expected_improvement(mu_cand, sigma_cand, best_so_far)
            j = int(np.argmax(ei))
            x_new = X_cand[j]
            y_new = branin(x_new)

            X_ei = np.vstack([X_ei, x_new])
            y_ei = np.append(y_ei, y_new)
            best_so_far = min(best_so_far, y_new)

            gaps_ei[run, t] = (best_init - best_so_far) / (
                best_init - global_min + 1e-12
            )

        # ---------- Random search ----------
        best_rs = best_init

        for t in range(max_evals_random):
            x_rs = rng.uniform(bounds[:, 0], bounds[:, 1], size=2)
            y_rs = branin(x_rs)
            best_rs = min(best_rs, y_rs)

            gaps_rs[run, t] = (best_init - best_rs) / (
                best_init - global_min + 1e-12
            )

    return {
        "gaps_ei": gaps_ei,
        "gaps_rs": gaps_rs,
        "global_min": global_min,
    }


# ---------------------------------------------------------------------
# BO on SVM / LDA CSV benchfunk datasets (discrete grid)
# ---------------------------------------------------------------------

def bo_experiment_from_csv(path: str,
                           name: str,
                           n_runs: int = 20,
                           n_init: int = 5,
                           n_iter: int = 30,
                           max_evals_random: int = 150,
                           noise_frac: float = 0.05,
                           random_state: int = 0) -> dict:
    """
    Run Bayesian optimization and random search on a discrete benchmark
    (SVM or LDA) defined by a CSV file.

    CSV format: 3 hyperparameters (cols 0–2), objective value (col 3), last col ignored.
    All problems are minimization.

    Returns a dict with:
      - 'gaps_ei' : (n_runs, n_iter)
      - 'gaps_rs' : (n_runs, max_evals_random)
      - 'global_min' : min objective over entire grid
    """
    df = pd.read_csv(path, header=None)
    X_all = df.iloc[:, :3].values
    y_all = df.iloc[:, 3].values
    n_total = len(y_all)

    # 🔹 NEW: normalize the three hyperparameter dimensions
    X_mean = X_all.mean(axis=0)
    X_std = X_all.std(axis=0) + 1e-8
    X_all = (X_all - X_mean) / X_std

    global_min = float(np.min(y_all))

    gaps_ei = np.zeros((n_runs, n_iter))
    gaps_rs = np.zeros((n_runs, max_evals_random))

    for run in range(n_runs):
        rng = np.random.default_rng(random_state + run)

        all_idx = np.arange(n_total)
        init_idx = rng.choice(all_idx, size=n_init, replace=False)

        X_init = X_all[init_idx]
        y_init = y_all[init_idx]
        best_init = float(y_init.min())

        # ----- EI (Bayesian optimization) -----
        observed = np.zeros(n_total, dtype=bool)
        observed[init_idx] = True

        X_ei = X_init.copy()
        y_ei = y_init.copy()
        best_so_far = best_init

        noise_std = noise_frac * np.std(y_ei) if np.std(y_ei) > 0 else noise_frac

        for t in range(n_iter):
            model = fit_gp_const_se(X_ei, y_ei, noise_std=noise_std)

            cand_idx = np.where(~observed)[0]
            X_cand = X_all[cand_idx]

            mu_cand, var_cand = gp_posterior(
                model["X_train"],
                model["y_train"],
                X_cand,
                model["noise_std"],
                model["lengthscales"],
                model["variance"],
                model["mean"],
            )
            sigma_cand = np.sqrt(var_cand)

            ei = expected_improvement(mu_cand, sigma_cand, best_so_far)
            j_rel = int(np.argmax(ei))
            j = cand_idx[j_rel]

            observed[j] = True
            x_new = X_all[j]
            y_new = y_all[j]

            X_ei = np.vstack([X_ei, x_new])
            y_ei = np.append(y_ei, y_new)
            best_so_far = min(best_so_far, y_new)

            gaps_ei[run, t] = (best_init - best_so_far) / (
                best_init - global_min + 1e-12
            )

        # ----- Random search -----
        observed_rs = np.zeros(n_total, dtype=bool)
        observed_rs[init_idx] = True
        best_rs = best_init

        for t in range(max_evals_random):
            cand_idx = np.where(~observed_rs)[0]
            if len(cand_idx) == 0:
                cand_idx = np.arange(n_total)

            j = rng.choice(cand_idx)
            observed_rs[j] = True
            y_rs = y_all[j]

            best_rs = min(best_rs, y_rs)

            gaps_rs[run, t] = (best_init - best_rs) / (
                best_init - global_min + 1e-12
            )

    return {
        "gaps_ei": gaps_ei,
        "gaps_rs": gaps_rs,
        "global_min": global_min,
    }


def _bo_loop_discrete(
    X_all,
    y_all,
    global_min,
    n_runs=20,
    n_init=5,
    n_iter=30,
    max_evals_random=150,
    noise_frac=0.05,
    refit_hyperparams=True,
    observation_noise_std=0.0,
    random_state=0,
    is_branin=False,
):
    """
    Core BO loop shared by:
      - Branin (on a dense candidate set)
      - svm / lda (on discrete CSV grid)

    Parameters
    ----------
    X_all : (N, d)
        All candidate points.
    y_all : (N,)
        True objective at these points (for svm/lda, precomputed).
        For Branin with noise, we'll use branin(X) + noise instead of y_all.
    global_min : float
        True minimum of the noiseless objective over X_all.
    refit_hyperparams : bool
        If True, refit GP hyperparameters every iteration from scratch.
        If False, fit them once on the initial 5 points and keep fixed.
    observation_noise_std : float
        Std of noise added to each observation (for noisy Branin).
    is_branin : bool
        If True, ignore y_all and call branin(x) (plus noise) as needed.

    Returns
    -------
    dict with 'gaps_ei', 'gaps_rs', 'global_min'
    """
    n_total = len(X_all)
    gaps_ei = np.zeros((n_runs, n_iter))
    gaps_rs = np.zeros((n_runs, max_evals_random))

    for run in range(n_runs):
        rng = np.random.default_rng(random_state + run)

        # ----- Initialization -----
        all_idx = np.arange(n_total)
        init_idx = rng.choice(all_idx, size=n_init, replace=False)

        X_init = X_all[init_idx]

        if is_branin:
            # Evaluate Branin (possibly noisy) at init points
            y_init_clean = np.array([branin(x) for x in X_init])
            noise = rng.normal(scale=observation_noise_std, size=n_init)
            y_init = y_init_clean + noise
        else:
            y_init = y_all[init_idx]

        best_init = float(y_init.min())

        # ----- EI state -----
        observed = np.zeros(n_total, dtype=bool)
        observed[init_idx] = True

        X_ei = X_init.copy()
        y_ei = y_init.copy()
        best_so_far = best_init

        # Noise level we pass to the GP (model's assumed noise)
        empirical_std = np.std(y_ei) if np.std(y_ei) > 0 else 1.0
        noise_std = max(noise_frac * empirical_std, observation_noise_std, 1e-3)

        # If we want fixed hyperparameters, fit once on initial data
        if not refit_hyperparams:
            base_model = fit_gp_const_se(X_ei, y_ei, noise_std=noise_std)
            fixed_params = (
                base_model["lengthscales"],
                base_model["variance"],
                base_model["mean"],
            )
        else:
            fixed_params = None

        # ----- EI loop -----
        for t in range(n_iter):
            if refit_hyperparams:
                model = fit_gp_const_se(X_ei, y_ei, noise_std=noise_std)
                ell = model["lengthscales"]
                var = model["variance"]
                m = model["mean"]
            else:
                # Use fixed hyperparams, but recompute posterior with new data
                ell, var, m = fixed_params

            cand_idx = np.where(~observed)[0]
            X_cand = X_all[cand_idx]

            mu_cand, var_cand = gp_posterior(
                X_ei,
                y_ei,
                X_cand,
                noise_std,
                ell,
                var,
                m,
            )
            sigma_cand = np.sqrt(var_cand)
            ei = expected_improvement(mu_cand, sigma_cand, best_so_far)

            j_rel = int(np.argmax(ei))
            j = cand_idx[j_rel]

            observed[j] = True
            x_new = X_all[j]

            if is_branin:
                y_clean = branin(x_new)
                y_new = y_clean + rng.normal(scale=observation_noise_std)
            else:
                y_new = y_all[j]

            X_ei = np.vstack([X_ei, x_new])
            y_ei = np.append(y_ei, y_new)
            best_so_far = min(best_so_far, y_new)

            gaps_ei[run, t] = (best_init - best_so_far) / (
                best_init - global_min + 1e-12
            )

        # ----- Random search -----
        observed_rs = np.zeros(n_total, dtype=bool)
        observed_rs[init_idx] = True
        best_rs = best_init

        for t in range(max_evals_random):
            cand_idx = np.where(~observed_rs)[0]
            if len(cand_idx) == 0:
                cand_idx = np.arange(n_total)

            j = rng.choice(cand_idx)
            observed_rs[j] = True

            x_rs = X_all[j]
            if is_branin:
                y_clean = branin(x_rs)
                y_rs = y_clean + rng.normal(scale=observation_noise_std)
            else:
                y_rs = y_all[j]

            best_rs = min(best_rs, y_rs)

            gaps_rs[run, t] = (best_init - best_rs) / (
                best_init - global_min + 1e-12
            )

    return {
        "gaps_ei": gaps_ei,
        "gaps_rs": gaps_rs,
        "global_min": global_min,
    }



# ---------------------------------------------------------------------
# Bonus A: fixed vs re-learned hyperparameters
# ---------------------------------------------------------------------

def bo_experiment_branin_fixed_vs_relearn(
    n_runs=20,
    n_init=5,
    n_iter=30,
    max_evals_random=150,
    noise_frac=0.05,
    observation_noise_std=0.0,
    random_state=0,
):
    # Build a dense grid of Branin candidates (same as your normal BO)
    bounds = np.array([[-5.0, 10.0], [0.0, 15.0]])
    x1 = np.linspace(bounds[0, 0], bounds[0, 1], 100)
    x2 = np.linspace(bounds[1, 0], bounds[1, 1], 100)
    X1, X2 = np.meshgrid(x1, x2)
    X_all = np.column_stack([X1.ravel(), X2.ravel()])

    # For global_min, we use clean (noise-free) Branin values
    y_clean = np.array([branin(x) for x in X_all])
    global_min = float(y_clean.min())

    # Fixed hyperparameters (fit once)
    res_fixed = _bo_loop_discrete(
        X_all,
        y_clean,      # we won't trust this when is_branin=True, but it's fine
        global_min,
        n_runs=n_runs,
        n_init=n_init,
        n_iter=n_iter,
        max_evals_random=max_evals_random,
        noise_frac=noise_frac,
        refit_hyperparams=False,
        observation_noise_std=observation_noise_std,
        random_state=random_state,
        is_branin=True,
    )

    # Re-learn hyperparameters every iteration
    res_relearn = _bo_loop_discrete(
        X_all,
        y_clean,
        global_min,
        n_runs=n_runs,
        n_init=n_init,
        n_iter=n_iter,
        max_evals_random=max_evals_random,
        noise_frac=noise_frac,
        refit_hyperparams=True,
        observation_noise_std=observation_noise_std,
        random_state=random_state,
        is_branin=True,
    )

    return res_fixed, res_relearn



# ---------------------------------------------------------------------
# Bonus C: Branin with observation noise
# ---------------------------------------------------------------------

def bo_experiment_branin_with_noise(
    observation_noise_std: float,
    n_runs: int = 20,
    n_init: int = 5,
    n_iter: int = 30,
    max_evals_random: int = 150,
    noise_frac: float = 0.05,
    random_state: int = 0,
):
    """
    Same Branin experiment as before, but each observation is corrupted by
    Gaussian noise with given std. The GP's noise_std is set based on
    noise_frac and this observation noise.
    """
    bounds = np.array([[-5.0, 10.0], [0.0, 15.0]])
    x1 = np.linspace(bounds[0, 0], bounds[0, 1], 100)
    x2 = np.linspace(bounds[1, 0], bounds[1, 1], 100)
    X1, X2 = np.meshgrid(x1, x2)
    X_all = np.column_stack([X1.ravel(), X2.ravel()])

    # Global min uses clean Branin (noiseless)
    y_clean = np.array([branin(x) for x in X_all])
    global_min = float(y_clean.min())

    res = _bo_loop_discrete(
        X_all,
        y_clean,
        global_min,
        n_runs=n_runs,
        n_init=n_init,
        n_iter=n_iter,
        max_evals_random=max_evals_random,
        noise_frac=noise_frac,
        refit_hyperparams=True,          # keep the usual "smart" BO
        observation_noise_std=observation_noise_std,
        random_state=random_state,
        is_branin=True,
    )
    return res



def bo_experiment_from_csv_fixed_vs_relearn(
    path: str,
    name: str,
    n_runs: int = 20,
    n_init: int = 5,
    n_iter: int = 30,
    max_evals_random: int = 150,
    noise_frac: float = 0.05,
    random_state: int = 0,
):
    import pandas as pd

    df = pd.read_csv(path, header=None)
    X_all = df.iloc[:, :3].values
    y_all = df.iloc[:, 3].values
    global_min = float(np.min(y_all))

    res_fixed = _bo_loop_discrete(
        X_all,
        y_all,
        global_min,
        n_runs=n_runs,
        n_init=n_init,
        n_iter=n_iter,
        max_evals_random=max_evals_random,
        noise_frac=noise_frac,
        refit_hyperparams=False,
        observation_noise_std=0.0,
        random_state=random_state,
        is_branin=False,
    )

    res_relearn = _bo_loop_discrete(
        X_all,
        y_all,
        global_min,
        n_runs=n_runs,
        n_init=n_init,
        n_iter=n_iter,
        max_evals_random=max_evals_random,
        noise_frac=noise_frac,
        refit_hyperparams=True,
        observation_noise_std=0.0,
        random_state=random_state,
        is_branin=False,
    )

    return res_fixed, res_relearn