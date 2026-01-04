# main.py
import numpy as np
from scipy.stats import ttest_rel
import numpy as np

from Src.visualization import (
    plot_branin_heatmap,
    load_dataset,
    plot_kde_for_dataset,
    plot_kde_transformed,
    plot_gp_posterior_heatmaps,
    plot_residual_zscore_kde,
    plot_learning_curves,
)
from Src.model_fitting import generate_branin_training_data, fit_gp_to_branin
from Src.model_fitting import bic_model_search_branin, bic_model_search_csv

from Src.bayesopt_loop import (
    bo_experiment_branin,
    bo_experiment_from_csv,
    bo_experiment_branin_fixed_vs_relearn,
    bo_experiment_from_csv_fixed_vs_relearn,
    bo_experiment_branin_with_noise,
)




# -----------------------------------------------------
# STEP 1
# -----------------------------------------------------
def run_step1():
    print("=== Step 1: Branin Heatmaps ===")
    plot_branin_heatmap(
        n_per_dim=1000,
        transform=None,
        save_path="Results/Figures/branin_heatmap_raw.png",
    )
    plot_branin_heatmap(
        n_per_dim=1000,
        transform=lambda z: np.log1p(z - z.min() + 1e-6),
        save_path="Results/Figures/branin_heatmap_log.png",
    )

    print("=== Step 1: KDE for svm.csv ===")
    svm_vals = load_dataset("Data/svm.csv")
    plot_kde_for_dataset(svm_vals, "svm", "Results/Figures/svm_kde.png")
    plot_kde_transformed(
        svm_vals, "svm",
        lambda x: np.log1p(x - x.min() + 1e-6),
        "Results/Figures/svm_kde_log.png"
    )

    print("=== Step 1: KDE for lda.csv ===")
    lda_vals = load_dataset("Data/lda.csv")
    plot_kde_for_dataset(lda_vals, "lda", "Results/Figures/lda_kde.png")
    plot_kde_transformed(
        lda_vals, "lda",
        lambda x: np.log1p(x - x.min() + 1e-6),
        "Results/Figures/lda_kde_log.png"
    )

    print("=== Step 1 Completed ===\n")


# -----------------------------------------------------
# STEP 2
# -----------------------------------------------------
def run_step2():
    print("=== Step 2: Generate Branin training data (Sobol) ===")
    X_train, y_train = generate_branin_training_data(n=32)

    print("=== Step 2: Fit GP to Branin ===")
    model = fit_gp_to_branin(X_train, y_train, noise_std=1e-3)
    print("Optimized lengthscales:", model["lengthscales"])
    print("Optimized variance:", model["variance"])
    print("Optimized mean:", model["mean"])

    print("=== Step 2: Posterior heatmaps ===")
    plot_gp_posterior_heatmaps(
        model,
        n_per_dim=200,
        save_mean="Results/Figures/branin_gp_posterior_mean.png",
        save_std="Results/Figures/branin_gp_posterior_std.png",
    )

    print("=== Step 2: Residual z-score KDE ===")
    plot_residual_zscore_kde(
        model,
        n_points=2500,
        save_path="Results/Figures/branin_gp_residual_z_kde.png",
    )

    print("=== Step 2 Completed ===\n")
    return X_train, y_train


# -----------------------------------------------------
# STEP 3
# -----------------------------------------------------

def run_step3(X_train, y_train):
    print("=== Step 3: BIC model search (Branin, SVM, LDA) ===")

    # Branin GP model search
    bic_model_search_branin(X_train, y_train, noise=1e-3)

    # SVM / LDA model search
    bic_model_search_csv("Data/svm.csv", name="svm", n_points=32, noise_frac=0.05)
    bic_model_search_csv("Data/lda.csv", name="lda", n_points=32, noise_frac=0.05)

    print("=== Step 3 Completed ===\n")

# -----------------------------------------------------
# STEP 4
# -----------------------------------------------------

def run_step4():
    print("=== Step 4: Bayesian optimization experiments ===")

    # ---- Run BO on Branin ----
    branin_res = bo_experiment_branin(
        n_runs=20,
        n_init=5,
        n_iter=30,
        random_state=0,
    )
    plot_learning_curves(
        branin_res["gaps_ei"],
        branin_res["gaps_rs"],
        "branin",
        "Results/Figures/bo_learning_branin.png",
    )

    # ---- Run BO on SVM ----
    svm_res = bo_experiment_from_csv(
        path="Data/svm.csv",
        name="svm",
        n_runs=20,
        n_init=5,
        n_iter=30,
        max_evals_random=150,
        noise_frac=0.05,
        random_state=0,
    )
    plot_learning_curves(
        svm_res["gaps_ei"],
        svm_res["gaps_rs"],
        "svm",
        "Results/Figures/bo_learning_svm.png",
    )

    # ---- Run BO on LDA ----
    lda_res = bo_experiment_from_csv(
        path="Data/lda.csv",
        name="lda",
        n_runs=20,
        n_init=5,
        n_iter=30,
        max_evals_random=150,
        noise_frac=0.05,
        random_state=0,
    )
    plot_learning_curves(
        lda_res["gaps_ei"],
        lda_res["gaps_rs"],
        "lda",
        "Results/Figures/bo_learning_lda.png",
    )

    print("=== Step 4 Completed ===")

    # 🔥 ADD THE STATISTICS CALLS HERE:
    compute_gap_statistics("branin", branin_res["gaps_ei"], branin_res["gaps_rs"])
    compute_gap_statistics("svm", svm_res["gaps_ei"], svm_res["gaps_rs"])
    compute_gap_statistics("lda", lda_res["gaps_ei"], lda_res["gaps_rs"])




def run_bonus_A():
    print("\n=== BONUS A: Effect of hyperparameter re-learning ===")

    # ----- Branin -----
    br_fixed, br_re = bo_experiment_branin_fixed_vs_relearn(
        n_runs=20,
        n_init=5,
        n_iter=30,
        max_evals_random=150,
        noise_frac=0.05,
        observation_noise_std=0.0,
        random_state=0,
    )
    print("\n[Bonus A] Branin: Fixed vs Re-learned hyperparameters")
    compute_gap_statistics("branin (fixed)", br_fixed["gaps_ei"], br_fixed["gaps_rs"])
    compute_gap_statistics("branin (relearn)", br_re["gaps_ei"], br_re["gaps_rs"])

    # ----- SVM -----
    svm_fixed, svm_re = bo_experiment_from_csv_fixed_vs_relearn(
        path="Data/svm.csv",
        name="svm",
        n_runs=20,
        n_init=5,
        n_iter=30,
        max_evals_random=150,
        noise_frac=0.05,
        random_state=0,
    )
    print("\n[Bonus A] SVM: Fixed vs Re-learned hyperparameters")
    compute_gap_statistics("svm (fixed)", svm_fixed["gaps_ei"], svm_fixed["gaps_rs"])
    compute_gap_statistics("svm (relearn)", svm_re["gaps_ei"], svm_re["gaps_rs"])

    # ----- LDA -----
    lda_fixed, lda_re = bo_experiment_from_csv_fixed_vs_relearn(
        path="Data/lda.csv",
        name="lda",
        n_runs=20,
        n_init=5,
        n_iter=30,
        max_evals_random=150,
        noise_frac=0.05,
        random_state=0,
    )
    print("\n[Bonus A] LDA: Fixed vs Re-learned hyperparameters")
    compute_gap_statistics("lda (fixed)", lda_fixed["gaps_ei"], lda_fixed["gaps_rs"])
    compute_gap_statistics("lda (relearn)", lda_re["gaps_ei"], lda_re["gaps_rs"])




def run_bonus_C():
    print("\n=== BONUS C: Effect of observation noise on Branin BO ===")

    noise_levels = [0.0, 0.1, 0.5]

    for sigma in noise_levels:
        print(f"\n[Bonus C] Branin with observation noise σ = {sigma}")
        res = bo_experiment_branin_with_noise(
            observation_noise_std=sigma,
            n_runs=20,
            n_init=5,
            n_iter=30,
            max_evals_random=150,
            noise_frac=0.05,
            random_state=0,
        )
        # Reuse our stats function to summarize
        compute_gap_statistics(f"branin_noise_{sigma}", res["gaps_ei"], res["gaps_rs"])




# -------------------------------------------------------------
# Step 4.5 - Statistics: mean gaps and paired t-tests
# -------------------------------------------------------------

def compute_gap_statistics(name, gaps_ei, gaps_rs):
    """
    Compute and print:
        - mean gaps at [30, 60, 90, 120, 150]
        - paired t-test between EI and RS at 30
        - smallest T where p > 0.05
    """

    eval_points = [30, 60, 90, 120, 150]

    print(f"\n=== Step 4.5 Statistics for {name.upper()} ===")

    # Random search only has 30 BO iterations in BO, but RS has 150 evaluations
    # gaps_ei: shape (n_runs, 30)
    # gaps_rs: shape (n_runs, 150)

    for T in eval_points:
        T_eff_ei = min(T, gaps_ei.shape[1])
        T_eff_rs = min(T, gaps_rs.shape[1])

        mean_ei = np.mean(gaps_ei[:, T_eff_ei - 1])
        mean_rs = np.mean(gaps_rs[:, T_eff_rs - 1])

        print(f"\n--- {name.upper()} @ {T} evaluations ---")
        print(f"Mean gap EI: {mean_ei:.4f}")
        print(f"Mean gap RS: {mean_rs:.4f}")

    # Paired t-test at 30 evaluations
    print("\nPaired t-test EI(30) vs RS(30):")
    t_stat, p_val = ttest_rel(gaps_ei[:, 29], gaps_rs[:, 29])
    print(f"t = {t_stat:.4f}, p = {p_val:.4g}")

    # Find smallest T where p > 0.05
    for T in eval_points:
        T_e = min(T, gaps_ei.shape[1]) - 1
        T_r = min(T, gaps_rs.shape[1]) - 1
        t_stat, p_val = ttest_rel(gaps_ei[:, T_e], gaps_rs[:, T_r])
        if p_val > 0.05:
            print(f"\nRandom search becomes statistically indistinguishable at T = {T}")
            print(f"(p = {p_val:.4f})")
            break



def main():
    run_step1()
    X_train, y_train = run_step2()
    run_step3(X_train, y_train)
    run_step4()
    run_bonus_A()      
    run_bonus_C()    


if __name__ == "__main__":
    main()
