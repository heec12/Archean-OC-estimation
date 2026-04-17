"""
mass_balance_7oxides.py
=======================
Bayesian inference of Archean oceanic crust bulk composition.
7-oxide system: SiO2, TiO2, Al2O3, FeO, MgO, CaO, Na2O (NCFMAST)

Two-stage sampling (v2.1):
  Stage 1 — MvNormal likelihood fits mu_UC and Sigma_UC to observed
             komatiitic basalt data.
  Stage 2 — For each crustal architecture scenario, sample delta and f
             subject to geological constraints on x_lower.

Usage
-----
    python mass_balance_7oxides.py --data komatiitic_basalt_subset.xlsx
    python mass_balance_7oxides.py --data /path/to/data.xlsx --output ./results --cores 4

Requirements
------------
    pip install pymc arviz openpyxl pandas numpy
"""

import argparse
import os

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt


# =============================================================================
# 1. USER SETTINGS  (override via CLI args or edit directly)
# =============================================================================
SHEET_NAME  = "strict_komatiitic_basalt"
RANDOM_SEED = 42

OXIDES = [
    "SiO2_pct",
    "TiO2_pct",
    "Al2O3_pct",
    "FeOtot_pct",
    "MgO_pct",
    "CaO_pct",
    "Na2O_pct",
]

MAX_LOI   = 5.0
MIN_TOTAL = 95.0
MAX_TOTAL = 105.0


# =============================================================================
# 2. SCENARIO DEFINITIONS
# =============================================================================
# delta = X_bulk - mu_UC  (bulk-to-upper-crust compositional offset)
# Values should be grounded in literature analogues for Archean lower crust
# (Phanerozoic xenolith suites, seismic velocity constraints, etc.)

SCENARIO_PRIORS = {
    "homogeneous_crust": {
        # Small offset: crust is compositionally uniform with depth
        "delta_mu"   : np.array([-0.3, -0.02, -0.2,  0.0,  0.8,  0.2, -0.1]),
        "delta_sigma": np.array([ 0.5,  0.05,  0.5,  0.5,  0.8,  0.5,  0.1]),
        "f_bounds"   : (0.25, 0.45),
        # Soft geological expectation for the lower crust
        "lc_mu"      : np.array([48.5,  0.4, 10.5, 11.0, 13.5, 10.3,  1.1]),
        "lc_sigma"   : np.array([ 2.0,  0.3,  2.0,  2.0,  3.0,  2.0,  0.8]),
        # Hard broad bounds
        "lc_lower"   : np.array([45.0,  0.0,  7.0,  4.0,  6.0,  6.0,  0.2]),
        "lc_upper"   : np.array([56.0,  2.5, 18.0, 18.0, 18.0, 16.0,  4.0]),
    },
    "layered_cumulate_lower_crust": {
        # Larger primitive shift: olivine-rich cumulate lower crust
        "delta_mu"   : np.array([-0.8, -0.06, -0.8,  0.0,  2.0,  0.4, -0.3]),
        "delta_sigma": np.array([ 0.7,  0.07,  0.7,  0.6,  1.2,  0.7,  0.15]),
        "f_bounds"   : (0.20, 0.45),
        # Soft geological expectation for the lower crust
        "lc_mu"      : np.array([46.5,  0.3,  8.5, 11.0, 17.0, 10.8,  0.6]),
        "lc_sigma"   : np.array([ 2.5,  0.25,  2.5,  2.0,  4.0,  2.5,  0.6]),
        # Hard broad bounds
        "lc_lower"   : np.array([42.0,  0.0,  3.0,  4.0,  8.0,  5.0,  0.0]),
        "lc_upper"   : np.array([54.0,  2.0, 16.0, 18.0, 25.0, 18.0,  2.5]),
    },
}


# =============================================================================
# 3. HELPER FUNCTIONS
# =============================================================================

def fe2o3t_to_feot(fe2o3t):
    return 0.8998 * fe2o3t


def renormalize_anhydrous(df, oxide_cols):
    """Re-normalise selected oxide columns to sum to 100 row-wise."""
    out = df.copy()
    row_sum = out[oxide_cols].sum(axis=1)
    out[oxide_cols] = out[oxide_cols].div(row_sum, axis=0) * 100.0
    return out


def positive_composition_potential(x, name="positive_comp"):
    """Hard positivity constraint — rejects draws with any negative oxide."""
    return pm.Potential(name, pt.switch(pt.all(x > 0), 0.0, -np.inf))


def bounded_range_potential(x, lower, upper, name="bounded_range"):
    """Hard oxide-wise bounds applied simultaneously."""
    cond = pt.all((x >= lower) & (x <= upper))
    return pm.Potential(name, pt.switch(cond, 0.0, -np.inf))


def soft_normal_potential(x, mu, sigma, name="soft_prior"):
    """Soft geological plausibility via element-wise Normal log-probability."""
    logp = pm.logp(pm.Normal.dist(mu=mu, sigma=sigma, shape=len(mu)), x).sum()
    return pm.Potential(name, logp)


# =============================================================================
# 4. DATA LOADING
# =============================================================================

def load_data(file_path, sheet_name=SHEET_NAME):
    """Load, filter, deduplicate, and renormalise the upper crust data."""
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Fe conversion: Fe2O3T -> FeOtot
    df["FeOtot_pct"] = fe2o3t_to_feot(df["Fe2O3T"])

    # Drop rows missing any of the 7 oxides
    df = df.dropna(subset=OXIDES)

    # Quality filters
    if "LOI_pct" in df.columns:
        df = df[df["LOI_pct"] <= MAX_LOI]
    if "Total_calc" in df.columns:
        df = df[(df["Total_calc"] >= MIN_TOTAL) & (df["Total_calc"] <= MAX_TOTAL)]

    # De-duplicate on SampleID
    if "SampleID" in df.columns:
        df = df.drop_duplicates(subset="SampleID")

    # Renormalise to 100% on the 7-oxide system
    df_norm = renormalize_anhydrous(df, OXIDES)

    meta_cols  = [c for c in ["SampleID", "TERRANE", "Rocktype"] if c in df_norm.columns]
    upper_pool = df_norm[meta_cols + OXIDES].reset_index(drop=True)

    print(f"Samples loaded after filtering and dedup: {len(upper_pool)}")
    print(upper_pool[OXIDES].describe().round(2).to_string())

    X_obs             = upper_pool[OXIDES].to_numpy(dtype=float)
    uc_mean_empirical = X_obs.mean(axis=0)
    uc_mean_empirical = uc_mean_empirical / uc_mean_empirical.sum() * 100.0
    uc_std_empirical  = X_obs.std(axis=0)

    print("\nEmpirical upper-crust mean (prior centre for mu_UC):")
    print(pd.Series(uc_mean_empirical, index=OXIDES).round(3).to_string())
    print("\nEmpirical upper-crust std:")
    print(pd.Series(uc_std_empirical, index=OXIDES).round(3).to_string())

    return X_obs, uc_mean_empirical, uc_std_empirical


# =============================================================================
# 5. MODEL BUILDERS
# =============================================================================

def build_uc_model(X_obs, uc_mean_empirical, uc_std_empirical):
    """
    Stage 1: fit the upper crust MvNormal model to observed data.
    Estimates mu_UC (latent true UC mean) and Sigma_UC (covariance).
    Run once; shared across all scenarios.
    """
    N, D = X_obs.shape
    with pm.Model() as uc_model:
        # Latent true upper-crust mean
        # Prior broad enough not to dominate the likelihood (3x empirical std)
        mu_UC = pm.MvNormal(
            "mu_UC",
            mu    = uc_mean_empirical,
            cov   = np.diag((uc_std_empirical * 3.0) ** 2),
            shape = D,
        )
        # LKJCholeskyCov: stable PyMC pattern for MvNormal covariance
        # eta=2 => weakly regularises correlations toward identity
        chol_UC, corr_UC, sigma_UC = pm.LKJCholeskyCov(
            "chol_UC",
            n           = D,
            eta         = 2.0,
            sd_dist     = pm.HalfNormal.dist(sigma=uc_std_empirical * 1.5, shape=D),
            compute_corr= True,
        )
        # Likelihood: each observed sample ~ MvNormal(mu_UC, chol chol^T)
        _ = pm.MvNormal(
            "x_obs_likelihood",
            mu      = mu_UC,
            chol    = chol_UC,
            observed= X_obs,
        )
    return uc_model


def build_scenario_model(mu_UC_fixed, priors):
    """
    Stage 2: sample delta and f given a fixed mu_UC.

    mu_UC is fixed to the Stage 1 posterior mean to avoid the identification
    ridge between mu_UC and delta (x_bulk = mu_UC + delta would otherwise
    allow the two to shift along a flat ridge while keeping x_bulk constant,
    causing NUTS to hit max_treedepth and produce terrible mixing).

    Parameters
    ----------
    mu_UC_fixed : np.ndarray, shape (7,)
        Posterior mean of mu_UC from Stage 1.
    priors : dict
        Scenario-specific prior parameters.
    """
    D = len(mu_UC_fixed)

    delta_mu    = priors["delta_mu"]
    delta_sigma = priors["delta_sigma"]
    f_lo, f_hi  = priors["f_bounds"]
    lc_mu       = priors["lc_mu"]
    lc_sigma    = priors["lc_sigma"]
    lc_lower    = priors["lc_lower"]
    lc_upper    = priors["lc_upper"]

    with pm.Model() as scenario_model:

        # Bulk offset — scenario-specific prior
        delta = pm.Normal("delta", mu=delta_mu, sigma=delta_sigma, shape=D)

        # Upper crust fraction
        f = pm.Uniform("f", lower=f_lo, upper=f_hi)

        # Bulk crust composition
        x_bulk_raw = mu_UC_fixed + delta
        positive_composition_potential(x_bulk_raw, name="positive_bulk")
        x_bulk = pm.Deterministic(
            "x_bulk",
            100.0 * x_bulk_raw / pt.sum(x_bulk_raw),
        )

        # Implied lower crust from mass balance
        x_lower_raw = (x_bulk - f * mu_UC_fixed) / (1.0 - f)
        positive_composition_potential(x_lower_raw, name="positive_lower")
        x_lower = pm.Deterministic(
            "x_lower",
            100.0 * x_lower_raw / pt.sum(x_lower_raw),
        )

        # Closure diagnostic: pre-normalisation sum should be ~100
        _ = pm.Deterministic("closure_error", pt.sum(x_lower_raw) - 100.0)

        # Hard geological bounds on lower crust
        bounded_range_potential(x_lower, lc_lower, lc_upper, name="lc_hard_bounds")

        # Soft geological plausibility prior on lower crust
        soft_normal_potential(x_lower, lc_mu, lc_sigma, name="lc_soft_prior")

    return scenario_model


# =============================================================================
# 6. SAMPLING
# =============================================================================

def run_stage1(X_obs, uc_mean_empirical, uc_std_empirical, cores):
    """Fit Stage 1 upper crust model and return trace + posterior mu_UC stats."""
    print("\n" + "="*60)
    print("STAGE 1: Upper crust MvNormal model")
    print("="*60)

    uc_model = build_uc_model(X_obs, uc_mean_empirical, uc_std_empirical)

    with uc_model:
        uc_trace = pm.sample(
            draws        = 2000,
            tune         = 2000,
            chains       = 4,
            cores        = cores,
            random_seed  = RANDOM_SEED,
            target_accept= 0.90,
            return_inferencedata=True,
        )

    # Convergence check
    print("\n--- Stage 1 convergence diagnostics ---")
    uc_summary = az.summary(uc_trace, var_names=["mu_UC"], round_to=3)
    rhat_bad = uc_summary["r_hat"] > 1.01
    ess_bad  = uc_summary["ess_bulk"] < 400
    print("R-hat > 1.01 :", uc_summary.index[rhat_bad].tolist() or "none")
    print("ESS < 400    :", uc_summary.index[ess_bad].tolist()  or "none")
    print(f"Divergences  : {uc_trace.sample_stats['diverging'].values.sum()}")

    mu_UC_post_mean = np.asarray(
        uc_trace.posterior["mu_UC"].mean(dim=("chain", "draw")).values
    )
    mu_UC_post_std = np.asarray(
        uc_trace.posterior["mu_UC"].std(dim=("chain", "draw")).values
    )

    print("\nPosterior mu_UC (mean ± std):")
    print(pd.DataFrame({
        "mu_UC_mean": mu_UC_post_mean,
        "mu_UC_std" : mu_UC_post_std,
    }, index=OXIDES).round(3).to_string())

    return uc_trace, mu_UC_post_mean, mu_UC_post_std


def run_stage2(mu_UC_post_mean, cores):
    """Fit Stage 2 scenario models and return traces and summaries."""
    traces    = {}
    summaries = {}

    for scenario_name, priors in SCENARIO_PRIORS.items():
        print("\n" + "="*60)
        print(f"STAGE 2 — scenario: {scenario_name}")
        print("="*60)

        model = build_scenario_model(mu_UC_post_mean, priors)

        with model:
            trace = pm.sample(
                draws        = 2000,
                tune         = 2000,
                chains       = 4,
                cores        = cores,
                random_seed  = RANDOM_SEED,
                target_accept= 0.92,
                return_inferencedata=True,
            )

        traces[scenario_name] = trace

        # Convergence diagnostics
        print(f"\n--- Convergence: {scenario_name} ---")
        diag_vars    = ["f", "delta", "x_bulk", "x_lower", "closure_error"]
        diag_summary = az.summary(trace, var_names=diag_vars, round_to=3)
        rhat_flag = diag_summary["r_hat"] > 1.01
        ess_flag  = diag_summary["ess_bulk"] < 400
        print("R-hat > 1.01 :", diag_summary.index[rhat_flag].tolist() or "none")
        print("ESS < 400    :", diag_summary.index[ess_flag].tolist()  or "none")
        print(f"Divergences  : {trace.sample_stats['diverging'].values.sum()}")

        summaries[scenario_name] = diag_summary
        print(f"\nPosterior summary — {scenario_name}:")
        print(diag_summary.to_string())

    return traces, summaries


# =============================================================================
# 7. RESULTS EXTRACTION
# =============================================================================

def extract_results(traces, mu_UC_post_mean, mu_UC_post_std):
    """Compute posterior means and stds for all quantities of interest."""
    posterior_results = {}

    for scenario_name, trace in traces.items():
        def pmean(var): return np.asarray(
            trace.posterior[var].mean(dim=("chain", "draw")).values)
        def pstd(var): return np.asarray(
            trace.posterior[var].std(dim=("chain", "draw")).values)

        post_bulk    = pmean("x_bulk")
        post_lower   = pmean("x_lower")
        post_delta   = pmean("delta")
        post_f       = float(trace.posterior["f"].mean().values)
        post_closure = float(trace.posterior["closure_error"].mean().values)
        std_bulk     = pstd("x_bulk")
        std_lower    = pstd("x_lower")

        posterior_results[scenario_name] = {
            "mu_UC_mean"   : mu_UC_post_mean,
            "mu_UC_std"    : mu_UC_post_std,
            "f_mean"       : post_f,
            "x_bulk_mean"  : post_bulk,
            "x_bulk_std"   : std_bulk,
            "x_lower_mean" : post_lower,
            "x_lower_std"  : std_lower,
            "delta_mean"   : post_delta,
            "closure_error": post_closure,
        }

        print(f"\n{'='*60}")
        print(f"Results: {scenario_name}")
        print(f"{'='*60}")
        print(f"Posterior mean f   = {post_f:.3f}")
        print(f"Mean closure error = {post_closure:+.3f} wt%  (ideal: 0)")
        print()
        print(pd.DataFrame({
            "mu_UC_mean"   : mu_UC_post_mean,
            "mu_UC_std"    : mu_UC_post_std,
            "x_bulk_mean"  : post_bulk,
            "x_bulk_std"   : std_bulk,
            "x_lower_mean" : post_lower,
            "x_lower_std"  : std_lower,
            "delta_mean"   : post_delta,
        }, index=OXIDES).round(3).to_string())
        print("\nNote: x_bulk_std/x_lower_std reflect uncertainty in delta and f only.")
        print("Add mu_UC_std in quadrature for full propagated uncertainty.")

    return posterior_results


# =============================================================================
# 8. SAVE OUTPUTS
# =============================================================================

def save_outputs(output_dir, uc_trace, traces, summaries,
                 posterior_results, uc_mean_empirical):
    """Save all posterior summaries, composition tables, and netCDF traces."""
    os.makedirs(output_dir, exist_ok=True)

    # Empirical anchor reference
    pd.Series(uc_mean_empirical, index=OXIDES, name="UC_mean_empirical").to_csv(
        os.path.join(output_dir, "upper_crust_mean_empirical.csv")
    )

    # Stage 1 trace
    uc_trace.to_netcdf(os.path.join(output_dir, "trace_stage1_uc.nc"))

    for scenario_name, result in posterior_results.items():
        # Posterior summary table
        summaries[scenario_name].to_csv(
            os.path.join(output_dir, f"summary_{scenario_name}.csv")
        )

        # Main composition outputs (mean ± std)
        pd.DataFrame({
            "mu_UC_mean"  : result["mu_UC_mean"],
            "mu_UC_std"   : result["mu_UC_std"],
            "x_bulk_mean" : result["x_bulk_mean"],
            "x_bulk_std"  : result["x_bulk_std"],
            "x_lower_mean": result["x_lower_mean"],
            "x_lower_std" : result["x_lower_std"],
            "delta_mean"  : result["delta_mean"],
        }, index=OXIDES).round(4).to_csv(
            os.path.join(output_dir, f"posterior_compositions_{scenario_name}.csv")
        )

        # Full trace for downstream ensemble use
        traces[scenario_name].to_netcdf(
            os.path.join(output_dir, f"trace_{scenario_name}.nc")
        )

    print(f"\nOutputs saved to: {output_dir}/")
    for fname in sorted(os.listdir(output_dir)):
        print(f"  {fname}")


# =============================================================================
# 9. MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Bayesian Archean oceanic crust bulk composition estimation"
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to the Excel file (e.g. komatiitic_basalt_subset.xlsx)"
    )
    parser.add_argument(
        "--sheet", default=SHEET_NAME,
        help=f"Sheet name in the Excel file (default: {SHEET_NAME})"
    )
    parser.add_argument(
        "--output", default="./bayesian_lower_crust_outputs",
        help="Output directory for results (default: ./bayesian_lower_crust_outputs)"
    )
    parser.add_argument(
        "--cores", type=int, default=2,
        help="Number of CPU cores for parallel chain sampling (default: 2)"
    )
    args = parser.parse_args()

    # -- Load data --
    X_obs, uc_mean_empirical, uc_std_empirical = load_data(args.data, args.sheet)

    # -- Stage 1: fit upper crust model --
    uc_trace, mu_UC_post_mean, mu_UC_post_std = run_stage1(
        X_obs, uc_mean_empirical, uc_std_empirical, args.cores
    )

    # -- Stage 2: fit scenario models --
    traces, summaries = run_stage2(mu_UC_post_mean, args.cores)

    # -- Extract and print results --
    posterior_results = extract_results(traces, mu_UC_post_mean, mu_UC_post_std)

    # -- Save outputs --
    save_outputs(
        args.output, uc_trace, traces, summaries,
        posterior_results, uc_mean_empirical
    )

    # -- Downstream note --
    print("\n" + "="*60)
    print("DOWNSTREAM: Perple_X ensemble")
    print("="*60)
    print("To propagate uncertainty into Perple_X, draw from the posterior:")
    print()
    print("  import xarray as xr")
    print("  trace = az.from_netcdf('trace_homogeneous_crust.nc')")
    print("  bulk  = trace.posterior['x_bulk'].values.reshape(-1, 7)")
    print("  idx   = np.random.choice(len(bulk), size=200, replace=False)")
    print("  bulk_ensemble = bulk[idx]  # 200 compositions -> 200 Perple_X runs")


if __name__ == "__main__":
    main()
