"""
build_composition_design.py
===========================
Stage 1 of the Archean oceanic crust bound-H2O pipeline.
Replaces mass_balance_7oxides_v2.py.

WHAT CHANGED AND WHY
--------------------
v2 fit an MvNormal to the `strict_komatiitic_basalt` subset and exported a
tight cloud of bulk compositions (MgO 12 +/- 1). The rock class was fixed by
data selection, upstream of any inference, so the ensemble could not answer
"how much does composition matter".

This script instead treats composition as an EXPERIMENTAL DESIGN over a broad
compilation:

    2-D design:  MgO bin  x  Al2O3/TiO2 type (Al-depleted vs Al-undepleted)

MgO is sampled UNIFORMLY across bins, not according to the compilation's
natural abundance, because the publication record is biased and we want
coverage. The output is therefore NOT a probability distribution and its mean
is meaningless -- the deliverable is a response curve of bound H2O vs MgO.
If volumetric constraints on komatiite abundance turn up later, the existing
ensemble can be importance-reweighted without rerunning Perple_X.

KEY DESIGN CHOICES
------------------
1. NO LOI FILTER. Komatiites routinely carry 6-12 wt% LOI from
   serpentinisation; MAX_LOI = 5.0 would have emptied every high-MgO bin.
   Alteration is screened on immobile / ratio-based criteria instead, after
   anhydrous renormalisation.

2. COMPOSITIONAL DATA HANDLED PROPERLY. v2 renormalised rows to sum to 100
   and then fit a full-rank 7x7 covariance, so the likelihood lived on a 6-D
   hyperplane in 7-D space and could gain unbounded density by driving the
   Cholesky toward singularity (which is why it needed target_accept=0.95 and
   max_treedepth=15 to mix). Here the trend is fit in additive log-ratio (ALR)
   coordinates with SiO2 as the reference: given MgO, the other six oxides sum
   to (100 - MgO), leaving exactly 5 free dimensions. So we regress 5 ALR
   coordinates on MgO. Closure is exact by construction, positivity is
   guaranteed, and the covariance is full-rank.

3. UPPER AND LOWER CRUST ARE COUPLED. The lower crust is not sampled from an
   independent prior -- it is the cumulate complement of the upper-crust draw
   in the same design cell. The crystallising assemblage is a function of the
   upper-crust MgO (olivine-only at komatiitic MgO, progressively joined by
   opx, cpx and plagioclase as MgO falls), with olivine Fo set by Fe-Mg
   exchange with the liquid. A trapped-intercumulus-liquid fraction keeps
   Al2O3 and Na2O non-zero so that amphibole can form in the lower crust.

4. f IS NOT A PERPLE_X INPUT. Bound H2O is not linear in bulk composition, so
   mass-balancing the layers BEFORE the phase equilibria is the same
   E[f(x)] != f(E[x]) error the whole redesign is meant to avoid. Upper and
   lower crust are run separately and the WATER is mixed afterwards, which
   also means f becomes a free post-processing scalar instead of a scenario
   axis. That drops the run count from 2 scenarios x 200 samples to ~200 runs
   and lets f be swept continuously for free.

5. NO HARD-WALL POTENTIALS. The -np.inf `pm.Potential` walls in v2 gave NUTS
   a discontinuous log-density with infinite gradients -- a standard
   divergence generator. Positivity and closure are now structural.

DATA REQUIREMENT
----------------
This needs a compilation spanning roughly MgO 8-32 wt%: komatiite through
komatiitic basalt through tholeiite. The `strict_komatiitic_basalt` sheet is
by definition ~10-18 wt% MgO, so pointed at that sheet every bin above 18
will be EMPTY. The script prints a bin-occupancy table and refuses to write a
manifest with empty cells unless --allow-gaps is passed. Point --sheet at the
broad compilation.

USAGE
-----
    python build_composition_design.py \
        --data komatiitic_basalt_subset.xlsx \
        --sheet all_archean_volcanics \
        --output ./design_outputs \
        --mode regression --replicates 4 --cores 4

OUTPUTS
-------
    composition_manifest.csv      <- the file the Julia Perple_X driver reads
    coverage_report.csv           <- analyses per design cell
    trend_posterior_<altype>.nc   <- ALR regression traces (regression mode)
    design_coverage.png           <- compilation + bins + sampled points

Requirements: pip install pymc arviz openpyxl pandas numpy matplotlib
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd

# =============================================================================
# 1. SETTINGS
# =============================================================================

RANDOM_SEED = 42

OXIDES = [
    "SiO2_pct", "TiO2_pct", "Al2O3_pct",
    "FeOtot_pct", "MgO_pct", "CaO_pct", "Na2O_pct",
]

# ALR parameterisation: MgO is the design index (conditioned on, not modelled),
# SiO2 is the ALR reference (closes the composition), and the remaining five
# oxides are the free coordinates.
MGO_OXIDE = "MgO_pct"
ALR_REF   = "SiO2_pct"
ALR_NUM   = ["TiO2_pct", "Al2O3_pct", "FeOtot_pct", "CaO_pct", "Na2O_pct"]

MGO_CENTER = 16.0  # centring for the regression, wt%

# --- Design grid -------------------------------------------------------------
# 12 bins of 2 wt% from 8 to 32. MgO is drawn uniformly within each bin.
MGO_EDGES = np.arange(8.0, 32.0 + 1e-9, 2.0)

# Al2O3/TiO2 splits Al-depleted (Barberton-type, ~10-12) from Al-undepleted
# (Munro-type, ~20, near-chondritic). This matters more for garnet stability
# than the komatiite / komatiitic basalt line does, which is why it is the
# second design axis rather than something to average over.
AL_TI_SPLIT = 15.0
AL_TYPES    = ["Al_depleted", "Al_undepleted"]

N_REPLICATES_DEFAULT = 4   # 12 bins x 2 Al types x 4 = 96 pairs = 192 runs

# --- Alteration screening (replaces the LOI filter) --------------------------
# Applied AFTER anhydrous renormalisation. Ratio-based where possible because
# ratios of immobile elements survive serpentinisation; absolute wt% does not.
SCREEN = {
    "SiO2_pct"  : (36.0, 60.0),
    "MgO_pct"   : ( 5.0, 42.0),
    "Al2O3_pct" : ( 1.5, 20.0),
    "CaO_pct"   : ( 1.0, 18.0),
    "FeOtot_pct": ( 4.0, 20.0),
    "TiO2_pct"  : (0.05,  3.0),
}
AL_TI_RANGE    = (4.0, 45.0)   # outside this is analytical junk, not petrology
CAO_AL2O3_RANGE = (0.3, 2.0)   # flags Ca loss (carbonate/silica alteration)
ANHYDROUS_TOTAL_RANGE = (95.0, 105.0)  # on the raw major-element sum

# Multiplicative replacement for zeros before the log-ratio transform.
# Altered komatiites can report Na2O at or below detection.
ZERO_IMPUTE = {"Na2O_pct": 0.02, "TiO2_pct": 0.02}

# --- Coupled lower crust: cumulate model -------------------------------------
# Crystallising assemblage as a function of upper-crust (liquid) MgO. Anchors
# are interpolated linearly and renormalised. Komatiitic liquids crystallise
# olivine alone; opx, cpx and finally plagioclase join as MgO falls.
CUMULATE_ANCHOR_MGO = np.array([ 8.0, 12.0, 16.0, 20.0, 24.0, 30.0])
CUMULATE_MODES = {
    "olivine":       np.array([0.20, 0.35, 0.55, 0.75, 0.90, 1.00]),
    "orthopyroxene": np.array([0.05, 0.10, 0.15, 0.15, 0.10, 0.00]),
    "clinopyroxene": np.array([0.35, 0.35, 0.25, 0.10, 0.00, 0.00]),
    "plagioclase":   np.array([0.40, 0.20, 0.05, 0.00, 0.00, 0.00]),
}

# Fraction of trapped intercumulus liquid in the lower crust. Pure cumulates
# are unrealistic and an Al-free olivine cumulate cannot make amphibole, which
# would produce a spurious dry lower crust.
TRAPPED_LIQUID_FRAC = 0.15

# Fe-Mg olivine/liquid exchange coefficient (molar).
KD_FE_MG = 0.30

# Mineral end-member compositions, wt% oxide, in OXIDES order.
# Olivine is computed from Fo, the rest are representative mafic-cumulate
# analyses. Sums are renormalised to 100 on assembly.
MINERAL_COMPOSITIONS = {
    # SiO2, TiO2, Al2O3, FeO, MgO, CaO, Na2O
    "orthopyroxene": np.array([56.5, 0.10,  1.60,  7.00, 33.50,  1.20, 0.02]),
    "clinopyroxene": np.array([52.0, 0.45,  3.40,  5.20, 16.80, 21.50, 0.30]),
    "plagioclase":   np.array([48.0, 0.02, 32.80,  0.40,  0.15, 16.20, 2.30]),
}

# Molar masses for the olivine calculation
MW = {"SiO2": 60.084, "MgO": 40.3044, "FeO": 71.844,
      "Mg2SiO4": 140.6931, "Fe2SiO4": 203.7771}


# =============================================================================
# 2. DATA LOADING AND SCREENING
# =============================================================================

def fe2o3t_to_feot(fe2o3t):
    return 0.8998 * fe2o3t


def renormalize_anhydrous(df, oxide_cols):
    out = df.copy()
    row_sum = out[oxide_cols].sum(axis=1)
    out[oxide_cols] = out[oxide_cols].div(row_sum, axis=0) * 100.0
    return out


def load_and_screen(file_path, sheet_name, verbose=True):
    """
    Load the broad compilation, convert Fe, screen on alteration-robust
    criteria, renormalise anhydrous, and assign Al type.

    Deliberately does NOT filter on LOI.
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    n0 = len(df)

    # Fe conversion. Accept either Fe2O3T or a pre-computed FeOtot.
    if "FeOtot_pct" not in df.columns:
        if "Fe2O3T" in df.columns:
            df["FeOtot_pct"] = fe2o3t_to_feot(df["Fe2O3T"])
        elif "FeO_pct" in df.columns:
            df["FeOtot_pct"] = df["FeO_pct"]
        else:
            raise KeyError("Need one of FeOtot_pct, Fe2O3T or FeO_pct.")

    missing = [c for c in OXIDES if c not in df.columns]
    if missing:
        raise KeyError(f"Missing oxide columns in sheet '{sheet_name}': {missing}")

    df = df.dropna(subset=OXIDES).copy()
    n_complete = len(df)

    # Screen on the RAW anhydrous total before renormalising. A raw major
    # sum far from 100 (once LOI is set aside) means a bad analysis, not
    # hydration.
    raw_total = df[OXIDES].sum(axis=1)
    df = df[(raw_total >= ANHYDROUS_TOTAL_RANGE[0]) &
            (raw_total <= ANHYDROUS_TOTAL_RANGE[1])]
    n_total_ok = len(df)

    # Renormalise to 100 on the 7-oxide anhydrous system.
    df = renormalize_anhydrous(df, OXIDES)

    # Ratio-based alteration screens
    df["al_ti_ratio"]   = df["Al2O3_pct"] / df["TiO2_pct"].replace(0.0, np.nan)
    df["cao_al2o3"]     = df["CaO_pct"]   / df["Al2O3_pct"].replace(0.0, np.nan)

    mask = pd.Series(True, index=df.index)
    for col, (lo, hi) in SCREEN.items():
        mask &= df[col].between(lo, hi)
    mask &= df["al_ti_ratio"].between(*AL_TI_RANGE)
    mask &= df["cao_al2o3"].between(*CAO_AL2O3_RANGE)
    df = df[mask]
    n_screened = len(df)

    if "SampleID" in df.columns:
        df = df.drop_duplicates(subset="SampleID")
    n_dedup = len(df)

    # Al type
    df["al_type"] = np.where(df["al_ti_ratio"] < AL_TI_SPLIT,
                             "Al_depleted", "Al_undepleted")

    # Zero imputation for the log-ratio transform
    for col, val in ZERO_IMPUTE.items():
        n_imp = int((df[col] < val).sum())
        if n_imp:
            df.loc[df[col] < val, col] = val
            if verbose:
                print(f"  imputed {n_imp} sub-threshold {col} values to {val}")
    df = renormalize_anhydrous(df, OXIDES)

    if verbose:
        print("\n--- Data screening ---")
        print(f"  rows in sheet                : {n0}")
        print(f"  complete 7-oxide analyses    : {n_complete}")
        print(f"  raw anhydrous total in range : {n_total_ok}")
        print(f"  passed alteration screens    : {n_screened}")
        print(f"  after SampleID dedup         : {n_dedup}")
        print(f"\n  MgO range: {df['MgO_pct'].min():.1f} - "
              f"{df['MgO_pct'].max():.1f} wt%")
        print(f"  Al_depleted   (Al2O3/TiO2 <  {AL_TI_SPLIT}): "
              f"{(df['al_type'] == 'Al_depleted').sum()}")
        print(f"  Al_undepleted (Al2O3/TiO2 >= {AL_TI_SPLIT}): "
              f"{(df['al_type'] == 'Al_undepleted').sum()}")
        print("\n" + df[OXIDES].describe().round(2).to_string())

    return df.reset_index(drop=True)


def coverage_report(df, verbose=True):
    """Analyses per design cell. Empty cells are the thing to worry about."""
    rows = []
    for i in range(len(MGO_EDGES) - 1):
        lo, hi = MGO_EDGES[i], MGO_EDGES[i + 1]
        in_bin = df[(df["MgO_pct"] >= lo) & (df["MgO_pct"] < hi)]
        for al in AL_TYPES:
            rows.append({
                "mgo_bin_index": i,
                "mgo_lo": lo,
                "mgo_hi": hi,
                "mgo_bin_center": 0.5 * (lo + hi),
                "al_type": al,
                "n_analyses": int((in_bin["al_type"] == al).sum()),
            })
    cov = pd.DataFrame(rows)
    if verbose:
        print("\n--- Design cell occupancy ---")
        pivot = cov.pivot(index="mgo_bin_center", columns="al_type",
                          values="n_analyses")
        print(pivot.to_string())
        empty = cov[cov["n_analyses"] == 0]
        if len(empty):
            print(f"\n  WARNING: {len(empty)} of {len(cov)} design cells are EMPTY.")
            print("  If the high-MgO cells are empty you are pointed at the")
            print("  komatiitic basalt subset, not the broad compilation.")
    return cov


# =============================================================================
# 3. ALR TREND MODEL (the Bayesian part that actually has a likelihood)
# =============================================================================

def to_alr(df):
    """
    Additive log-ratio coordinates with SiO2 as reference.
    Returns (N, 5) array of log(x_i / x_SiO2) for the five free oxides.
    """
    ref = df[ALR_REF].to_numpy(dtype=float)
    num = df[ALR_NUM].to_numpy(dtype=float)
    return np.log(num / ref[:, None])


def from_alr(alr_vec, mgo):
    """
    Invert the ALR transform at a given MgO.

    Given MgO, the other six oxides must sum to (100 - MgO). With
    r_i = x_i / x_SiO2 for the five free oxides,

        x_SiO2 * (1 + sum_i r_i) = 100 - MgO

    so SiO2 is determined and every oxide is strictly positive. Closure is
    exact -- no renormalisation step, no closure_error diagnostic needed.
    """
    r = np.exp(alr_vec)
    x_sio2 = (100.0 - mgo) / (1.0 + r.sum())
    out = np.empty(len(OXIDES))
    out[OXIDES.index(ALR_REF)] = x_sio2
    out[OXIDES.index(MGO_OXIDE)] = mgo
    for name, ri in zip(ALR_NUM, r):
        out[OXIDES.index(name)] = x_sio2 * ri
    return out


def fit_alr_trend(df_sub, al_type, cores, seed, draws=1500, tune=1500):
    """
    Bayesian multivariate regression of the 5 ALR coordinates on MgO.

        alr = intercept + slope * (MgO - MGO_CENTER) + eps
        eps ~ MvNormal(0, Sigma),  Sigma via LKJCholeskyCov

    Posterior uncertainty in the coefficients AND the residual covariance both
    propagate into the sampled compositions, so parameter uncertainty and
    natural rock-to-rock heterogeneity are both represented.
    """
    import pymc as pm

    mgo = df_sub[MGO_OXIDE].to_numpy(dtype=float)
    Y = to_alr(df_sub)
    N, D = Y.shape
    mgo_c = mgo - MGO_CENTER

    with pm.Model() as model:
        intercept = pm.Normal("intercept", mu=Y.mean(axis=0),
                              sigma=2.0, shape=D)
        slope = pm.Normal("slope", mu=0.0, sigma=0.5, shape=D)

        chol, _, _ = pm.LKJCholeskyCov(
            "chol", n=D, eta=2.0,
            sd_dist=pm.HalfNormal.dist(sigma=Y.std(axis=0) + 1e-3, shape=D),
            compute_corr=True,
        )

        mu = intercept + slope * mgo_c[:, None]
        pm.MvNormal("obs", mu=mu, chol=chol, observed=Y)

        idata = pm.sample(
            draws=draws, tune=tune, chains=4, cores=cores,
            random_seed=seed, target_accept=0.9,
            return_inferencedata=True, progressbar=False,
        )

    import arviz as az
    summ = az.summary(idata, var_names=["intercept", "slope"], round_to=3)
    n_div = int(idata.sample_stats["diverging"].values.sum())
    print(f"\n  [{al_type}] N = {N}, divergences = {n_div}")
    bad_rhat = summ.index[summ["r_hat"] > 1.01].tolist()
    print(f"  [{al_type}] R-hat > 1.01: {bad_rhat or 'none'}")

    return idata


def _cholesky_from_posterior(post, c, d, D):
    """
    Reconstruct the residual Cholesky factor for one posterior draw.

    With compute_corr=True, PyMC stores the correlation matrix and the standard
    deviations as Deterministics ("chol_corr", "chol_stds") and the underlying
    free RV in packed lower-triangular form
    ("chol_cholesky-cov-packed__"). Prefer corr + stds because those names are
    stable; fall back to unpacking.
    """
    if "chol_corr" in post and "chol_stds" in post:
        corr = post["chol_corr"].values[c, d]
        stds = post["chol_stds"].values[c, d]
        cov = corr * np.outer(stds, stds)
        # jitter guards against tiny numerical asymmetry / indefiniteness
        cov = 0.5 * (cov + cov.T) + 1e-12 * np.eye(D)
        return np.linalg.cholesky(cov)

    packed_keys = [k for k in post.data_vars if "cholesky-cov-packed" in k]
    if packed_keys:
        L = np.zeros((D, D))
        L[np.tril_indices(D)] = post[packed_keys[0]].values[c, d]
        return L

    raise KeyError(
        "Cannot find the residual covariance in the posterior. Expected "
        "'chol_corr' + 'chol_stds' or a packed Cholesky variable."
    )


def draw_from_trend(idata, mgo_target, rng):
    """
    One composition draw at a target MgO from the ALR trend posterior.

    Draws a random posterior index, so coefficient uncertainty (epistemic) and
    the residual scatter (real rock-to-rock heterogeneity) both propagate.
    """
    post = idata.posterior
    c = int(rng.integers(post.sizes["chain"]))
    d = int(rng.integers(post.sizes["draw"]))

    intercept = post["intercept"].values[c, d]
    slope = post["slope"].values[c, d]
    D = len(intercept)

    L = _cholesky_from_posterior(post, c, d, D)

    mu = intercept + slope * (mgo_target - MGO_CENTER)
    eps = L @ rng.standard_normal(D)
    return from_alr(mu + eps, mgo_target)


def draw_stratified(df_sub, rng):
    """One composition drawn as a real analysed rock from this design cell."""
    i = rng.integers(len(df_sub))
    return df_sub.iloc[i][OXIDES].to_numpy(dtype=float)


# =============================================================================
# 4. COUPLED LOWER CRUST
# =============================================================================

def olivine_composition(mgo_liq, feo_liq):
    """
    Olivine in Fe-Mg exchange equilibrium with the liquid.
    Fo = 1 / (1 + KD * (Fe/Mg)_liq_molar)
    """
    mg_mol = mgo_liq / MW["MgO"]
    fe_mol = feo_liq / MW["FeO"]
    fo = 1.0 / (1.0 + KD_FE_MG * (fe_mol / max(mg_mol, 1e-9)))
    fo = float(np.clip(fo, 0.70, 0.96))

    mass = fo * MW["Mg2SiO4"] + (1.0 - fo) * MW["Fe2SiO4"]
    comp = np.zeros(len(OXIDES))
    comp[OXIDES.index("SiO2_pct")] = 100.0 * MW["SiO2"] / mass
    comp[OXIDES.index("MgO_pct")] = 100.0 * fo * 2.0 * MW["MgO"] / mass
    comp[OXIDES.index("FeOtot_pct")] = 100.0 * (1.0 - fo) * 2.0 * MW["FeO"] / mass
    return comp, fo


def cumulate_modes(mgo_liq):
    """Interpolated crystallising assemblage at a given liquid MgO."""
    modes = {}
    for phase, anchors in CUMULATE_MODES.items():
        modes[phase] = float(np.interp(mgo_liq, CUMULATE_ANCHOR_MGO, anchors))
    tot = sum(modes.values())
    return {k: v / tot for k, v in modes.items()}


def build_lower_crust(x_upper, trapped_frac=TRAPPED_LIQUID_FRAC):
    """
    Cumulate complement of an upper-crust composition.

    Coupled by construction: the assemblage and the olivine Fo both follow from
    the upper-crust liquid, so a komatiitic upper crust cannot sit on a
    low-MgO lower crust the way independently-sampled layers would allow.
    """
    mgo_liq = x_upper[OXIDES.index("MgO_pct")]
    feo_liq = x_upper[OXIDES.index("FeOtot_pct")]

    ol_comp, fo = olivine_composition(mgo_liq, feo_liq)
    modes = cumulate_modes(mgo_liq)

    cumulate = modes["olivine"] * ol_comp
    for phase in ("orthopyroxene", "clinopyroxene", "plagioclase"):
        cumulate = cumulate + modes[phase] * MINERAL_COMPOSITIONS[phase]

    # Trapped intercumulus liquid keeps Al2O3 / Na2O non-zero
    x_lower = (1.0 - trapped_frac) * cumulate + trapped_frac * x_upper
    x_lower = 100.0 * x_lower / x_lower.sum()
    return x_lower, modes, fo


# =============================================================================
# 5. BUILD THE DESIGN
# =============================================================================

def build_design(df, mode, n_replicates, seed, cores, allow_gaps, output_dir):
    rng = np.random.default_rng(seed)
    cov = coverage_report(df)

    # Fit one ALR trend per Al type (regression mode only)
    traces = {}
    if mode == "regression":
        print("\n--- Fitting ALR trend models ---")
        for al in AL_TYPES:
            sub = df[df["al_type"] == al]
            if len(sub) < 10:
                warnings.warn(
                    f"Only {len(sub)} analyses for {al}; trend will be "
                    f"prior-dominated."
                )
            if len(sub) < 3:
                print(f"  [{al}] too few analyses, skipping")
                continue
            traces[al] = fit_alr_trend(sub, al, cores, seed)
            traces[al].to_netcdf(
                os.path.join(output_dir, f"trend_posterior_{al}.nc"))

    rows = []
    pair_id = 0
    skipped = []

    for i in range(len(MGO_EDGES) - 1):
        lo, hi = MGO_EDGES[i], MGO_EDGES[i + 1]
        center = 0.5 * (lo + hi)

        for al in AL_TYPES:
            cell = df[(df["MgO_pct"] >= lo) & (df["MgO_pct"] < hi) &
                      (df["al_type"] == al)]

            if mode == "stratified" and len(cell) == 0:
                skipped.append((center, al))
                continue
            if mode == "regression" and al not in traces:
                skipped.append((center, al))
                continue

            for rep in range(n_replicates):
                # MgO uniform within the bin: design coverage, not abundance
                mgo_target = float(rng.uniform(lo, hi))

                if mode == "stratified":
                    x_upper = draw_stratified(cell, rng)
                    mgo_target = x_upper[OXIDES.index("MgO_pct")]
                else:
                    x_upper = draw_from_trend(traces[al], mgo_target, rng)

                x_lower, modes, fo = build_lower_crust(x_upper)

                al_ti_u = (x_upper[OXIDES.index("Al2O3_pct")] /
                           max(x_upper[OXIDES.index("TiO2_pct")], 1e-6))

                base = {
                    "pair_id": pair_id,
                    "mgo_bin_index": i,
                    "mgo_bin_center": center,
                    "mgo_target": round(mgo_target, 4),
                    "mgo_upper": round(float(x_upper[OXIDES.index("MgO_pct")]), 4),
                    "al_type": al,
                    "al_ti_upper": round(float(al_ti_u), 3),
                    "replicate": rep,
                    "olivine_fo": round(fo, 4),
                    "mode_olivine": round(modes["olivine"], 4),
                    "mode_opx": round(modes["orthopyroxene"], 4),
                    "mode_cpx": round(modes["clinopyroxene"], 4),
                    "mode_plag": round(modes["plagioclase"], 4),
                    "trapped_liquid": TRAPPED_LIQUID_FRAC,
                }

                for layer, comp in (("upper", x_upper), ("lower", x_lower)):
                    row = dict(base)
                    row["layer"] = layer
                    row["run_id"] = f"p{pair_id:04d}_{layer}"
                    for name, val in zip(OXIDES, comp):
                        row[name] = round(float(val), 4)
                    rows.append(row)

                pair_id += 1

    manifest = pd.DataFrame(rows)

    if skipped:
        print(f"\n  Skipped {len(skipped)} empty design cells:")
        for center, al in skipped:
            print(f"    MgO ~{center:.1f}, {al}")
        if not allow_gaps:
            raise SystemExit(
                "\nERROR: the design has gaps. The response curve will have "
                "holes in it.\nEither point --sheet at a broader compilation, "
                "or pass --allow-gaps to proceed anyway."
            )

    # Column order: metadata first, oxides last -- the Julia driver reads the
    # oxide columns by name, so order is cosmetic, but it makes the manifest
    # readable.
    meta_cols = [c for c in manifest.columns if c not in OXIDES]
    manifest = manifest[["run_id"] + [c for c in meta_cols if c != "run_id"] + OXIDES]

    return manifest, cov


# =============================================================================
# 6. PLOTS
# =============================================================================

def plot_design(df, manifest, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    up = manifest[manifest["layer"] == "upper"]
    lo = manifest[manifest["layer"] == "lower"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    # (a) compilation + sampled upper crust in MgO-Al2O3
    ax = axes[0, 0]
    for al, c in zip(AL_TYPES, ["tab:red", "tab:blue"]):
        s = df[df["al_type"] == al]
        ax.scatter(s["MgO_pct"], s["Al2O3_pct"], s=12, alpha=0.35,
                   color=c, label=f"{al} (data)")
        s2 = up[up["al_type"] == al]
        ax.scatter(s2["MgO_pct"], s2["Al2O3_pct"], s=28, marker="x",
                   color=c, label=f"{al} (sampled)")
    for e in MGO_EDGES:
        ax.axvline(e, color="0.85", lw=0.6, zorder=0)
    ax.set_xlabel("MgO (wt%)"); ax.set_ylabel("Al$_2$O$_3$ (wt%)")
    ax.set_title("(a) Upper crust design coverage")
    ax.legend(fontsize=7)

    # (b) Al2O3/TiO2 vs MgO
    ax = axes[0, 1]
    ax.scatter(df["MgO_pct"], df["al_ti_ratio"], s=12, alpha=0.35, color="0.4")
    ax.axhline(AL_TI_SPLIT, color="k", ls="--", lw=1,
               label=f"split = {AL_TI_SPLIT}")
    ax.set_xlabel("MgO (wt%)"); ax.set_ylabel("Al$_2$O$_3$/TiO$_2$")
    ax.set_title("(b) Al type assignment")
    ax.legend(fontsize=8)

    # (c) coupled upper vs lower MgO
    ax = axes[1, 0]
    ax.scatter(up["MgO_pct"].values, lo["MgO_pct"].values, s=24,
               c=up["mgo_bin_center"].values, cmap="viridis")
    lim = [0, max(lo["MgO_pct"].max(), up["MgO_pct"].max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlabel("upper crust MgO (wt%)")
    ax.set_ylabel("lower crust MgO (wt%)")
    ax.set_title("(c) Layer coupling")

    # (d) sampled compositions, all oxides
    ax = axes[1, 1]
    labels = [o.replace("_pct", "") for o in OXIDES]
    pos = np.arange(len(OXIDES))
    ax.boxplot([up[o].values for o in OXIDES], positions=pos - 0.18,
               widths=0.3, patch_artist=True,
               boxprops=dict(facecolor="tab:orange", alpha=0.6))
    ax.boxplot([lo[o].values for o in OXIDES], positions=pos + 0.18,
               widths=0.3, patch_artist=True,
               boxprops=dict(facecolor="tab:green", alpha=0.6))
    ax.set_xticks(pos); ax.set_xticklabels(labels, rotation=45, fontsize=8)
    ax.set_ylabel("wt%")
    ax.set_title("(d) Ensemble spread (orange = upper, green = lower)")

    fig.tight_layout()
    path = os.path.join(output_dir, "design_coverage.png")
    fig.savefig(path, dpi=160)
    print(f"  Saved: {path}")


# =============================================================================
# 7. MAIN
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Build the 2-D (MgO x Al2O3/TiO2) composition design "
                    "with coupled upper and lower crust."
    )
    p.add_argument("--data", required=True, help="Excel compilation path")
    p.add_argument("--sheet", required=True,
                   help="Sheet name. Must span MgO ~8-32 wt%%, i.e. the broad "
                        "compilation, NOT strict_komatiitic_basalt.")
    p.add_argument("--output", default="./design_outputs")
    p.add_argument("--mode", choices=["regression", "stratified"],
                   default="regression",
                   help="regression = ALR trend fit then sample (fills sparse "
                        "bins); stratified = resample real analyses (every "
                        "ensemble member is an observed rock)")
    p.add_argument("--replicates", type=int, default=N_REPLICATES_DEFAULT)
    p.add_argument("--cores", type=int, default=4)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--allow-gaps", action="store_true",
                   help="proceed even if some design cells are empty")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)

    df = load_and_screen(args.data, args.sheet)

    manifest, cov = build_design(
        df, args.mode, args.replicates, args.seed, args.cores,
        args.allow_gaps, args.output,
    )

    cov_path = os.path.join(args.output, "coverage_report.csv")
    cov.to_csv(cov_path, index=False)

    man_path = os.path.join(args.output, "composition_manifest.csv")
    manifest.to_csv(man_path, index=False)

    n_pairs = manifest["pair_id"].nunique()
    print("\n" + "=" * 62)
    print("DESIGN SUMMARY")
    print("=" * 62)
    print(f"  mode              : {args.mode}")
    print(f"  design cells      : {manifest.groupby(['mgo_bin_index','al_type']).ngroups}")
    print(f"  replicates / cell : {args.replicates}")
    print(f"  pairs             : {n_pairs}")
    print(f"  Perple_X runs     : {len(manifest)}  (upper + lower)")
    print(f"  MgO span (upper)  : {manifest[manifest.layer=='upper']['MgO_pct'].min():.1f}"
          f" - {manifest[manifest.layer=='upper']['MgO_pct'].max():.1f} wt%")
    print(f"  MgO span (lower)  : {manifest[manifest.layer=='lower']['MgO_pct'].min():.1f}"
          f" - {manifest[manifest.layer=='lower']['MgO_pct'].max():.1f} wt%")
    print(f"\n  Saved: {man_path}")
    print(f"  Saved: {cov_path}")

    if not args.no_plot:
        plot_design(df, manifest, args.output)

    print("\nNEXT: run the Perple_X driver over this manifest.")
    print("  julia --threads 8 run_perplex_manifest.jl \\")
    print(f"      --manifest {man_path}")
    print("\nNote: f (upper crust fraction) is deliberately NOT in this")
    print("manifest. Upper and lower crust are run separately and the WATER")
    print("is mixed afterwards in postprocess_response.py, because bound H2O")
    print("is not linear in bulk composition.")


if __name__ == "__main__":
    main()
