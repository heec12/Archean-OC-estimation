"""
postprocess_response.py
=======================
Stage 3 of the Archean oceanic crust bound-H2O pipeline. New script -- there
was no equivalent before, because the old Julia driver did its reduction
inline and only ever emitted ensemble percentiles.

WHAT THIS DOES
--------------
1. Loads the per-run long-format tables written by run_perplex_manifest.jl.
2. Masks undersaturated cells. A cell with no free fluid is a censored
   observation, not a capacity -- including it would bias bound H2O downward
   exactly where the input H2O ran out.
3. Mixes the layers AFTER the phase equilibria:

       H2O_crust(P,T) = f * H2O_upper(P,T) + (1-f) * H2O_lower(P,T)

   Never before. Mass-balancing compositions first and running one pseudosection
   is the E[f(x)] != f(E[x]) error, because reaction boundaries move with
   composition. Because the mixing is now a post-processing step, f is free:
   sweep it, or set it per scenario, at no compute cost.
4. Collapses each (P,T) field to scalars, four ways:
       - bound H2O at a fixed reference P-T
       - bound H2O along a slab-top P-T path
       - depth of the main dehydration step along that path
       - integrated H2O released between two depths
   All four are computed and written; which one carries the paper can be
   decided later, since they are cheap once the tables exist.
5. Builds the RESPONSE CURVE: scalar vs MgO, split by Al2O3/TiO2 type. This is
   the deliverable. The ensemble is a design, not a probability distribution,
   so its mean is meaningless -- the curve and its residual scatter are the
   result.
6. Runs a sensitivity regression of the scalar on all seven oxides, so
   "which oxide controls the dehydration band" is answered quantitatively
   rather than asserted.
7. Exports final P-T lookup tables in the Wilson-style CR-only matrix format
   for TerraFERMA, for a chosen subset of the design.

USAGE
-----
    python postprocess_response.py \
        --manifest design_outputs/composition_manifest.csv \
        --runs h2o_runs/v3 \
        --output ./response_outputs \
        --f 0.35

    # sweep f
    python postprocess_response.py ... --f-sweep 0.20,0.30,0.40,0.50

    # emit a TerraFERMA table for one design cell
    python postprocess_response.py ... --export-lookup mgo=14,al_type=Al_depleted

Requirements: pip install pandas numpy matplotlib scikit-learn
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

OXIDES = [
    "SiO2_pct", "TiO2_pct", "Al2O3_pct",
    "FeOtot_pct", "MgO_pct", "CaO_pct", "Na2O_pct",
]

# --- Reduction settings ------------------------------------------------------
REF_P_GPA = 2.0     # reference point for the simplest scalar
REF_T_K   = 873.0   # 600 C

# Dehydration step: the depth at which bound H2O along the path falls below
# this fraction of its shallow value.
DEHYD_FRACTION = 0.5

# Integrated release window, GPa
RELEASE_WINDOW_GPA = (1.0, 5.0)

# Default upper-crust fraction if none given. Kept as a post-processing scalar
# precisely so it is easy to change.
F_DEFAULT = 0.35

# Rough conversion for plotting a depth axis. Crust/mantle mean density
# ~3300 kg/m3 -> 1 GPa ~ 30.9 km.
GPA_TO_KM = 30.9


# =============================================================================
# LOADING
# =============================================================================

def load_runs(runs_dir, verbose=True):
    """Load every per-run long table into a dict of run_id -> DataFrame."""
    paths = sorted(glob.glob(os.path.join(runs_dir, "h2o_*.csv")))
    if not paths:
        raise SystemExit(f"No h2o_*.csv files found in {runs_dir}")

    runs = {}
    for p in paths:
        df = pd.read_csv(p)
        run_id = df["run_id"].iloc[0]
        runs[run_id] = df
    if verbose:
        print(f"  loaded {len(runs)} run tables from {runs_dir}")
    return runs


def to_grid(df, value_col="h2o_solid", mask_unsaturated=True):
    """
    Long table -> (P values, T values, 2-D array).
    Undersaturated cells become NaN.
    """
    d = df.copy()
    if mask_unsaturated and "saturated" in d.columns:
        d.loc[~d["saturated"].astype(bool), value_col] = np.nan
    pivot = d.pivot_table(index="P_GPa", columns="T_K", values=value_col)
    return pivot.index.to_numpy(), pivot.columns.to_numpy(), pivot.to_numpy()


def check_coverage(runs, manifest, verbose=True):
    """Report which manifest rows have no output, and saturation health."""
    expected = set(manifest["run_id"])
    got = set(runs)
    missing = sorted(expected - got)

    unsat = {}
    for rid, df in runs.items():
        if "saturated" in df.columns:
            unsat[rid] = 1.0 - df["saturated"].astype(bool).mean()

    if verbose:
        print(f"  manifest rows: {len(expected)}, run tables: {len(got)}")
        if missing:
            print(f"  MISSING {len(missing)} runs (rerun the array job):")
            for m in missing[:15]:
                print(f"    {m}")
            if len(missing) > 15:
                print(f"    ... and {len(missing) - 15} more")
        worst = sorted(unsat.items(), key=lambda kv: -kv[1])[:5]
        if worst and worst[0][1] > 0.05:
            print("  Highest undersaturated fractions (raise H2O_EXCESS if large):")
            for rid, frac in worst:
                print(f"    {rid}: {frac:.1%}")
    return missing


# =============================================================================
# PT PATH
# =============================================================================

def default_slab_top_path(n=60):
    """
    Placeholder slab-top P-T path: a warm Archean-ish trajectory.

    TODO: replace with the actual slab-top path extracted from the TerraFERMA /
    Wilson et al. (2014) thermal model. Use --path-csv to supply one with
    columns P_GPa,T_K. The dehydration-depth scalar is only as meaningful as
    this path, so this default is for pipeline testing, not for the paper.
    """
    p = np.linspace(0.2, 7.0, n)
    t = 273.0 + 420.0 * p ** 0.55
    return p, t


def load_path(path_csv):
    if path_csv is None:
        return default_slab_top_path()
    d = pd.read_csv(path_csv)
    return d["P_GPa"].to_numpy(), d["T_K"].to_numpy()


def sample_along_path(P, T, grid, path_p, path_t):
    """Bilinear sample of a P-T grid along a path."""
    out = np.full(len(path_p), np.nan)
    for i, (pp, tt) in enumerate(zip(path_p, path_t)):
        if pp < P.min() or pp > P.max() or tt < T.min() or tt > T.max():
            continue
        j = np.clip(np.searchsorted(P, pp) - 1, 0, len(P) - 2)
        k = np.clip(np.searchsorted(T, tt) - 1, 0, len(T) - 2)
        wp = (pp - P[j]) / (P[j + 1] - P[j])
        wt = (tt - T[k]) / (T[k + 1] - T[k])
        block = grid[j:j + 2, k:k + 2]
        if np.all(np.isnan(block)):
            continue
        w = np.array([[(1 - wp) * (1 - wt), (1 - wp) * wt],
                      [wp * (1 - wt), wp * wt]])
        m = ~np.isnan(block)
        if m.any():
            out[i] = float((block[m] * w[m]).sum() / w[m].sum())
    return out


# =============================================================================
# SCALAR REDUCTIONS
# =============================================================================

def value_at(P, T, grid, p_target, t_target):
    j = int(np.argmin(np.abs(P - p_target)))
    k = int(np.argmin(np.abs(T - t_target)))
    return float(grid[j, k])


def dehydration_depth(path_p, profile, fraction=DEHYD_FRACTION):
    """
    Pressure at which bound H2O first falls below `fraction` of its shallow
    value. This is the scalar most directly tied to "where does the water go".
    """
    good = ~np.isnan(profile)
    if good.sum() < 3:
        return np.nan
    p = path_p[good]
    v = profile[good]
    shallow = np.nanmax(v[:max(3, len(v) // 5)])
    if not np.isfinite(shallow) or shallow <= 0:
        return np.nan
    below = np.where(v < fraction * shallow)[0]
    if len(below) == 0:
        return np.nan
    i = below[0]
    if i == 0:
        return float(p[0])
    # linear interpolation onto the threshold
    target = fraction * shallow
    p0, p1 = p[i - 1], p[i]
    v0, v1 = v[i - 1], v[i]
    if v1 == v0:
        return float(p1)
    return float(p0 + (target - v0) * (p1 - p0) / (v1 - v0))


def integrated_release(path_p, profile, window=RELEASE_WINDOW_GPA):
    """
    H2O lost between two pressures along the path: value at the shallow end
    minus value at the deep end. Positive means release.
    """
    good = ~np.isnan(profile)
    if good.sum() < 2:
        return np.nan
    p, v = path_p[good], profile[good]
    lo = np.interp(window[0], p, v, left=np.nan, right=np.nan)
    hi = np.interp(window[1], p, v, left=np.nan, right=np.nan)
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return np.nan
    return float(lo - hi)


# =============================================================================
# LAYER MIXING AND SCALAR TABLE
# =============================================================================

def build_scalars(runs, manifest, f_values, path_p, path_t, verbose=True):
    """
    For every pair and every f, mix the layer water fields and reduce to
    scalars. Returns a tidy DataFrame, one row per (pair_id, f).
    """
    meta_cols = [c for c in manifest.columns
                 if c not in OXIDES + ["run_id", "layer"]]
    pairs = manifest[manifest["layer"] == "upper"].set_index("pair_id")

    records = []
    n_incomplete = 0

    for pair_id, meta in pairs.iterrows():
        rid_u = f"p{pair_id:04d}_upper"
        rid_l = f"p{pair_id:04d}_lower"
        if rid_u not in runs or rid_l not in runs:
            n_incomplete += 1
            continue

        Pu, Tu, Gu = to_grid(runs[rid_u])
        Pl, Tl, Gl = to_grid(runs[rid_l])
        if Gu.shape != Gl.shape:
            n_incomplete += 1
            continue

        prof_u = sample_along_path(Pu, Tu, Gu, path_p, path_t)
        prof_l = sample_along_path(Pl, Tl, Gl, path_p, path_t)

        for f in f_values:
            G = f * Gu + (1.0 - f) * Gl
            prof = f * prof_u + (1.0 - f) * prof_l

            rec = {c: meta[c] for c in meta_cols if c in meta.index}
            rec.update({
                "pair_id": pair_id,
                "f": f,
                # the four reductions
                "h2o_at_ref": value_at(Pu, Tu, G, REF_P_GPA, REF_T_K),
                "h2o_path_mean": float(np.nanmean(prof)),
                "h2o_path_shallow": float(np.nanmax(prof[:max(3, len(prof) // 5)]))
                                     if np.any(~np.isnan(prof)) else np.nan,
                "dehyd_P_GPa": dehydration_depth(path_p, prof),
                "release_wt": integrated_release(path_p, prof),
                # layer-resolved, useful for attributing the signal
                "h2o_upper_at_ref": value_at(Pu, Tu, Gu, REF_P_GPA, REF_T_K),
                "h2o_lower_at_ref": value_at(Pl, Tl, Gl, REF_P_GPA, REF_T_K),
            })
            # carry the upper-crust oxides for the sensitivity regression
            for ox in OXIDES:
                rec[ox] = meta[ox]
            records.append(rec)

    df = pd.DataFrame(records)
    df["dehyd_depth_km"] = df["dehyd_P_GPa"] * GPA_TO_KM

    if verbose:
        print(f"  scalars built for {df['pair_id'].nunique()} pairs "
              f"x {len(f_values)} f values")
        if n_incomplete:
            print(f"  skipped {n_incomplete} incomplete pairs")
    return df


# =============================================================================
# RESPONSE CURVE AND SENSITIVITY
# =============================================================================

def response_curve(scalars, scalar_col, f_value, output_dir):
    """
    Binned response of `scalar_col` to MgO, split by Al type. Statistics are
    taken WITHIN a design cell only (across replicates) -- never across MgO,
    which is the axis of interest.
    """
    sub = scalars[np.isclose(scalars["f"], f_value)]
    g = (sub.groupby(["al_type", "mgo_bin_center"])[scalar_col]
            .agg(["count", "mean", "std", "min", "max",
                  lambda x: np.nanpercentile(x, 25),
                  lambda x: np.nanpercentile(x, 75)]))
    g.columns = ["n", "mean", "std", "min", "max", "p25", "p75"]
    g = g.reset_index()
    path = os.path.join(output_dir, f"response_{scalar_col}_f{f_value:.2f}.csv")
    g.to_csv(path, index=False)
    print(f"  Saved: {path}")
    return g


def sensitivity_regression(scalars, scalar_col, f_value, output_dir):
    """
    Standardised regression coefficients of the scalar on all seven oxides.

    Because MgO is sampled by design over a wide range, these coefficients
    answer "which oxide controls the dehydration band, and how strongly" --
    a sensitivity result, which is what the design was built to produce.
    Compositional data are collinear (they sum to 100), so read these as
    relative importance, not as independent causal effects.
    """
    sub = scalars[np.isclose(scalars["f"], f_value)].dropna(subset=[scalar_col])
    if len(sub) < 20:
        print(f"  too few rows ({len(sub)}) for a sensitivity regression")
        return None

    X = sub[OXIDES].to_numpy(dtype=float)
    y = sub[scalar_col].to_numpy(dtype=float)

    Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
    ys = (y - y.mean()) / (y.std() + 1e-12)

    beta, *_ = np.linalg.lstsq(np.column_stack([Xs, np.ones(len(Xs))]),
                               ys, rcond=None)
    coefs = beta[:len(OXIDES)]

    # simple partial correlation for comparison
    partial = [np.corrcoef(Xs[:, i], ys)[0, 1] for i in range(len(OXIDES))]

    out = pd.DataFrame({
        "oxide": [o.replace("_pct", "") for o in OXIDES],
        "std_beta": np.round(coefs, 4),
        "marginal_corr": np.round(partial, 4),
        "abs_beta": np.round(np.abs(coefs), 4),
    }).sort_values("abs_beta", ascending=False)

    path = os.path.join(output_dir, f"sensitivity_{scalar_col}_f{f_value:.2f}.csv")
    out.to_csv(path, index=False)
    print(f"  Saved: {path}")
    print(out.to_string(index=False))
    return out


def plot_response(scalars, f_value, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = scalars[np.isclose(scalars["f"], f_value)]
    panels = [
        ("h2o_at_ref", f"bound H$_2$O at {REF_P_GPA} GPa, {REF_T_K-273:.0f} $\\degree$C (wt%)"),
        ("h2o_path_mean", "path-mean bound H$_2$O (wt%)"),
        ("dehyd_depth_km", "dehydration depth (km)"),
        ("release_wt", f"H$_2$O released {RELEASE_WINDOW_GPA[0]}-{RELEASE_WINDOW_GPA[1]} GPa (wt%)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    colors = {"Al_depleted": "tab:red", "Al_undepleted": "tab:blue"}

    for ax, (col, label) in zip(axes.ravel(), panels):
        for al, c in colors.items():
            s = sub[sub["al_type"] == al]
            if s.empty:
                continue
            ax.scatter(s["MgO_pct"], s[col], s=18, alpha=0.45, color=c)
            g = (s.groupby("mgo_bin_center")[col]
                   .agg(["mean", "std"]).reset_index().dropna(subset=["mean"]))
            ax.errorbar(g["mgo_bin_center"], g["mean"], yerr=g["std"],
                        color=c, lw=1.8, marker="o", ms=5, capsize=3,
                        label=al.replace("_", "-"))
        ax.set_xlabel("upper crust MgO (wt%)")
        ax.set_ylabel(label, fontsize=9)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle(f"Bound H$_2$O response to bulk composition (f = {f_value:.2f})")
    fig.tight_layout()
    path = os.path.join(output_dir, f"response_curves_f{f_value:.2f}.png")
    fig.savefig(path, dpi=160)
    print(f"  Saved: {path}")


def plot_f_sensitivity(scalars, output_dir):
    """How much does the layer fraction matter relative to composition?"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if scalars["f"].nunique() < 2:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for f, g in scalars.groupby("f"):
        gg = (g.groupby("mgo_bin_center")["h2o_at_ref"]
                .mean().reset_index())
        ax.plot(gg["mgo_bin_center"], gg["h2o_at_ref"],
                marker="o", ms=4, label=f"f = {f:.2f}")
    ax.set_xlabel("upper crust MgO (wt%)")
    ax.set_ylabel(f"bound H$_2$O at {REF_P_GPA} GPa (wt%)")
    ax.set_title("Composition vs layer fraction")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(output_dir, "f_sensitivity.png")
    fig.savefig(path, dpi=160)
    print(f"  Saved: {path}")


# =============================================================================
# TERRAFERMA LOOKUP EXPORT
# =============================================================================

def write_lookup_table(P_gpa, T_k, grid, path):
    """
    Wilson-style lookup table: header row is an empty cell then T in K, first
    column is P in GPa, CR-only line endings. Matches the format the old
    pipeline emitted and that TerraFERMA reads.
    """
    lines = [","+ ",".join(str(round(float(t), 4)) for t in T_k)]
    for j, p in enumerate(P_gpa):
        vals = ["" if np.isnan(grid[j, k]) else str(round(float(grid[j, k]), 6))
                for k in range(len(T_k))]
        lines.append(str(round(float(p), 6)) + "," + ",".join(vals))
    with open(path, "w", newline="") as fh:
        fh.write("\r".join(lines))
    print(f"  Saved: {path} ({len(P_gpa)} P x {len(T_k)} T, CR-only)")


def export_lookup(runs, manifest, selector, f_value, output_dir):
    """
    Emit percentile lookup tables for a SUBSET of the design.

    Percentiles across a design subset are only meaningful when the subset is
    one design cell (or one Al type at fixed MgO). Averaging across MgO would
    destroy the very signal the design was built to expose, so a selector is
    required rather than defaulted.

    selector: "mgo=14,al_type=Al_depleted" or "al_type=Al_undepleted"
    """
    conds = dict(kv.split("=", 1) for kv in selector.split(",") if "=" in kv)
    pairs = manifest[manifest["layer"] == "upper"].copy()

    if "mgo" in conds:
        target = float(conds["mgo"])
        idx = (pairs["mgo_bin_center"] - target).abs().idxmin()
        center = pairs.loc[idx, "mgo_bin_center"]
        pairs = pairs[pairs["mgo_bin_center"] == center]
        tag = f"mgo{center:.0f}"
    else:
        tag = "allmgo"
    if "al_type" in conds:
        pairs = pairs[pairs["al_type"] == conds["al_type"]]
        tag += f"_{conds['al_type']}"

    if pairs.empty:
        print(f"  selector '{selector}' matched no pairs")
        return

    stacked, P_ref, T_ref = [], None, None
    for pid in pairs["pair_id"]:
        rid_u, rid_l = f"p{pid:04d}_upper", f"p{pid:04d}_lower"
        if rid_u not in runs or rid_l not in runs:
            continue
        Pu, Tu, Gu = to_grid(runs[rid_u])
        _, _, Gl = to_grid(runs[rid_l])
        if Gu.shape != Gl.shape:
            continue
        stacked.append(f_value * Gu + (1.0 - f_value) * Gl)
        P_ref, T_ref = Pu, Tu

    if not stacked:
        print(f"  selector '{selector}' matched no complete pairs")
        return

    cube = np.stack(stacked)
    print(f"  lookup subset '{tag}': {cube.shape[0]} pairs")

    with np.errstate(all="ignore"):
        for name, arr in [
            ("p05", np.nanpercentile(cube, 5, axis=0)),
            ("p50", np.nanpercentile(cube, 50, axis=0)),
            ("p95", np.nanpercentile(cube, 95, axis=0)),
            ("mean", np.nanmean(cube, axis=0)),
        ]:
            write_lookup_table(
                P_ref, T_ref, arr,
                os.path.join(output_dir,
                             f"h2o_bound_{name}_{tag}_f{f_value:.2f}.csv"))


# =============================================================================
# MAIN
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Mix layers, reduce to scalars, build response curves, "
                    "export TerraFERMA lookup tables.")
    p.add_argument("--manifest", required=True)
    p.add_argument("--runs", required=True,
                   help="directory of h2o_<run_id>.csv tables")
    p.add_argument("--output", default="./response_outputs")
    p.add_argument("--f", type=float, default=F_DEFAULT,
                   help="upper crust fraction for the primary figures")
    p.add_argument("--f-sweep", default="",
                   help="comma-separated f values, e.g. 0.20,0.35,0.50")
    p.add_argument("--path-csv", default=None,
                   help="slab-top P-T path with columns P_GPa,T_K")
    p.add_argument("--export-lookup", default="",
                   help="selector for TerraFERMA export, e.g. "
                        "'mgo=14,al_type=Al_depleted'")
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("--- Loading ---")
    manifest = pd.read_csv(args.manifest)
    runs = load_runs(args.runs)
    check_coverage(runs, manifest)

    path_p, path_t = load_path(args.path_csv)
    if args.path_csv is None:
        print("\n  NOTE: using the placeholder slab-top P-T path. Supply the "
              "\n  real path with --path-csv before trusting dehyd_depth_km.")

    f_values = [args.f]
    if args.f_sweep:
        f_values = sorted({float(x) for x in args.f_sweep.split(",")} | {args.f})

    print("\n--- Mixing layers and reducing ---")
    scalars = build_scalars(runs, manifest, f_values, path_p, path_t)
    sc_path = os.path.join(args.output, "scalars.csv")
    scalars.to_csv(sc_path, index=False)
    print(f"  Saved: {sc_path}")

    print("\n--- Response curves ---")
    for col in ["h2o_at_ref", "h2o_path_mean", "dehyd_depth_km", "release_wt"]:
        response_curve(scalars, col, args.f, args.output)

    print("\n--- Sensitivity: which oxide controls the signal ---")
    for col in ["h2o_at_ref", "dehyd_depth_km"]:
        print(f"\n  [{col}]")
        sensitivity_regression(scalars, col, args.f, args.output)

    print("\n--- Figures ---")
    plot_response(scalars, args.f, args.output)
    plot_f_sensitivity(scalars, args.output)

    if args.export_lookup:
        print("\n--- TerraFERMA lookup export ---")
        export_lookup(runs, manifest, args.export_lookup, args.f, args.output)

    print("\n" + "=" * 62)
    print("Done. The deliverable is the response curve, not a posterior mean:")
    print("the ensemble is an experimental design over MgO, so its mean")
    print("corresponds to no actual rock. If volumetric constraints on")
    print("komatiite abundance turn up later, importance-reweight scalars.csv")
    print("by bin -- no Perple_X rerun needed.")
    print("=" * 62)


if __name__ == "__main__":
    main()
