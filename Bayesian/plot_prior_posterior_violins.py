"""
plot_prior_posterior_violins.py
================================
Visualise prior vs. posterior distributions for the two-stage Bayesian
Archean oceanic crust composition model (mass_balance_7oxides_v2.py).

Produces two figures:
  Fig 1 — Stage 1: prior vs. posterior of mu_UC (7 oxides)
  Fig 2 — Stage 2: prior vs. posterior of x_bulk for both scenarios
           (homogeneous_crust and layered_cumulate_lower_crust)

Usage
-----
    python plot_prior_posterior_violins.py \
        --trace_s1  bayesian_lower_crust_outputs/trace_stage1_uc.nc \
        --trace_hom bayesian_lower_crust_outputs/trace_homogeneous_crust.nc \
        --trace_lay bayesian_lower_crust_outputs/trace_layered_cumulate_lower_crust.nc \
        --data      komatiitic_basalt_subset.xlsx \
        --outdir    ./figures

Alternatively, run without --trace_* flags to generate synthetic
prior/posterior samples for layout testing (no real data needed):
    python plot_prior_posterior_violins.py --test

Requirements
------------
    pip install arviz numpy pandas matplotlib scipy openpyxl
"""

import argparse
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import gaussian_kde

# ---------------------------------------------------------------------------
# Shared constants (mirror mass_balance_7oxides_v2.py)
# ---------------------------------------------------------------------------
OXIDES = [
    "SiO2", "TiO2", "Al2O3", "FeO", "MgO", "CaO", "Na2O"
]

# Empirical prior centres for mu_UC (used to draw prior samples in Fig 1).
# These match the broad MvNormal prior in build_uc_model():
#   mu_UC ~ MvNormal(uc_mean_empirical, diag(3*uc_std_empirical)^2)
# We approximate uc_std_empirical from the literature-informed values below.
# Override by passing --data to load real sample statistics.
PRIOR_MU_UC = np.array([49.5, 0.65, 13.5, 10.5, 10.5, 10.2, 1.8])
PRIOR_STD_UC = np.array([2.0, 0.15, 1.5, 1.5, 2.0, 1.5, 0.4])

# delta priors (Stage 2) — straight from SCENARIO_PRIORS in the main script
SCENARIO_PRIORS = {
    "Homogeneous crust": {
        "delta_mu":    np.array([-0.3, -0.02, -0.2,  0.0,  0.8,  0.2, -0.1]),
        "delta_sigma": np.array([ 0.5,  0.05,  0.5,  0.5,  0.8,  0.5,  0.1]),
    },
    "Layered cumulate LC": {
        "delta_mu":    np.array([-0.8, -0.06, -0.8,  0.0,  2.0,  0.4, -0.3]),
        "delta_sigma": np.array([ 0.7,  0.07,  0.7,  0.6,  1.2,  0.7,  0.15]),
    },
}

N_PRIOR_SAMPLES = 4000   # synthetic prior draws

# ---------------------------------------------------------------------------
# Colour palette — clean, publication-friendly
# ---------------------------------------------------------------------------
C_PRIOR      = "#9E9E9E"   # neutral grey  — prior
C_POST_S1    = "#5C6BC0"   # indigo        — Stage 1 posterior (mu_UC)
C_POST_HOM   = "#26A69A"   # teal          — homogeneous crust x_bulk
C_POST_LAY   = "#EF8C2C"   # amber         — layered cumulate x_bulk
C_OBS        = "#E53935"   # red           — observed data median tick

ALPHA_VIOLIN = 0.80
ALPHA_PRIOR  = 0.45

# ---------------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def half_violin(ax, data, pos, width=0.35, color="#5C6BC0",
                alpha=0.8, side="right", zorder=3):
    """
    Draw a single half-violin (KDE) at x=pos.
    side='right'  → KDE extends to the right of pos
    side='left'   → KDE extends to the left  of pos
    Returns the KDE peak value for optional annotation.
    """
    kde = gaussian_kde(data, bw_method="scott")
    ymin, ymax = data.min(), data.max()
    y_grid = np.linspace(ymin - 0.5 * (ymax - ymin),
                         ymax + 0.5 * (ymax - ymin), 300)
    density = kde(y_grid)
    density = density / density.max() * width   # normalise to given width

    if side == "right":
        ax.fill_betweenx(y_grid, pos, pos + density,
                         color=color, alpha=alpha, zorder=zorder, linewidth=0)
        ax.plot(pos + density, y_grid,
                color=color, linewidth=0.8, alpha=min(alpha + 0.2, 1.0),
                zorder=zorder + 1)
    else:
        ax.fill_betweenx(y_grid, pos - density, pos,
                         color=color, alpha=alpha, zorder=zorder, linewidth=0)
        ax.plot(pos - density, y_grid,
                color=color, linewidth=0.8, alpha=min(alpha + 0.2, 1.0),
                zorder=zorder + 1)


def draw_median_iqr(ax, data, pos, color, side="right", width=0.35, zorder=5):
    """Overlay median line and IQR bar on a half-violin."""
    q25, med, q75 = np.percentile(data, [25, 50, 75])
    sign = 1 if side == "right" else -1
    ax.plot([pos, pos + sign * width * 0.6], [med, med],
            color=color, linewidth=1.5, zorder=zorder, solid_capstyle="round")
    ax.plot([pos + sign * width * 0.15, pos + sign * width * 0.15],
            [q25, q75],
            color=color, linewidth=4, alpha=0.5, zorder=zorder - 1,
            solid_capstyle="round")


def obs_tick(ax, value, pos, color=C_OBS, zorder=6):
    """Small horizontal tick marking an observed data statistic."""
    ax.plot([pos - 0.08, pos + 0.08], [value, value],
            color=color, linewidth=2.0, zorder=zorder, solid_capstyle="round")


# ---------------------------------------------------------------------------
# Figure 1 — Stage 1: prior vs. posterior of mu_UC
# ---------------------------------------------------------------------------

def fig_stage1(uc_posterior_samples, obs_medians, outdir):
    """
    uc_posterior_samples : dict {oxide_name: 1-D array of posterior draws}
    obs_medians          : dict {oxide_name: float} — observed sample medians
    """
    n = len(OXIDES)
    fig, axes = plt.subplots(1, n, figsize=(12, 4), sharey=False)
    fig.suptitle(
        "Stage 1 — Prior vs. posterior of $\\mu_{UC}$ (upper crust mean)",
        fontsize=11, y=1.02
    )

    for i, (ax, oxide) in enumerate(zip(axes, OXIDES)):
        # Prior samples: MvNormal prior centred on empirical mean, 3× std
        prior_samples = np.random.default_rng(42 + i).normal(
            PRIOR_MU_UC[i], PRIOR_STD_UC[i] * 3.0, N_PRIOR_SAMPLES
        )
        post_samples = uc_posterior_samples[oxide]

        # Left half = prior, right half = posterior (split violin)
        half_violin(ax, prior_samples,  pos=0, width=0.42,
                    color=C_PRIOR,   alpha=ALPHA_PRIOR, side="left",  zorder=2)
        half_violin(ax, post_samples,   pos=0, width=0.42,
                    color=C_POST_S1, alpha=ALPHA_VIOLIN, side="right", zorder=3)

        draw_median_iqr(ax, prior_samples,  0, C_PRIOR,   side="left",  width=0.42)
        draw_median_iqr(ax, post_samples,   0, C_POST_S1, side="right", width=0.42)

        # Observed median tick
        if oxide in obs_medians:
            obs_tick(ax, obs_medians[oxide], pos=0)

        # Axis formatting
        ax.axvline(0, color="0.7", linewidth=0.5, zorder=1)
        ax.set_xlim(-0.55, 0.55)
        ax.set_xticks([])
        ax.set_title(oxide, pad=4)
        if i == 0:
            ax.set_ylabel("Composition (wt%)")

        # Y range: cover both distributions
        all_data = np.concatenate([prior_samples, post_samples])
        lo, hi = np.percentile(all_data, [0.5, 99.5])
        pad = (hi - lo) * 0.15
        ax.set_ylim(lo - pad, hi + pad)

    # Legend
    legend_handles = [
        mpatches.Patch(color=C_PRIOR,   alpha=0.7, label="Prior"),
        mpatches.Patch(color=C_POST_S1, alpha=0.9, label="Posterior $\\mu_{UC}$"),
        mpl.lines.Line2D([0], [0], color=C_OBS, linewidth=2,
                         label="Observed median"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=3, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.06))

    fig.tight_layout()
    path = os.path.join(outdir, "fig1_stage1_mu_UC_prior_vs_posterior.pdf")
    fig.savefig(path)
    print(f"Saved: {path}")
    # Also save PNG for quick preview
    fig.savefig(path.replace(".pdf", ".png"))
    return fig


# ---------------------------------------------------------------------------
# Figure 2 — Stage 2: prior vs. posterior of x_bulk (both scenarios)
# ---------------------------------------------------------------------------

def fig_stage2(stage2_posterior, mu_UC_post_mean, outdir):
    """
    stage2_posterior : dict {scenario_label: {oxide: 1-D array of x_bulk draws}}
    mu_UC_post_mean  : 1-D array of length 7 — Stage 1 posterior mu_UC mean
                       (used as centre for x_bulk prior approximation)
    """
    scenario_labels = list(stage2_posterior.keys())
    colors = {scenario_labels[0]: C_POST_HOM, scenario_labels[1]: C_POST_LAY}

    n = len(OXIDES)
    fig, axes = plt.subplots(1, n, figsize=(12, 4), sharey=False)
    fig.suptitle(
        "Stage 2 — Prior vs. posterior of $x_{bulk}$ (crustal bulk composition)",
        fontsize=11, y=1.02
    )

    for i, (ax, oxide) in enumerate(zip(axes, OXIDES)):

        # x_bulk prior approximation:
        # x_bulk_raw = mu_UC + delta  (before renormalisation)
        # We approximate the marginal prior as:
        #   N(mu_UC_post_mean + delta_mu, sqrt(post_std^2 + delta_sigma^2))
        # This is a conservative prior envelope shown as a shared grey background.
        delta_mus    = [SCENARIO_PRIORS[s]["delta_mu"][i]    for s in SCENARIO_PRIORS]
        delta_sigmas = [SCENARIO_PRIORS[s]["delta_sigma"][i] for s in SCENARIO_PRIORS]
        prior_mu_i  = mu_UC_post_mean[i] + np.mean(delta_mus)
        prior_std_i = np.sqrt(PRIOR_STD_UC[i]**2 + np.mean(delta_sigmas)**2)
        prior_samples = np.random.default_rng(99 + i).normal(
            prior_mu_i, prior_std_i * 2.5, N_PRIOR_SAMPLES
        )

        # Draw shared prior (grey, full width centred)
        half_violin(ax, prior_samples, pos=0, width=0.42,
                    color=C_PRIOR, alpha=ALPHA_PRIOR * 0.8, side="left", zorder=2)
        half_violin(ax, prior_samples, pos=0, width=0.42,
                    color=C_PRIOR, alpha=ALPHA_PRIOR * 0.8, side="right", zorder=2)
        draw_median_iqr(ax, prior_samples, 0, C_PRIOR, side="left",  width=0.42)
        draw_median_iqr(ax, prior_samples, 0, C_PRIOR, side="right", width=0.42)

        # Draw scenario posteriors offset slightly left/right for readability
        offsets = {"left": -0.10, "right": 0.10}
        sides   = [("left", scenario_labels[0]), ("right", scenario_labels[1])]
        for side, label in sides:
            post = stage2_posterior[label][oxide]
            off  = offsets[side]
            half_violin(ax, post, pos=off, width=0.38,
                        color=colors[label], alpha=ALPHA_VIOLIN,
                        side=side, zorder=4)
            draw_median_iqr(ax, post, off, colors[label],
                            side=side, width=0.38, zorder=6)

        ax.axvline(0, color="0.7", linewidth=0.5, zorder=1)
        ax.set_xlim(-0.60, 0.60)
        ax.set_xticks([])
        ax.set_title(oxide, pad=4)
        if i == 0:
            ax.set_ylabel("Composition (wt%)")

        all_data = np.concatenate(
            [prior_samples] +
            [stage2_posterior[s][oxide] for s in scenario_labels]
        )
        lo, hi = np.percentile(all_data, [0.5, 99.5])
        pad = (hi - lo) * 0.15
        ax.set_ylim(lo - pad, hi + pad)

    # Legend
    legend_handles = [
        mpatches.Patch(color=C_PRIOR,    alpha=0.55, label="Prior envelope"),
        mpatches.Patch(color=C_POST_HOM, alpha=0.9,  label=f"Posterior — {scenario_labels[0]}"),
        mpatches.Patch(color=C_POST_LAY, alpha=0.9,  label=f"Posterior — {scenario_labels[1]}"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=3, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.06))

    fig.tight_layout()
    path = os.path.join(outdir, "fig2_stage2_xbulk_prior_vs_posterior.pdf")
    fig.savefig(path)
    print(f"Saved: {path}")
    fig.savefig(path.replace(".pdf", ".png"))
    return fig


# ---------------------------------------------------------------------------
# Synthetic data generator (--test mode, no real traces needed)
# ---------------------------------------------------------------------------

def make_synthetic_posteriors(rng):
    """
    Generate plausible synthetic posterior samples for layout testing.
    Posteriors are narrower than priors and shifted slightly from the prior
    centre, mimicking what real MCMC output would look like.
    """
    # Stage 1: mu_UC posterior — tighter than prior, shifted by data
    mu_UC_post_mean = PRIOR_MU_UC + np.array([-0.5, -0.05, 0.3, -0.2, 0.8, -0.1, 0.05])
    mu_UC_post_std  = PRIOR_STD_UC * 0.35   # posterior much tighter than prior

    uc_post = {
        oxide: rng.normal(mu_UC_post_mean[i], mu_UC_post_std[i], 4000)
        for i, oxide in enumerate(OXIDES)
    }
    obs_medians = {
        oxide: mu_UC_post_mean[i] + rng.normal(0, 0.1)
        for i, oxide in enumerate(OXIDES)
    }

    # Stage 2: x_bulk posteriors per scenario
    scenario_labels = list(SCENARIO_PRIORS.keys())
    stage2_post = {}
    for label, key in zip(["Homogeneous crust", "Layered cumulate LC"], scenario_labels):
        sp = SCENARIO_PRIORS[key]
        post_mean = mu_UC_post_mean + sp["delta_mu"]
        post_std  = sp["delta_sigma"] * 0.5   # posterior tighter than delta prior
        stage2_post[label] = {
            oxide: rng.normal(post_mean[i], post_std[i], 4000)
            for i, oxide in enumerate(OXIDES)
        }

    return uc_post, obs_medians, stage2_post, mu_UC_post_mean


# ---------------------------------------------------------------------------
# Real trace loader
# ---------------------------------------------------------------------------

def load_real_traces(trace_s1_path, trace_hom_path, trace_lay_path, data_path):
    """Load ArviZ InferenceData traces and extract posterior samples."""
    import arviz as az
    import pandas as pd

    print("Loading Stage 1 trace …")
    t1 = az.from_netcdf(trace_s1_path)
    # mu_UC shape: (chain, draw, oxide)
    mu_uc_arr = t1.posterior["mu_UC"].values.reshape(-1, 7)
    uc_post = {oxide: mu_uc_arr[:, i] for i, oxide in enumerate(OXIDES)}
    mu_UC_post_mean = mu_uc_arr.mean(axis=0)

    print("Loading observed data for median ticks …")
    df = pd.read_excel(data_path, sheet_name="strict_komatiitic_basalt")
    # Derive FeOtot_pct from Fe2O3T exactly as load_data() does
    if "FeOtot_pct" not in df.columns:
        if "Fe2O3T" in df.columns:
            df["FeOtot_pct"] = 0.8998 * df["Fe2O3T"]
        else:
            raise KeyError(
                "Neither 'FeOtot_pct' nor 'Fe2O3T' found in the Excel sheet. "
                "Check that the sheet name and column names are correct."
            )
    oxide_cols = ["SiO2_pct","TiO2_pct","Al2O3_pct","FeOtot_pct",
                  "MgO_pct","CaO_pct","Na2O_pct"]
    df = df.dropna(subset=oxide_cols)
    obs_medians = {
        oxide: float(np.median(df[col].values))
        for oxide, col in zip(OXIDES, oxide_cols)
    }

    print("Loading Stage 2 traces …")
    t_hom = az.from_netcdf(trace_hom_path)
    t_lay = az.from_netcdf(trace_lay_path)
    # x_bulk shape: (chain, draw, oxide)
    hom_arr = t_hom.posterior["x_bulk"].values.reshape(-1, 7)
    lay_arr = t_lay.posterior["x_bulk"].values.reshape(-1, 7)

    stage2_post = {
        "Homogeneous crust":    {oxide: hom_arr[:, i] for i, oxide in enumerate(OXIDES)},
        "Layered cumulate LC":  {oxide: lay_arr[:, i] for i, oxide in enumerate(OXIDES)},
    }

    return uc_post, obs_medians, stage2_post, mu_UC_post_mean


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot prior vs. posterior violin figures for the "
                    "Archean OC Bayesian model."
    )
    parser.add_argument("--trace_s1",  help="Path to trace_stage1_uc.nc")
    parser.add_argument("--trace_hom", help="Path to trace_homogeneous_crust.nc")
    parser.add_argument("--trace_lay", help="Path to trace_layered_cumulate_lower_crust.nc")
    parser.add_argument("--data",      help="Path to Excel data file (for observed medians)")
    parser.add_argument("--outdir",    default="./figures",
                        help="Output directory for figures (default: ./figures)")
    parser.add_argument("--test",      action="store_true",
                        help="Use synthetic data — no trace files needed")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(42)

    if args.test:
        print("Running in TEST mode — using synthetic posterior samples.")
        uc_post, obs_medians, stage2_post, mu_UC_post_mean = \
            make_synthetic_posteriors(rng)
    else:
        missing = [n for n, v in [
            ("--trace_s1",  args.trace_s1),
            ("--trace_hom", args.trace_hom),
            ("--trace_lay", args.trace_lay),
            ("--data",      args.data),
        ] if not v]
        if missing:
            print(f"ERROR: Missing required arguments: {', '.join(missing)}")
            print("       Run with --test to use synthetic data instead.")
            sys.exit(1)

        uc_post, obs_medians, stage2_post, mu_UC_post_mean = load_real_traces(
            args.trace_s1, args.trace_hom, args.trace_lay, args.data
        )

    fig1 = fig_stage1(uc_post, obs_medians, args.outdir)
    fig2 = fig_stage2(stage2_post, mu_UC_post_mean, args.outdir)

    plt.show()
    print(f"\nDone. Figures written to: {args.outdir}/")


if __name__ == "__main__":
    main()
