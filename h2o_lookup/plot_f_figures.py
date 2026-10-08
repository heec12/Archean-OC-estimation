"""
plot_f_figures.py
=================
Figures for every upper-crust fraction f, not just the primary one.

postprocess_response.py computes scalars.csv for every f passed to --f-sweep,
but only draws response curves (and runs the full sensitivity set) at --f.
This script reads the existing scalars.csv and fills in the rest. No Perple_X
tables are touched, so it runs in seconds on a laptop.

For each f it writes
  response_curves_f{f}.png, response_*_f{f}.csv    (via postprocess_response)
  sensitivity_*_f{f}.csv, sensitivity_axes_*_f{f}.csv  (via postprocess_response)
  sensitivity_f{f}.png                              (new: oxide + design-axes panels)
and across all f
  response_curves_f_compare.png   (shared y-axes, one column per f)
  sensitivity_f_compare.png       (how each effect moves with f)

Usage (from the repo root)
--------------------------
    python h2o_lookup/plot_f_figures.py \
        --scalars response_outputs/scalars.csv \
        --output  response_outputs \
        --f       0.20,0.35,0.50

The regression and response-curve functions are imported from
postprocess_response.py (same directory by default, or --pp), so the numbers
and the cluster bootstrap are identical to the main pipeline.
"""

import argparse
import importlib.util
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCALARS = ["h2o_at_ref", "h2o_path_mean", "dehyd_depth_km", "release_wt"]
LABELS = {
    "h2o_at_ref":     "Bound H$_2$O at 2 GPa, 600 °C (wt%)",
    "h2o_path_mean":  "Path-mean bound H$_2$O (wt%)",
    "dehyd_depth_km": "Dehydration-step depth (km)",
    "release_wt":     "H$_2$O released, 1–5 GPa (wt%)",
}
# Scalars that depend on the slab-top path, still the placeholder
PLACEHOLDER_PATH = {"dehyd_depth_km", "release_wt"}

OXIDE_ORDER = ["MgO", "SiO2", "Al2O3", "CaO", "FeOtot", "TiO2", "Na2O"]
AXES_TERMS = ["MgO slope (per wt%)", "Al_undepleted offset", "MgO x Al_undepleted"]
AL_STYLE = {"Al_depleted":   dict(color="#c0392b", marker="o"),
            "Al_undepleted": dict(color="#2471a3", marker="s")}


def load_pp(path):
    spec = importlib.util.spec_from_file_location("postprocess_response", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ftag(f):
    return f"{f:.2f}"


SHOW_PLACEHOLDER = False   # set by --placeholder-path


def title_for(col):
    t = LABELS.get(col, col)
    flag = SHOW_PLACEHOLDER and col in PLACEHOLDER_PATH
    return t + ("\n(placeholder P–T path)" if flag else "")


def subset_f(scalars, f):
    return scalars[np.isclose(scalars["f"].astype(float), f)]


def binned_median(sub, col, xcol):
    """Median and 16-84 % band per MgO bin, per Al type."""
    if "mgo_bin_center" in sub.columns:
        key = sub["mgo_bin_center"]
    else:
        key = (np.floor(sub[xcol] / 2.0) * 2.0 + 1.0)
    g = sub.assign(_bin=key).groupby(["al_type", "_bin"])[col]
    return g.median(), g.quantile(0.16), g.quantile(0.84)


# ---------------------------------------------------------------------------
# Response-curve comparison across f
# ---------------------------------------------------------------------------
def plot_response_compare(scalars, f_values, output_dir, xcol="MgO_pct"):
    cols = [c for c in SCALARS if c in scalars.columns]
    fig, axs = plt.subplots(len(cols), len(f_values),
                            figsize=(3.6 * len(f_values), 2.9 * len(cols)),
                            sharex=True, sharey="row", squeeze=False)
    for i, col in enumerate(cols):
        for j, f in enumerate(f_values):
            ax = axs[i, j]
            sub = subset_f(scalars, f).dropna(subset=[col])
            med, lo, hi = binned_median(sub, col, xcol)
            for al, st in AL_STYLE.items():
                s = sub[sub["al_type"] == al]
                ax.scatter(s[xcol], s[col], s=12, alpha=0.35, color=st["color"],
                           marker=st["marker"], lw=0)
                if al in med.index.get_level_values(0):
                    m = med.loc[al]
                    ax.fill_between(m.index, lo.loc[al].values, hi.loc[al].values,
                                    color=st["color"], alpha=0.15, lw=0)
                    ax.plot(m.index, m.values, color=st["color"], lw=1.8,
                            label=al.replace("_", "-"))
            if i == 0:
                ax.set_title(f"f = {f:.2f}", fontsize=11)
            if j == 0:
                ax.set_ylabel(title_for(col), fontsize=8.5)
            if i == len(cols) - 1:
                ax.set_xlabel("Upper-crust MgO (wt%)")
            ax.grid(alpha=0.25)
    axs[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("Response curves vs upper-crust fraction f "
                 "(line = bin median, band = 16–84 %)", fontsize=11)
    fig.tight_layout()
    path = os.path.join(output_dir, "response_curves_f_compare.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Sensitivity figures
# ---------------------------------------------------------------------------
def read_sens(output_dir, col, f):
    p1 = os.path.join(output_dir, f"sensitivity_{col}_f{ftag(f)}.csv")
    p2 = os.path.join(output_dir, f"sensitivity_axes_{col}_f{ftag(f)}.csv")
    if not (os.path.exists(p1) and os.path.exists(p2)):
        return None, None
    ox = pd.read_csv(p1).set_index("oxide")
    ox = ox.reindex([o for o in OXIDE_ORDER if o in ox.index] +
                    [o for o in ox.index if o not in OXIDE_ORDER])
    ax = pd.read_csv(p2).set_index("term")
    return ox, ax


def _errbar_h(ax, y, val, lo, hi, color):
    ax.barh(y, val, color=color, alpha=0.8, height=0.6)
    ax.errorbar(val, y, xerr=[val - lo, hi - val], fmt="none",
                ecolor="k", elinewidth=1, capsize=2.5)
    ax.axvline(0, color="k", lw=0.7)


def plot_sensitivity_one_f(output_dir, f):
    cols = [c for c in SCALARS
            if read_sens(output_dir, c, f)[0] is not None]
    if not cols:
        print(f"  no sensitivity CSVs for f = {f:.2f}; skipped figure")
        return
    fig, axs = plt.subplots(2, len(cols), figsize=(3.7 * len(cols), 6.4),
                            squeeze=False)
    for j, col in enumerate(cols):
        ox, ta = read_sens(output_dir, col, f)
        # top: clr design effect per oxide
        a = axs[0, j]
        y = np.arange(len(ox))[::-1]
        v = ox["design_effect"].to_numpy()
        colors = ["#1e8449" if x > 0 else "#7d3c98" for x in v]
        _errbar_h(a, y, v, ox["ci_lo"].to_numpy(), ox["ci_hi"].to_numpy(), colors)
        a.set_yticks(y, ox.index)
        a.set_title(title_for(col), fontsize=9)
        a.set_xlabel("design effect (clr coef × spread)", fontsize=8)
        a.grid(axis="x", alpha=0.25)
        # bottom: design-axes model, intercept omitted (different scale)
        b = axs[1, j]
        terms = [t for t in AXES_TERMS if t in ta.index]
        yb = np.arange(len(terms))[::-1]
        tv = ta.loc[terms, "coef"].to_numpy()
        _errbar_h(b, yb, tv, ta.loc[terms, "ci_lo"].to_numpy(),
                  ta.loc[terms, "ci_hi"].to_numpy(), "#566573")
        b.set_yticks(yb, ["MgO slope\n(per wt%)", "Al-undepleted\noffset",
                          "MgO × Al-und."][:len(terms)])
        b.set_xlabel("coefficient (scalar units)", fontsize=8)
        b.grid(axis="x", alpha=0.25)
    axs[0, 0].set_ylabel("clr model: which oxide", fontsize=9)
    axs[1, 0].set_ylabel("design-axes model", fontsize=9)
    fig.suptitle(f"Sensitivity of bound-H$_2$O scalars, f = {f:.2f}  "
                 "(95 % bootstrap CI, resampled by source analysis)",
                 fontsize=11)
    fig.tight_layout()
    path = os.path.join(output_dir, f"sensitivity_f{ftag(f)}.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_sensitivity_compare(output_dir, f_values):
    cols = [c for c in SCALARS
            if all(read_sens(output_dir, c, f)[0] is not None for f in f_values)]
    if not cols:
        return
    fig, axs = plt.subplots(2, len(cols), figsize=(3.7 * len(cols), 6.4),
                            squeeze=False)
    fs = np.array(f_values)
    cmap = plt.get_cmap("tab10")
    for j, col in enumerate(cols):
        data = {f: read_sens(output_dir, col, f) for f in f_values}
        a = axs[0, j]
        for k, o in enumerate(data[f_values[0]][0].index):
            v = np.array([data[f][0].loc[o, "design_effect"] for f in f_values])
            lo = np.array([data[f][0].loc[o, "ci_lo"] for f in f_values])
            hi = np.array([data[f][0].loc[o, "ci_hi"] for f in f_values])
            a.errorbar(fs + (k - 3) * 0.004, v, yerr=[v - lo, hi - v],
                       marker="o", ms=4, capsize=2, lw=1.2,
                       color=cmap(k), label=o)
        a.axhline(0, color="k", lw=0.7)
        a.set_title(title_for(col), fontsize=9)
        a.set_ylabel("design effect" if j == 0 else "")
        a.grid(alpha=0.25)
        b = axs[1, j]
        for k, t in enumerate(AXES_TERMS):
            if t not in data[f_values[0]][1].index:
                continue
            v = np.array([data[f][1].loc[t, "coef"] for f in f_values])
            lo = np.array([data[f][1].loc[t, "ci_lo"] for f in f_values])
            hi = np.array([data[f][1].loc[t, "ci_hi"] for f in f_values])
            b.errorbar(fs + (k - 1) * 0.005, v, yerr=[v - lo, hi - v],
                       marker="s", ms=4, capsize=2, lw=1.2,
                       color=["#34495e", "#e67e22", "#95a5a6"][k],
                       label=t.replace("Al_undepleted", "Al-und."))
        b.axhline(0, color="k", lw=0.7)
        b.set_xlabel("upper-crust fraction f")
        b.set_ylabel("coefficient (scalar units)" if j == 0 else "")
        b.set_xticks(fs)
        a.set_xticks(fs)
        b.grid(alpha=0.25)
    axs[0, -1].legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1))
    axs[1, -1].legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.suptitle("How the sensitivity results move with f", fontsize=11)
    fig.tight_layout()
    path = os.path.join(output_dir, "sensitivity_f_compare.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--scalars", required=True, help="scalars.csv from postprocess")
    p.add_argument("--output", default="./response_outputs")
    p.add_argument("--f", default="0.20,0.35,0.50",
                   help="comma-separated f values to plot")
    p.add_argument("--pp", default=os.path.join(here, "postprocess_response.py"),
                   help="path to postprocess_response.py")
    p.add_argument("--placeholder-path", action="store_true",
                   help="label path-dependent panels as placeholder-path results")
    p.add_argument("--no-recompute", action="store_true",
                   help="only plot; reuse existing sensitivity/response CSVs")
    args = p.parse_args()
    global SHOW_PLACEHOLDER
    SHOW_PLACEHOLDER = args.placeholder_path

    os.makedirs(args.output, exist_ok=True)
    f_values = sorted({float(x) for x in args.f.split(",")})
    scalars = pd.read_csv(args.scalars)

    if "f" not in scalars.columns:
        sys.exit("scalars.csv has no 'f' column -- rerun postprocess_response.py "
                 "with --f-sweep first.")
    have = np.unique(scalars["f"].astype(float))
    missing = [f for f in f_values if not np.any(np.isclose(have, f))]
    if missing:
        sys.exit(f"f = {missing} not in scalars.csv (has {list(have)}). Rerun "
                 f"postprocess_response.py with --f-sweep "
                 f"{','.join(ftag(f) for f in f_values)}")
    print(f"scalars.csv: {len(scalars)} rows, f = {list(have)}")

    if not args.no_recompute:
        pp = load_pp(args.pp)
        for f in f_values:
            print(f"\n=== f = {f:.2f} ===")
            for col in SCALARS:
                pp.response_curve(scalars, col, f, args.output)
            pp.plot_response(scalars, f, args.output)
            for col in SCALARS:
                print(f"\n  [{col}]")
                pp.sensitivity_regression(scalars, col, f, args.output)

    print("\n=== Figures ===")
    for f in f_values:
        plot_sensitivity_one_f(args.output, f)
    plot_response_compare(scalars, f_values, args.output)
    plot_sensitivity_compare(args.output, f_values)


if __name__ == "__main__":
    main()
