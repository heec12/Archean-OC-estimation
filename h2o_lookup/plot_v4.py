"""
plot_v4.py
==========
Figures comparing the v4 scenario ladder, from the outputs of postprocess_v4.py.

    v4a  MAGEMin `all`, v3-like phase list, ds62, NCFMAST   (closest to v3)
    v4b  `mb` (Green et al. 2016) for both layers, + Fe3+ + K
    v4c  `mb` upper, `ume` lower, + Fe3+ + K

Figures (PNG + PDF in --output):

  fig1_initial_bound_vs_mgo   Initial bound H2O (capacity at the start of the
                              path) vs layer MgO. One panel per layer; colour =
                              scenario; open/filled marker = Al-depleted /
                              Al-undepleted; line = MgO-bin median.
  fig2_release_by_mineral     Water released along the path, split by the
                              mineral that released it (mean over replicates).
                              Rows = scenario, columns = layer; per MgO bin,
                              left bar = Al-depleted, right bar = Al-undepleted.
  fig3_release_depth          Where the water leaves: depth by which 50 % of
                              each layer's total release is done, vs MgO, with
                              the 10-90 % depth range as a bar.
  fig4_crust_profiles         Mixed-crust bound H2O vs depth (f = upper-crust
                              fraction), for three MgO bins, per scenario and
                              Al type.

Also writes summary_by_bin.csv (bin means per scenario x layer x Al type).

Usage
-----
    python h2o_lookup/plot_v4.py --response response_v4 --output figures_v4 --f 0.3
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# -----------------------------------------------------------------------------
# Style (validated categorical palette, fixed order; recessive axes)
# -----------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"

SCEN_ORDER = ["v4a", "v4b", "v4c"]
SCEN_COLOR = {"v4a": "#2a78d6", "v4b": "#eb6834", "v4c": "#1baf7a"}
SCEN_LABEL = {
    "v4a": "v4a  all, v3-like (ds62, NCFMAST)",
    "v4b": "v4b  mb both layers",
    "v4c": "v4c  mb upper / ume lower",
}

# Colour follows the mineral, never its rank. Anything else -> "other".
PHASE_COLOR = {
    "atg": ("antigorite", "#2a78d6"),
    "chl": ("chlorite",   "#eb6834"),
    "law": ("lawsonite",  "#1baf7a"),
    "amp": ("amphibole",  "#eda100"),
    "br":  ("brucite",    "#e87ba4"),
    "ta":  ("talc",       "#008300"),
    "mu":  ("phengite",   "#6250d6"),
    "ep":  ("epidote/zoisite", "#e34948"),
    "zo":  ("epidote/zoisite", "#e34948"),
}
OTHER = ("other", "#a8a7a2")
PHASE_STACK = ["atg", "br", "ta", "chl", "law", "ep", "amp", "mu", "other"]

# Ordinal ramp for three MgO bins (validated: monotone, visible steps)
BIN_RAMP = ["#86b6ef", "#2a78d6", "#104281"]

AL_ORDER = ["Al_depleted", "Al_undepleted"]
AL_LABEL = {"Al_depleted": "Al-depleted", "Al_undepleted": "Al-undepleted"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "font.family": "sans-serif", "lines.linewidth": 1.5,
    "axes.axisbelow": True,
})


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.png / .pdf")


def phase_key(p):
    return p if p in PHASE_COLOR else "other"


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------
def fig1(summ, out):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=False)
    for ax, layer in zip(axes, ["upper", "lower"]):
        d = summ[summ.layer == layer]
        for scen in SCEN_ORDER:
            for al in AL_ORDER:
                s = d[(d.scenario == scen) & (d.al_type == al)]
                if s.empty:
                    continue
                filled = al == "Al_undepleted"
                ax.scatter(s.MgO, s.initial_bound_anh, s=22, linewidths=1.2,
                           edgecolors=SCEN_COLOR[scen],
                           facecolors=SCEN_COLOR[scen] if filled else SURFACE,
                           zorder=3)
                med = s.groupby("mgo_bin_center").agg(x=("MgO", "median"),
                                                      y=("initial_bound_anh", "median"))
                ax.plot(med.x, med.y, color=SCEN_COLOR[scen],
                        ls="-" if filled else "--", lw=1.5, zorder=2)
        ax.set_title(f"{layer} crust ({d.path.iloc[0] if len(d) else ''} path)", loc="left")
        ax.set_xlabel(f"{layer}-crust MgO (wt%)")
    axes[0].set_ylabel("initial bound H$_2$O (g / 100 g anhydrous)")
    handles = [Line2D([], [], color=SCEN_COLOR[s], lw=2, label=SCEN_LABEL[s]) for s in SCEN_ORDER]
    handles += [Line2D([], [], marker="o", ls="--", color=INK2, mfc=SURFACE, label="Al-depleted"),
                Line2D([], [], marker="o", ls="-", color=INK2, mfc=INK2, label="Al-undepleted")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.12))
    save(fig, out, "fig1_initial_bound_vs_mgo")


def fig2(rel, summ, out):
    bins = sorted(summ.mgo_bin_center.unique())
    fig, axes = plt.subplots(len(SCEN_ORDER), 2, figsize=(10, 2.6 * len(SCEN_ORDER)),
                             sharex=True, sharey="col")
    axes = np.atleast_2d(axes)
    used = set()
    width = 0.38
    for i, scen in enumerate(SCEN_ORDER):
        for j, layer in enumerate(["upper", "lower"]):
            ax = axes[i, j]
            s = summ[(summ.scenario == scen) & (summ.layer == layer)]
            if s.empty:
                ax.set_visible(False)
                continue
            path = s.path.iloc[0]
            r = rel[rel.run_id.isin(s.run_id) & (rel.path == path)].copy()
            r["pk"] = r.phase.map(phase_key)
            meta = s.drop_duplicates("run_id").set_index("run_id")[["mgo_bin_center", "al_type"]]
            r = r.join(meta, on="run_id", rsuffix="_s")
            # mean over replicates: sum per run, then average over runs in the cell
            per_run = r.groupby(["run_id", "pk"]).released_wt.sum().unstack(fill_value=0)
            per_run = per_run.join(meta)
            cell = per_run.groupby(["mgo_bin_center", "al_type"]).mean()
            for k, al in enumerate(AL_ORDER):
                xoff = (k - 0.5) * (width + 0.04)
                bottom = np.zeros(len(bins))
                for pk in PHASE_STACK:
                    if pk not in cell.columns:
                        continue
                    vals = np.array([cell[pk].get((b, al), np.nan) for b in bins])
                    vals = np.nan_to_num(np.clip(vals, 0, None))
                    if vals.sum() <= 1e-6:
                        continue
                    name, col = PHASE_COLOR.get(pk, OTHER) if pk != "other" else OTHER
                    used.add((name, col))
                    ax.bar(np.arange(len(bins)) + xoff, vals, width, bottom=bottom,
                           color=col, edgecolor=SURFACE, linewidth=1.0,
                           hatch="////" if al == "Al_undepleted" else None)
                    bottom += vals
            ax.set_title(f"{SCEN_LABEL[scen]}  -  {layer} crust ({path})", loc="left", fontsize=9)
            ax.grid(axis="x", visible=False)
            if j == 0:
                ax.set_ylabel("released H$_2$O\n(g / 100 g anhydrous)")
    for ax in axes[-1]:
        ax.set_xticks(range(len(bins)))
        ax.set_xticklabels([f"{b:g}" for b in bins])
        ax.set_xlabel("MgO bin (upper-crust MgO, wt%)")
    order = {n: i for i, n in enumerate(dict.fromkeys(PHASE_COLOR[p][0] if p in PHASE_COLOR else "other"
                                                       for p in PHASE_STACK))}
    handles = [Patch(color=c, label=n) for n, c in sorted(used, key=lambda x: order.get(x[0], 99))]
    handles += [Patch(facecolor=SURFACE, edgecolor=INK2, label="left: Al-depleted"),
                Patch(facecolor=SURFACE, edgecolor=INK2, hatch="////", label="right: Al-undepleted")]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, out, "fig2_release_by_mineral")


def release_depths(steps, summ):
    """Per run: depth by which 10/50/90 % of the path's total release is done."""
    rows = []
    for _, s in summ.iterrows():
        st = steps[(steps.run_id == s.run_id) & (steps.path == s.path)].sort_values("z_to")
        if st.empty:
            continue
        c = st.release.cumsum().values
        tot = c[-1]
        if tot <= 0:
            continue
        z = st.z_to.values
        q = [float(np.interp(p * tot, c, z)) for p in (0.1, 0.5, 0.9)]
        rows.append(dict(scenario=s.scenario, layer=s.layer, run_id=s.run_id, MgO=s.MgO, path=s.path,
                         mgo_bin_center=s.mgo_bin_center, al_type=s.al_type,
                         total_release=tot, z10=q[0], z50=q[1], z90=q[2]))
    return pd.DataFrame(rows)


def fig3(dep, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    offs = {s: (i - 1) * 0.26 for i, s in enumerate(SCEN_ORDER)}
    bins = sorted(dep.mgo_bin_center.unique())
    bidx = {b: k for k, b in enumerate(bins)}
    for ax, layer in zip(axes, ["upper", "lower"]):
        d = dep[dep.layer == layer]
        for scen in SCEN_ORDER:
            for al in AL_ORDER:
                s = d[(d.scenario == scen) & (d.al_type == al)]
                if s.empty:
                    continue
                g = s.groupby("mgo_bin_center")
                x = np.array([bidx[b] for b in g.groups]) + offs[scen] + (0.06 if al == "Al_undepleted" else -0.06)
                filled = al == "Al_undepleted"
                ax.vlines(x, g.z10.median(), g.z90.median(), color=SCEN_COLOR[scen],
                          lw=1.5, alpha=0.8)
                ax.scatter(x, g.z50.median(), s=26, linewidths=1.2,
                           edgecolors=SCEN_COLOR[scen],
                           facecolors=SCEN_COLOR[scen] if filled else SURFACE, zorder=3)
        ax.invert_yaxis()
        ax.set_xticks(range(len(bins)))
        ax.set_xticklabels([f"{b:g}" for b in bins])
        ax.grid(axis="x", visible=False)
        for k in range(1, len(bins)):
            ax.axvline(k - 0.5, color=GRID, lw=0.6)
        ax.set_title(f"{layer} crust ({d.path.iloc[0]} path)" if len(d) else f"{layer} crust", loc="left")
        ax.set_xlabel("MgO bin (upper-crust MgO, wt%)")
    axes[0].set_ylabel("depth (km): 50 % released (marker),\n10-90 % range (bar)")
    handles = [Line2D([], [], color=SCEN_COLOR[s], lw=2, label=SCEN_LABEL[s]) for s in SCEN_ORDER]
    handles += [Line2D([], [], marker="o", ls="", color=INK2, mfc=SURFACE, label="Al-depleted"),
                Line2D([], [], marker="o", ls="", color=INK2, mfc=INK2, label="Al-undepleted")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.12))
    save(fig, out, "fig3_release_depth")


def fig4(prof, f, bins, out):
    p = prof[np.isclose(prof.f, f)]
    if p.empty:
        print(f"  [skip] fig4: no crust profiles for f = {f}")
        return
    bins = [b for b in bins if b in set(p.mgo_bin_center)]
    fig, axes = plt.subplots(len(AL_ORDER), len(SCEN_ORDER), figsize=(10, 6.4),
                             sharex=True, sharey=True)
    for i, al in enumerate(AL_ORDER):
        for j, scen in enumerate(SCEN_ORDER):
            ax = axes[i, j]
            for b, col in zip(bins, BIN_RAMP):
                s = p[(p.scenario == scen) & (p.al_type == al) & (p.mgo_bin_center == b)]
                if s.empty:
                    continue
                m = s.groupby("depth_km").crust.median()
                ax.plot(m.values, m.index, color=col, lw=1.8, label=f"MgO bin {b:g}")
            if i == 0:
                ax.set_title(SCEN_LABEL[scen], loc="left", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{AL_LABEL[al]}\ndepth (km)")
            if i == len(AL_ORDER) - 1:
                ax.set_xlabel("crust bound H$_2$O (g / 100 g anhydrous)")
    axes[0, 0].invert_yaxis()
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(bins), frameon=False,
               bbox_to_anchor=(0.5, -0.03), title=f"median over replicates;  f (upper fraction) = {f:g}",
               title_fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, out, "fig4_crust_profiles")


def main():
    ap = argparse.ArgumentParser(description="v4 scenario-ladder figures")
    ap.add_argument("--response", required=True, help="output dir of postprocess_v4.py")
    ap.add_argument("--output", default="figures_v4")
    ap.add_argument("--f", type=float, default=0.3, help="upper-crust fraction for fig4")
    ap.add_argument("--profile-bins", type=float, nargs=3, default=[7, 11, 17],
                    help="three MgO bins shown in fig4")
    a = ap.parse_args()
    os.makedirs(a.output, exist_ok=True)

    rd = lambda n: pd.read_csv(os.path.join(a.response, n))
    summ = rd("pair_summary.csv")
    rel = rd("release_by_phase.csv")
    steps = rd("release_steps.csv")
    prof = rd("crust_profiles.csv")

    present = [s for s in SCEN_ORDER if s in set(summ.scenario)]
    print(f"scenarios: {', '.join(present)};  pairs: {summ.pair_id.nunique()}")

    fig1(summ, a.output)
    fig2(rel, summ, a.output)
    dep = release_depths(steps, summ)
    fig3(dep, a.output)
    fig4(prof, a.f, a.profile_bins, a.output)

    tab = (summ.merge(dep[["run_id", "total_release", "z10", "z50", "z90"]], on="run_id", how="left")
               .groupby(["scenario", "layer", "al_type", "mgo_bin_center"])
               [["MgO", "initial_bound_anh", "total_release", "z10", "z50", "z90"]]
               .mean().round(2).reset_index())
    tab.to_csv(os.path.join(a.output, "summary_by_bin.csv"), index=False)
    print("  wrote summary_by_bin.csv")


if __name__ == "__main__":
    main()
