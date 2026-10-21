"""
plot_pt_maps_v4.py
==================
Water-budget figure for the v4 runs: bound-H2O fields in P-T, the slab paths
through them, and the water profile along the slab with the minerals that
release it.

One row per selected pair, three panels per row:
  (a) upper-crust bound-H2O field (P-T grid) with the upper-crust path
  (b) lower-crust (cumulate) bound-H2O field with the lower-crust path
  (c) along the slab: upper, lower and mixed crust (f) vs slab-surface depth,
      and below it, where each mineral releases its water (10-90 % range as a
      bar, 50 % as a dot, total released as a number).

On (a) and (b) each diamond marks where a mineral has released 50 % of its
water, drawn on that layer's own path. Hatching = water-undersaturated grid
cells (bound H2O there is censored, not a capacity); cross-hatching = melt
present. Wet solidi are drawn if given (or found in --path-dir).

All water amounts are g H2O per 100 g anhydrous rock. All depths are
slab-surface depth (same convention as postprocess_v4.py).

Needs runs made WITH the grid (the default; not --grid false).

Usage (repo root)
-----------------
    python h2o_lookup/plot_pt_maps_v4.py \
        --manifest design_outputs/composition_manifest_v4.csv \
        --runs h2o_runs/v4 --response response_v4 \
        --path-dir design_outputs/pt_paths \
        --scenario all --f 0.35 --output figures_v4
    # optional:  --pairs 8,88   (default: lowest- and highest-MgO bins,
    #                            Al-undepleted, replicate 0)
    #            --solidus-csv FILE --solidus-label TEXT   (repeatable)
"""

import argparse
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
PATH_COLOR = "#e34948"
UPPER_COLOR = "#2a78d6"
LOWER_COLOR = "#6250d6"

# Same fixed mineral colours as plot_v4.py
PHASE_COLOR = {
    "atg": ("antigorite", "#2a78d6"), "chl": ("chlorite", "#eb6834"),
    "law": ("lawsonite", "#1baf7a"), "amp": ("amphibole", "#eda100"),
    "br": ("brucite", "#e87ba4"), "ta": ("talc", "#008300"),
    "mu": ("phengite", "#6250d6"), "ep": ("epidote", "#e34948"),
    "zo": ("zoisite", "#e34948"),
}
OTHER = ("other", "#a8a7a2")

SCENARIOS_DEFAULT_LABEL = {
    "v4a": "v4a: all, v3-like (ds62, NCFMAST)",
    "v4b": "v4b: mb both layers",
    "v4c": "v4c: mb upper / ume lower",
}

LEVELS = [0, 0.5, 1, 2, 3, 4, 6, 8, 10, 12, 15]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlesize": 9, "axes.labelsize": 9, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 7.5, "font.family": "sans-serif",
    "hatch.color": "#8a8984", "hatch.linewidth": 0.6, "axes.axisbelow": True,
})


def load_pp():
    spec = importlib.util.spec_from_file_location(
        "postprocess_v4", os.path.join(HERE, "postprocess_v4.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def anh(h):
    h = np.asarray(h, float)
    return 100.0 * h / (100.0 - h)


def read_grid(runs, rid):
    fp = os.path.join(runs, f"pts_{rid}.csv")
    if not os.path.isfile(fp):
        sys.exit(f"missing run table {fp}")
    d = pd.read_csv(fp)
    g = d[d.kind == "grid"]
    if g.empty:
        sys.exit(f"{rid}: no grid points. Rerun the driver with the grid (no --grid false).")
    P = np.sort(g.P_GPa.unique())
    T = np.sort(g.T_K.unique())
    piv = lambda col: g.pivot_table(index="P_GPa", columns="T_K", values=col,
                                    aggfunc="first").reindex(index=P, columns=T).values
    H = anh(piv("h2o_solid"))
    sat = piv("saturated").astype(float) > 0.5
    melt = piv("melt_present").astype(float) > 0.5
    bad = piv("status").astype(float) >= 3
    G = np.where(sat & ~melt & ~bad, H, np.nan)
    return P, T, G, (~sat & ~melt), melt


def draw_field(ax, P, T, G, unsat, melt, cmap, norm):
    Tc = T - 273.15
    mesh = ax.pcolormesh(Tc, P, np.ma.masked_invalid(G), cmap=cmap, norm=norm,
                         shading="nearest", rasterized=True)
    if unsat.any():
        ax.contourf(Tc, P, unsat.astype(float), levels=[0.5, 1.5], colors="none",
                    hatches=["////"], zorder=2)
    if melt.any():
        ax.contourf(Tc, P, melt.astype(float), levels=[0.5, 1.5], colors="none",
                    hatches=["xxxx"], zorder=2)
    cs = ax.contour(Tc, P, np.nan_to_num(G, nan=-1), levels=LEVELS[1:-1],
                    colors=INK, linewidths=0.4, alpha=0.55)
    ax.clabel(cs, fmt="%g", fontsize=6)
    ax.set_xlim(Tc.min(), Tc.max())
    ax.set_ylim(P.min(), P.max())
    ax.set_xlabel("T (°C)")
    ax.set_ylabel("P (GPa)")
    return mesh


def pick_pairs(manifest, scen_runs, requested):
    up = manifest[(manifest.layer == "upper") & manifest.run_id.isin(scen_runs)]
    up = up.drop_duplicates("pair_id").set_index("pair_id")
    if requested:
        ids = [int(x) for x in requested.split(",")]
        missing = [i for i in ids if i not in up.index]
        if missing:
            sys.exit(f"pairs not in manifest for this scenario: {missing}")
        return up.loc[ids]
    und = up[up.al_type == "Al_undepleted"].sort_values("replicate")
    lo = und[und.mgo_bin_center == und.mgo_bin_center.min()].index[0]
    hi = und[und.mgo_bin_center == und.mgo_bin_center.max()].index[0]
    return up.loc[[lo, hi]]


def main():
    ap = argparse.ArgumentParser(description="v4 water-budget P-T maps")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--runs", required=True, help="per-run tables (with grid)")
    ap.add_argument("--response", required=True, help="output of postprocess_v4.py")
    ap.add_argument("--path-dir", default=os.path.join("design_outputs", "pt_paths"))
    ap.add_argument("--scenario", default="all", choices=["all", "v4a", "v4b", "v4c"],
                    help="one scenario, or 'all' for one figure per scenario (shared colour scale)")
    ap.add_argument("--f", type=float, default=0.35, help="upper-crust mass fraction")
    ap.add_argument("--pairs", default="", help="comma-separated pair ids, e.g. 8,88")
    ap.add_argument("--min-release", type=float, default=0.3,
                    help="minerals releasing less than this (g/100 g) are not marked")
    ap.add_argument("--solidus-csv", action="append", default=[])
    ap.add_argument("--solidus-label", action="append", default=[])
    ap.add_argument("--output", default="figures_v4")
    a = ap.parse_args()
    os.makedirs(a.output, exist_ok=True)

    pp = load_pp()
    maps = pp.build_depth_maps(a.path_dir)
    scens = ["v4a", "v4b", "v4c"] if a.scenario == "all" else [a.scenario]
    # first pass: common colour scale so the three figures can be compared
    vmax = max(make_figure(a, sc, pp, maps, None, dry=True) for sc in scens)
    for sc in scens:
        make_figure(a, sc, pp, maps, vmax)


def make_figure(a, scenario, pp, maps, vmax_common, dry=False):
    spec = pp.SCENARIOS[scenario]

    summ = pd.read_csv(os.path.join(a.response, "pair_summary.csv"))
    rel = pd.read_csv(os.path.join(a.response, "release_by_phase.csv"))
    prof = pd.read_csv(os.path.join(a.response, "crust_profiles.csv"))
    summ = summ[summ.scenario == scenario]
    if summ.empty:
        sys.exit(f"no {scenario} rows in {a.response}/pair_summary.csv")
    path_up = summ[summ.layer == "upper"].path.iloc[0]
    path_lo = summ[summ.layer == "lower"].path.iloc[0]

    manifest = pd.read_csv(a.manifest)
    pairs = pick_pairs(manifest, set(summ.run_id), a.pairs)

    # paths in P-T, and their slab-surface depth
    def load_path(name):
        d = pd.read_csv(os.path.join(a.path_dir, f"pt_path_{name}.csv")).sort_values("depth_km")
        own, zs = maps.get(name, (d.depth_km.values, d.depth_km.values))
        d["z_slab"] = np.interp(d.depth_km, own, zs)
        return d
    P_up, P_lo = load_path(path_up), load_path(path_lo)
    surf_name = pp._surface_name(path_lo) or "SlabSurface"
    P_surf = load_path(surf_name) if os.path.isfile(
        os.path.join(a.path_dir, f"pt_path_{surf_name}.csv")) else None

    # solidi: explicit, else the VS01 fig 6a curve if it is in the path dir
    sols = []
    files = list(a.solidus_csv)
    labels = list(a.solidus_label)
    vs = os.path.join(a.path_dir, "solidus_VS01_fig6a.csv")
    if not files and os.path.isfile(vs):
        files = [vs]
        labels = ["fluid-saturated basalt solidus (Vielzeuf & Schmidt 2001, fig. 6a)"]
    styles = [dict(color="#eda100", ls="--"), dict(color="#eb6834", ls="-."),
              dict(color="#52514e", ls=":")]
    for i, fp in enumerate(files):
        lab = labels[i] if i < len(labels) else os.path.splitext(os.path.basename(fp))[0]
        sols.append((pd.read_csv(fp).sort_values("P_GPa"), lab, styles[i % len(styles)]))

    # grids
    grids, vmax = {}, 0.0
    for pid in pairs.index:
        ru = f"p{pid:04d}_upper_{spec['upper']}"
        rl = f"p{pid:04d}_lower_{spec['lower']}"
        grids[pid] = (ru, read_grid(a.runs, ru), rl, read_grid(a.runs, rl))
        vmax = max(vmax, np.nanmax(grids[pid][1][2]), np.nanmax(grids[pid][3][2]))
    if dry:
        return vmax
    vmax = vmax_common or vmax
    levels = [l for l in LEVELS if l < vmax] + [float(np.ceil(vmax))]
    cmap = plt.get_cmap("Blues").copy()      # one hue, light -> dark
    cmap.set_bad("#efeeea")
    norm = BoundaryNorm(levels, cmap.N)

    n = len(pairs)
    fig = plt.figure(figsize=(16, 5.6 * n))
    gs = fig.add_gridspec(n, 5, width_ratios=[1, 1, 0.045, 0.2, 1.25],
                          wspace=0.28, hspace=0.45, top=0.93, bottom=0.06)
    mesh = None
    for r, pid in enumerate(pairs.index):
        meta = pairs.loc[pid]
        ru, (Pu, Tu, Gu, Uu, Mu), rl, (Pl, Tl, Gl, Ul, Ml) = grids[pid]
        head = (f"pair {pid}: upper MgO {meta.MgO:.1f} wt%, "
                f"{meta.al_type.replace('_', '-')}  ({SCENARIOS_DEFAULT_LABEL[scenario]})")
        rel_pid = rel[rel.run_id.isin([ru, rl])]

        for c, (lay, rid, P, T, G, U, M, path) in enumerate((
                ("upper crust", ru, Pu, Tu, Gu, Uu, Mu, P_up),
                ("lower crust (cumulate)", rl, Pl, Tl, Gl, Ul, Ml, P_lo))):
            ax = fig.add_subplot(gs[r, c])
            mesh = draw_field(ax, P, T, G, U, M, cmap, norm)
            if P_surf is not None:
                ax.plot(P_surf.T_K - 273.15, P_surf.P_GPa, color=INK2, ls=":", lw=1)
            ax.plot(path.T_K - 273.15, path.P_GPa, color=PATH_COLOR, lw=1.8)
            for sol, lab, st in sols:
                ax.plot(sol.T_K - 273.15, sol.P_GPa, lw=1.6, **st)
            # 50 % release point of each mineral, on this layer's path
            rr = rel_pid[(rel_pid.run_id == rid) &
                         (rel_pid.path == (path_up if c == 0 else path_lo)) &
                         (rel_pid.released_wt >= a.min_release)]
            for _, q in rr.iterrows():
                if not np.isfinite(q.z50_km):
                    continue
                Pm = np.interp(q.z50_km, path.z_slab, path.P_GPa)
                Tm = np.interp(q.z50_km, path.z_slab, path.T_K) - 273.15
                col = PHASE_COLOR.get(q.phase, OTHER)[1]
                ax.plot(Tm, Pm, "D", ms=7, mfc=col, mec=INK, mew=0.8, zorder=6)
            ax.set_title(f"{head if c == 0 else ''}\n{lay}  —  {(path_up if c == 0 else path_lo)} path",
                         fontsize=9, loc="left")

        # ---- along-slab profile + mineral release strip ----
        sub = gs[r, 4].subgridspec(2, 1, height_ratios=[2.2, 1], hspace=0.08)
        ax = fig.add_subplot(sub[0])
        axs = fig.add_subplot(sub[1], sharex=ax)
        p = prof[(prof.scenario == scenario) & (prof.pair_id == pid)]
        p = p[np.isclose(p.f, p.f.iloc[0])].sort_values("depth_km") if len(p) else p
        z = p.depth_km.values
        up_v, lo_v = p.upper.values, p.lower.values
        crust = a.f * up_v + (1 - a.f) * lo_v
        ax.plot(z, up_v, color=UPPER_COLOR, lw=1.5, label="upper crust")
        ax.plot(z, lo_v, color=LOWER_COLOR, lw=1.5, label="lower crust")
        ax.plot(z, crust, color=INK, lw=2.4, label=f"crust, f = {a.f:g}")
        if np.isfinite(crust).any():
            i0 = np.nanargmax(np.isfinite(crust))
            c0, c1 = crust[i0], crust[np.where(np.isfinite(crust))[0][-1]]
            ax.text(0.98, 0.95, f"crust: {c0:.1f} → {c1:.1f} g/100 g\n"
                                f"released: {c0 - c1:.1f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8)
        ax.set_ylabel("bound H$_2$O\n(g / 100 g anhydrous)")
        ax.set_title("along the slab", fontsize=9, loc="left")
        ax.grid(color=GRID, lw=0.6)
        ax.legend(frameon=False, loc="lower left")
        plt.setp(ax.get_xticklabels(), visible=False)

        # strip: one row per (layer, mineral)
        rows = []
        for lay, rid, path in (("upper", ru, path_up), ("lower", rl, path_lo)):
            q = rel_pid[(rel_pid.run_id == rid) & (rel_pid.path == path) &
                        (rel_pid.released_wt >= a.min_release)].sort_values("z50_km")
            for _, x in q.iterrows():
                rows.append((lay, x))
        for k, (lay, x) in enumerate(rows):
            name, col = PHASE_COLOR.get(x.phase, OTHER)
            y = len(rows) - 1 - k
            axs.plot([x.z10_km, x.z90_km], [y, y], color=col, lw=5, solid_capstyle="butt")
            axs.plot(x.z50_km, y, "o", ms=6, mfc=col, mec=INK, mew=0.8)
            axs.text(x.z90_km + 2, y, f"{lay}: {name}  {x.released_wt:.1f}",
                     va="center", fontsize=7, color=INK)
        axs.set_yticks([])
        axs.set_ylim(-0.8, max(len(rows), 1) - 0.2)
        axs.set_xlabel("slab-surface depth (km)")
        axs.set_ylabel("release\n10–90 %", fontsize=8)
        axs.grid(axis="x", color=GRID, lw=0.6)
        ax.set_xlim(0, max(P_up.z_slab.max(), P_lo.z_slab.max()) + 25)   # same axis in every figure

    # legend + colourbar
    h = [Line2D([], [], color=INK2, ls=":", lw=1, label="slab surface"),
         Line2D([], [], color=PATH_COLOR, lw=1.8, label="layer path"),
         Line2D([], [], marker="D", ls="", mfc="#d6d5d0", mec=INK,
                label="50 % of a mineral's water released (colour = mineral)")]
    for sol, lab, st in sols:
        h.append(Line2D([], [], lw=1.6, label=lab, **st))
    h += [Patch(facecolor=SURFACE, edgecolor="#8a8984", hatch="////",
                label="water-undersaturated (masked)"),
          Patch(facecolor=SURFACE, edgecolor="#8a8984", hatch="xxxx", label="melt present (masked)")]
    shown = [g[0] for g in grids.values()] + [g[2] for g in grids.values()]
    used = sorted({x for x in rel[rel.run_id.isin(shown) & (rel.released_wt >= a.min_release)].phase})
    h += [Patch(color=PHASE_COLOR.get(p, OTHER)[1], label=PHASE_COLOR.get(p, OTHER)[0])
          for p in dict.fromkeys(used) if p in PHASE_COLOR]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.02 - 0.01 * n))
    cax = fig.add_subplot(gs[:, 2])
    cb = fig.colorbar(mesh, cax=cax)
    cb.set_label("bound H$_2$O (g / 100 g anhydrous)")
    fig.suptitle("Water budget: bound-H$_2$O fields, slab paths, and release along the slab",
                 y=0.975, fontsize=11)
    out = os.path.join(a.output, f"pt_maps_{scenario}_f{a.f:.2f}")
    fig.savefig(out + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}.png / .pdf")
    return vmax


if __name__ == "__main__":
    main()
