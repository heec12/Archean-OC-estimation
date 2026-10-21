"""
plot_prelim_figure.py
=====================
Preliminary-results figure for the proposal (v3 Perple_X outputs).

Shows WHERE the crust releases its water and HOW MUCH, without the two v3
metrics Palin objected to (the 2 GPa / 600 C node and the 50 % step).

  (a) bound H2O of the crustal column (f-mixed) vs slab-surface depth,
      one line per selected MgO bin (median of the replicates), with a
      16-84 % band on the lowest and highest bins
  (b) net release per depth interval (wt% per DZ_BIN km) for the same bins:
      release as pulses, read straight off (a)
  (c) H2O released between 1 and 5 GPa vs upper-crust MgO, split into the
      upper-crust and lower-crust contributions (f-weighted, so they sum to
      the column total); bin medians as lines, pairs as points,
      Al type as marker shape
  (d) optional (--panel-d): bound-H2O field of the most magnesian cumulate
      with its SlabGabbro path and the TTG window drawn as a box. The caption
      should show the window, not interpret the crossing (scope note).

All path handling and reductions are imported from postprocess_response.py,
so the numbers match scalars.csv.

Usage (repo root)
-----------------
    python h2o_lookup/plot_prelim_figure.py \
        --manifest design_outputs/composition_manifest.csv \
        --runs     h2o_runs/v3 \
        --path-dir design_outputs/pt_paths \
        --output   response_outputs \
        --f 0.35
    # options:  --bins 7,13,17      MgO bins drawn in (a) and (b)
    #           --al Al_undepleted  restrict (a)/(b) to one Al type ("all" pools)
    #           --panel-d           add the P-T field panel
    #           --dz 5              depth bin for (b), km

Writes prelim_figure_f<f>.svg (text kept as text, for Illustrator), .pdf, .png
and prelim_figure_f<f>_numbers_ab.csv / _numbers_c.csv with every value the
text quotes (release interval z10-z90, release by bin and layer).
"""

import argparse
import importlib.util
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.colors import BoundaryNorm
import numpy as np
import pandas as pd

# ---- typography: change these two together --------------------------------
FONT_FAMILY = "DejaVu Sans"
MATH_SET = "dejavusans"

# ---- colours ---------------------------------------------------------------
# One-hue ordinal ramp for MgO bins (validated: monotone L, visible gaps).
MGO_RAMP = ["#86b6ef", "#2a78d6", "#104281"]
# Layer contributions in (c): categorical slots 2 and 3 (CVD-validated pair).
C_UPPER = "#eb6834"
C_LOWER = "#1baf7a"
C_TOTAL = "#1f1f1e"
INK_2 = "#55554f"

# ---- geometry and settings -------------------------------------------------
TTG_WINDOW = dict(P=(1.0, 2.5), T_C=(700.0, 900.0))
DZ_BIN = 5.0            # km, depth bin for the release-rate panel
DEPTH_MAX = 200.0       # km, x-limit for (a) and (b)
BAND = (16, 84)         # percentiles for the replicate band


def load_pp(path):
    spec = importlib.util.spec_from_file_location("postprocess_response", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def style():
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "mathtext.fontset": MATH_SET,
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "axes.linewidth": 0.6,
        "axes.edgecolor": INK_2,
        "axes.labelcolor": "#1f1f1e",
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.fontsize": 7,
        "svg.fonttype": "none",     # live text in Illustrator
        "pdf.fonttype": 42,
    })


def panel_label(ax, s):
    ax.text(-0.14, 1.03, s, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="bottom")


def release_per_bin(z, prof, dz, zmax):
    """Net loss of bound H2O in consecutive depth bins (positive = release)."""
    edges = np.arange(0.0, zmax + dz, dz)
    good = np.isfinite(prof)
    if good.sum() < 2:
        return edges, np.full(len(edges) - 1, np.nan)
    v = np.interp(edges, z[good], prof[good], left=np.nan, right=np.nan)
    return edges, v[:-1] - v[1:]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--path-dir", required=True)
    ap.add_argument("--output", default="./response_outputs")
    ap.add_argument("--f", type=float, default=0.35)
    ap.add_argument("--bins", default="7,13,17")
    ap.add_argument("--al", default="Al_undepleted",
                    help="Al type for (a)/(b): Al_undepleted, Al_depleted or all")
    ap.add_argument("--dz", type=float, default=DZ_BIN)
    ap.add_argument("--panel-d", action="store_true")
    ap.add_argument("--pp", default=os.path.join(here, "postprocess_response.py"))
    args = ap.parse_args()

    pp = load_pp(args.pp)
    style()
    os.makedirs(args.output, exist_ok=True)
    f = args.f
    bins = [float(b) for b in args.bins.split(",")]
    if len(bins) > len(MGO_RAMP):
        sys.exit(f"at most {len(MGO_RAMP)} bins: the ramp has {len(MGO_RAMP)} "
                 "validated steps (more need direct labels and a new ramp)")
    ramp = MGO_RAMP if len(bins) == 3 else MGO_RAMP[::2][:len(bins)]

    # ---- paths, aligned exactly as in postprocess_response ---------------
    pdir = args.path_dir
    surf = pp.surface_reference(pp.load_path(
        os.path.join(pdir, "pt_path_SlabSurface.csv"), "slab-surface path"))
    pu = pp.align_on_surface(pp.load_path(
        os.path.join(pdir, "pt_path_SlabMORB.csv"), "upper-crust path"),
        surf, "upper-crust path")
    pl = pp.align_on_surface(pp.load_path(
        os.path.join(pdir, "pt_path_SlabGabbro.csv"), "lower-crust path"),
        surf, "lower-crust path")
    s_ref, p_ref, z_ref = surf["x"], surf["P"], surf["depth"]
    win = pp.RELEASE_WINDOW_GPA

    # ---- per-pair profiles ------------------------------------------------
    manifest = pd.read_csv(args.manifest)
    runs = pp.load_runs(args.runs)
    pp.check_coverage(runs, manifest)
    up = manifest[manifest["layer"] == "upper"].set_index("pair_id")

    rows, profs = [], {}
    grids_lower = {}
    for pid, meta in up.iterrows():
        ru, rl = f"p{pid:04d}_upper", f"p{pid:04d}_lower"
        if ru not in runs or rl not in runs:
            continue
        Pu, Tu, Gu = pp.to_grid(runs[ru])
        Pl, Tl, Gl = pp.to_grid(runs[rl])
        prof_u = pp.profile_on(pu, Pu, Tu, Gu, s_ref)
        prof_l = pp.profile_on(pl, Pl, Tl, Gl, s_ref)
        prof = f * prof_u + (1 - f) * prof_l
        profs[pid] = prof
        grids_lower[pid] = (Pl, Tl, Gl)
        rel_u = pp.integrated_release(p_ref, prof_u)
        rel_l = pp.integrated_release(p_ref, prof_l)
        rows.append(dict(
            pair_id=pid, MgO_upper=meta["MgO_pct"],
            MgO_lower=float(manifest.loc[manifest["run_id"] == rl, "MgO_pct"].iloc[0]),
            mgo_bin_center=meta["mgo_bin_center"], al_type=meta["al_type"],
            release_upper=f * rel_u, release_lower=(1 - f) * rel_l,
            release_total=pp.integrated_release(p_ref, prof),
            path_mean=float(np.nanmean(prof))))
    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit("no complete upper/lower pairs found")

    # ---- figure layout ----------------------------------------------------
    ncol = 4 if args.panel_d else 3
    fig, axes = plt.subplots(1, ncol, figsize=(3.0 * ncol + 0.6, 3.1),
                             gridspec_kw=dict(wspace=0.38))
    ax_a, ax_b, ax_c = axes[:3]

    sel_al = (df["al_type"] == args.al) if args.al != "all" else np.ones(len(df), bool)
    numbers = []

    # ---- (a) and (b) --------------------------------------------------------
    for b, col in zip(bins, ramp):
        cell = df[sel_al & np.isclose(df["mgo_bin_center"], b)]
        if cell.empty:
            print(f"  bin {b:g}: no pairs for {args.al}")
            continue
        stack = np.vstack([profs[p] for p in cell["pair_id"]])
        import warnings
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(stack, axis=0)
            lo, hi = np.nanpercentile(stack, BAND, axis=0)
        lab = f"{b - 1:g}–{b + 1:g} wt% MgO"
        if b in (bins[0], bins[-1]):
            ax_a.fill_between(z_ref, lo, hi, color=col, alpha=0.18, lw=0)
        ax_a.plot(z_ref, med, color=col, lw=1.8, label=lab)

        edges, rel = release_per_bin(z_ref, med, args.dz, DEPTH_MAX)
        rel = np.clip(rel, 0, None)
        ax_b.stairs(rel, edges, color=col, lw=1.6, label=lab)

        # depth interval holding the central 80 % of the crust's 1-5 GPa loss
        inwin = (p_ref >= win[0]) & (p_ref <= win[1]) & np.isfinite(med)
        if inwin.sum() > 2:
            zz, vv = z_ref[inwin], med[inwin]
            lost = vv[0] - vv
            tot = lost[-1]
            z10 = np.interp(0.1 * tot, lost, zz) if tot > 0 else np.nan
            z90 = np.interp(0.9 * tot, lost, zz) if tot > 0 else np.nan
        else:
            z10 = z90 = np.nan
        numbers.append(dict(panel="a/b", bin=b, al=args.al, n=len(cell),
                            shallow_median=np.nanmax(med[:max(3, len(med) // 5)]),
                            release_1_5GPa_median=float(np.nanmedian(cell["release_total"])),
                            z10_km=z10, z90_km=z90))

    ax_a.set_xlim(0, DEPTH_MAX)
    ax_a.set_ylim(bottom=0)
    ax_a.set_xlabel("slab-surface depth (km)")
    ax_a.set_ylabel("bound H$_2$O in crust (wt%)")
    ax_a.set_ylim(top=ax_a.get_ylim()[1] * 1.3)
    ax_a.legend(frameon=False, loc="upper center", ncol=1, handlelength=1.6,
                bbox_to_anchor=(0.5, 1.02), fontsize=6.5)
    ax_b.set_xlim(0, DEPTH_MAX)
    ax_b.set_ylim(0, ax_b.get_ylim()[1] * 1.15)
    ax_b.set_xlabel("slab-surface depth (km)")
    ax_b.set_ylabel(f"H$_2$O released per {args.dz:g} km (wt%)")

    # shade the 1-5 GPa window on the depth axes
    inw = (p_ref >= win[0]) & (p_ref <= win[1])
    if inw.any():
        for ax in (ax_a, ax_b):
            ax.axvspan(z_ref[inw].min(), z_ref[inw].max(), color="0.5",
                       alpha=0.08, lw=0, zorder=0)
        ax_b.text(z_ref[inw].mean(), 0.97, f"{win[0]:g}–{win[1]:g} GPa",
                  transform=ax_b.get_xaxis_transform(), ha="center", va="top",
                  fontsize=7, color=INK_2)

    # ---- (c) ----------------------------------------------------------------
    markers = {"Al_depleted": "o", "Al_undepleted": "s"}
    for col_name, c, lab in (("release_total", C_TOTAL, "whole crust"),
                             ("release_lower", C_LOWER, "lower crust"),
                             ("release_upper", C_UPPER, "upper crust")):
        for al, mk in markers.items():
            s = df[df["al_type"] == al]
            ax_c.scatter(s["MgO_upper"], s[col_name], s=9, marker=mk,
                         color=c, alpha=0.35, lw=0)
        g = df.groupby("mgo_bin_center")[col_name].median()
        ax_c.plot(g.index, g.values, color=c, lw=1.8)
        ax_c.text(df["MgO_upper"].max() + 0.6, g.values[-1], lab, color="#1f1f1e",
                  fontsize=7, va="center")
        for b, v in g.items():
            numbers.append(dict(panel="c", quantity=col_name, bin=b,
                                median_all_al=v))
    ax_c.set_xlim(df["MgO_upper"].min() - 0.5, df["MgO_upper"].max() + 3.2)
    ax_c.set_ylim(bottom=0)
    ax_c.set_xlabel("upper-crust MgO (wt%)")
    ax_c.set_ylabel(f"H$_2$O released, {win[0]:g}–{win[1]:g} GPa (wt%)")
    ax_c.legend(handles=[Line2D([], [], marker=m, ls="", color=INK_2, ms=4,
                                label=a.replace("_", "-"))
                         for a, m in markers.items()],
                frameon=False, loc="upper left")

    # ---- (d) optional ---------------------------------------------------------
    if args.panel_d:
        ax_d = axes[3]
        hi_pid = df.loc[df["MgO_lower"].idxmax(), "pair_id"]
        Pl, Tl, Gl = grids_lower[hi_pid]
        levels = [0, 0.5, 1, 2, 3, 4, 6, 8, 10, 12, 15]
        cmap = plt.get_cmap("Blues").copy()
        cmap.set_bad("#e5e5e5")
        m = ax_d.pcolormesh(Tl - 273.15, Pl, np.ma.masked_invalid(Gl),
                            cmap=cmap, norm=BoundaryNorm(levels, cmap.N),
                            shading="nearest", rasterized=True)
        ax_d.plot(pl["T"] - 273.15, pl["P"], color=C_UPPER, lw=1.4)
        r = Rectangle((TTG_WINDOW["T_C"][0], TTG_WINDOW["P"][0]),
                      np.ptp(TTG_WINDOW["T_C"]), np.ptp(TTG_WINDOW["P"]),
                      fill=False, ec="#1f1f1e", lw=1.2, ls="--")
        ax_d.add_patch(r)
        ax_d.text(TTG_WINDOW["T_C"][1], TTG_WINDOW["P"][0] - 0.1, "TTG window",
                  ha="right", va="top", fontsize=7)
        ax_d.set_xlim(0, 1300)
        ax_d.set_ylim(0, 6)
        ax_d.set_xlabel("T (°C)")
        ax_d.set_ylabel("P (GPa)")
        cb = fig.colorbar(m, ax=ax_d, fraction=0.05, pad=0.03)
        cb.set_label("bound H$_2$O (wt%)")
        mg = df.loc[df["pair_id"] == hi_pid, "MgO_lower"].iloc[0]
        ax_d.set_title(f"cumulate, {mg:.1f} wt% MgO", fontsize=8)

    for ax, s in zip(axes, "abcd"):
        panel_label(ax, s)
        ax.grid(alpha=0.2, lw=0.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    base = os.path.join(args.output, f"prelim_figure_f{f:.2f}")
    for ext in ("svg", "pdf", "png"):
        fig.savefig(f"{base}.{ext}", dpi=300, bbox_inches="tight")
    nums = pd.DataFrame(numbers)
    ab = nums[nums["panel"] == "a/b"].dropna(axis=1, how="all")
    c = (nums[nums["panel"] == "c"]
         .pivot(index="bin", columns="quantity", values="median_all_al"))
    ab.round(3).to_csv(f"{base}_numbers_ab.csv", index=False)
    c.round(3).to_csv(f"{base}_numbers_c.csv")
    print(f"\nSaved: {base}.svg/.pdf/.png and {base}_numbers_ab.csv / _c.csv")
    print("\n(a)/(b): per selected bin")
    print(ab.round(2).to_string(index=False))
    print("\n(c): median 1-5 GPa release by MgO bin, both Al types")
    print(c.round(2).to_string())


if __name__ == "__main__":
    main()

