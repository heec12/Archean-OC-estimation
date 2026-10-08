"""
plot_pt_maps.py
===============
What the three path metrics actually measure, for two contrasting compositions.

One row per selected design pair, three panels per row:
  (a) upper-crust bound-H2O field, P-T, with the SlabMORB path
  (b) lower-crust (cumulate) bound-H2O field, with the SlabGabbro path
  (c) the same water sampled along the slab: upper, lower and the f-mixed crust
      against slab-surface depth, with the 50 % dehydration step and the
      1-5 GPa release window drawn exactly as postprocess_response.py defines them

On (a) and (b): the star is the 2 GPa / 600 C reference node, the thick part of
each path is the 1-5 GPa window (measured on the slab SURFACE pressure at the
same slab position), the diamond is where that layer sits when the crust as a
whole passes the 50 % dehydration step, and grey hatching is water-
undersaturated (masked) cells. An optional wet solidus is drawn if given.

The reductions are imported from postprocess_response.py, so the markers are
the same numbers that end up in scalars.csv.

Usage (repo root)
-----------------
    python h2o_lookup/plot_pt_maps.py \
        --manifest design_outputs/composition_manifest.csv \
        --runs     h2o_runs/v3 \
        --path-dir design_outputs/pt_paths \
        --output   response_outputs_realpath \
        --f 0.35
    # optional:  --pairs p0003,p0091     (default: lowest- and highest-MgO
    #                                      Al-undepleted pairs, replicate 0)
    #            --solidus-csv wet_basalt_solidus.csv   (columns P_GPa,T_K)
"""

import argparse
import importlib.util
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import numpy as np
import pandas as pd


def load_pp(path):
    spec = importlib.util.spec_from_file_location("postprocess_response", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pick_pairs(manifest, requested):
    up = manifest[manifest["layer"] == "upper"].copy()
    up["pair_key"] = up["run_id"].str.replace("_upper", "", regex=False)
    if requested:
        keys = [k.strip() for k in requested.split(",")]
        missing = [k for k in keys if k not in set(up["pair_key"])]
        if missing:
            sys.exit(f"pairs not in manifest: {missing}")
        return up.set_index("pair_key").loc[keys]
    und = up[up["al_type"].astype(str).str.contains("undepleted", case=False)]
    pool = und if len(und) else up
    if "replicate" in pool.columns:
        pool = pool.sort_values("replicate")
    lo = pool.loc[pool["mgo_bin_center"] == pool["mgo_bin_center"].min()].iloc[0]
    hi = pool.loc[pool["mgo_bin_center"] == pool["mgo_bin_center"].max()].iloc[0]
    return up.set_index("pair_key").loc[[lo["pair_key"], hi["pair_key"]]]


def k2c(t):
    return np.asarray(t) - 273.15


def draw_field(ax, P, T, G, levels, cmap, norm):
    Tc = k2c(T)
    m = ax.pcolormesh(Tc, P, np.ma.masked_invalid(G), cmap=cmap, norm=norm,
                      shading="nearest")
    # hatch undersaturated cells
    nan = np.isnan(G).astype(float)
    if nan.any():
        ax.contourf(Tc, P, nan, levels=[0.5, 1.5], colors="none",
                    hatches=["////"], zorder=2)
    cs = ax.contour(Tc, P, np.nan_to_num(G, nan=-1), levels=levels[1:-1],
                    colors="k", linewidths=0.4, alpha=0.6)
    ax.clabel(cs, fmt="%g", fontsize=6)
    ax.set_xlim(Tc.min(), Tc.max())
    ax.set_ylim(P.min(), P.max())
    return m


def window_mask(ref_p, window):
    return (ref_p >= window[0]) & (ref_p <= window[1])


def at_ref_position(layer_path, s):
    """P,T of a layer path at slab-surface arc length s."""
    if not np.isfinite(s):
        return np.nan, np.nan
    return (np.interp(s, layer_path["x"], layer_path["P"]),
            np.interp(s, layer_path["x"], layer_path["T"]))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--path-dir", required=True,
                    help="dir with pt_path_SlabSurface/MORB/Gabbro.csv")
    ap.add_argument("--output", default="./response_outputs")
    ap.add_argument("--f", type=float, default=0.35)
    ap.add_argument("--pairs", default="")
    ap.add_argument("--solidus-csv", action="append", default=[],
                    help="solidus curve, columns P_GPa,T_K; repeat for several")
    ap.add_argument("--solidus-label", action="append", default=[],
                    help="legend label for each --solidus-csv, in the same order")
    ap.add_argument("--pp", default=os.path.join(here, "postprocess_response.py"))
    args = ap.parse_args()
    pp = load_pp(args.pp)
    os.makedirs(args.output, exist_ok=True)

    # ---- paths, aligned exactly as in the main pipeline -------------------
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

    sols = []
    styles = [dict(color="darkorange", ls="--"), dict(color="firebrick", ls="-."),
              dict(color="saddlebrown", ls=":")]
    for i, fp in enumerate(args.solidus_csv):
        lab = (args.solidus_label[i] if i < len(args.solidus_label)
               else os.path.splitext(os.path.basename(fp))[0])
        sols.append((pd.read_csv(fp).sort_values("P_GPa"), lab,
                     styles[i % len(styles)]))

    manifest = pd.read_csv(args.manifest)
    pairs = pick_pairs(manifest, args.pairs)
    lower_ids = dict(zip(
        manifest.loc[manifest["layer"] == "lower", "run_id"]
        .str.replace("_lower", "", regex=False),
        manifest.loc[manifest["layer"] == "lower", "run_id"]))

    def read_run(rid):
        fp = os.path.join(args.runs, f"h2o_{rid}.csv")
        if not os.path.exists(fp):
            sys.exit(f"missing run table {fp}")
        return pp.to_grid(pd.read_csv(fp))

    grids = {}
    vmax = 0
    for key in pairs.index:
        gu = read_run(f"{key}_upper")
        gl = read_run(lower_ids[key])
        grids[key] = (gu, gl)
        vmax = max(vmax, np.nanmax(gu[2]), np.nanmax(gl[2]))

    levels = [0, 0.5, 1, 2, 3, 4, 6, 8, 10, 12, 15]
    levels = [l for l in levels if l < vmax] + [np.ceil(vmax)]
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#e5e5e5")
    norm = BoundaryNorm(levels, cmap.N)

    n = len(pairs)
    fig = plt.figure(figsize=(16, 5.0 * n))
    gs = fig.add_gridspec(n, 5, width_ratios=[1, 1, 0.045, 0.22, 1.2],
                          wspace=0.28, hspace=0.42)
    axs = np.empty((n, 3), dtype=object)
    for r in range(n):
        axs[r, 0] = fig.add_subplot(gs[r, 0])
        axs[r, 1] = fig.add_subplot(gs[r, 1])
        axs[r, 2] = fig.add_subplot(gs[r, 4])
    cax = fig.add_subplot(gs[:, 2])
    f = args.f
    summary = []

    # plain-language row labels, ranked by upper-crust MgO
    mgo = pairs["MgO_pct"].astype(float)
    mgo_label = {}
    for key in pairs.index:
        if len(pairs) > 1 and mgo[key] == mgo.min():
            mgo_label[key] = "Lower-MgO"
        elif len(pairs) > 1 and mgo[key] == mgo.max():
            mgo_label[key] = "Higher-MgO"
        else:
            mgo_label[key] = "Intermediate-MgO" if len(pairs) > 2 else "Archean"

    for r, key in enumerate(pairs.index):
        meta = pairs.loc[key]
        (Pu, Tu, Gu), (Pl, Tl, Gl) = grids[key]
        prof_u = pp.profile_on(pu, Pu, Tu, Gu, s_ref)
        prof_l = pp.profile_on(pl, Pl, Tl, Gl, s_ref)
        prof = f * prof_u + (1 - f) * prof_l

        dP = pp.dehydration_depth(p_ref, prof)
        good = np.isfinite(prof)
        s_step = (np.interp(dP, p_ref[good], s_ref[good])
                  if np.isfinite(dP) else np.nan)
        z_step = np.interp(dP, p_ref, z_ref) if np.isfinite(dP) else np.nan
        rel = pp.integrated_release(p_ref, prof)
        Gmix = f * Gu + (1 - f) * Gl
        ref_val = pp.value_at(Pu, Tu, Gmix, pp.REF_P_GPA, pp.REF_T_K)
        jn = np.argmin(np.abs(Pu - pp.REF_P_GPA))
        kn = np.argmin(np.abs(Tu - pp.REF_T_K))
        inwin = window_mask(p_ref, win)

        title = f"{mgo_label[key]} crust ({meta['MgO_pct']:.1f} wt% MgO)"
        for c, (lay, P, T, G, path) in enumerate(
                (("upper crust", Pu, Tu, Gu, pu),
                 ("lower crust (cumulate)", Pl, Tl, Gl, pl))):
            ax = axs[r, c]
            mesh = draw_field(ax, P, T, G, levels, cmap, norm)
            ax.plot(k2c(surf["T"]), surf["P"], color="0.35", ls=":", lw=1,
                    label="slab surface")
            ax.plot(k2c(path["T"]), path["P"], color="crimson", lw=1.3,
                    label="layer path")
            # release window on this layer's path, by matched slab position
            s_win = s_ref[inwin]
            if len(s_win):
                sel = (path["x"] >= s_win.min()) & (path["x"] <= s_win.max())
                ax.plot(k2c(path["T"][sel]), path["P"][sel], color="crimson",
                        lw=4, alpha=0.45, solid_capstyle="butt",
                        label=f"{win[0]:g}\u2013{win[1]:g} GPa window")
            ps, ts = at_ref_position(path, s_step)
            if np.isfinite(ps):
                ax.plot(k2c(ts), ps, "D", ms=8, mfc="gold", mec="k",
                        label="crust at 50 % step", zorder=6)
            ax.plot(k2c(T[kn]), P[jn], "*", ms=14, mfc="white", mec="k",
                    label="2 GPa / 600 \u00b0C node", zorder=6)
            for sol, lab, st in sols:
                ax.plot(k2c(sol["T_K"]), sol["P_GPa"], lw=1.8, label=lab, **st)
            ax.set_title(f"{title}\n{lay}", fontsize=9)
            ax.set_xlabel("T (\u00b0C)")
            ax.set_ylabel("P (GPa)")
        # ---- along-slab profile ----
        ax = axs[r, 2]
        ax.plot(z_ref, prof_u, color="#2471a3", lw=1.3, label="upper crust")
        ax.plot(z_ref, prof_l, color="#7d3c98", lw=1.3, label="lower crust")
        ax.plot(z_ref, prof, color="k", lw=2.2, label=f"crust, f = {f:.2f}")
        if good.sum() >= 3:
            v = prof[good]
            shallow = np.nanmax(v[:max(3, len(v) // 5)])
            ax.axhline(pp.DEHYD_FRACTION * shallow, color="k", ls="--", lw=0.8)
            ax.text(z_ref[good].max(), pp.DEHYD_FRACTION * shallow,
                    f"{pp.DEHYD_FRACTION:.0%} of shallow value", ha="right",
                    va="bottom", fontsize=7)
        if inwin.any():
            ax.axvspan(z_ref[inwin].min(), z_ref[inwin].max(), color="crimson",
                       alpha=0.08, label=f"{win[0]:g}\u2013{win[1]:g} GPa "
                                         f"(release = {rel:.2f} wt%)")
        if np.isfinite(z_step):
            ax.axvline(z_step, color="goldenrod", lw=1.5)
            ax.plot(z_step, pp.DEHYD_FRACTION * shallow, "D", ms=8,
                    mfc="gold", mec="k",
                    label=f"50 % step: {z_step:.0f} km")
        ax.set_xlabel("slab-surface depth (km)")
        ax.set_ylabel("bound H$_2$O (wt%)")
        ax.set_title(f"along the slab  \u2014  crust at 2 GPa/600 \u00b0C: "
                     f"{ref_val:.2f} wt%", fontsize=9)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, frameon=False, loc="upper right")
        summary.append(dict(pair=key, MgO_upper=meta["MgO_pct"],
                            al_type=meta["al_type"], h2o_at_ref=ref_val,
                            dehyd_depth_km=z_step, release_wt=rel))

    from matplotlib.patches import Patch
    h, l = axs[0, 0].get_legend_handles_labels()
    has_masked = any(np.isnan(g[2]).any() for pair in grids.values()
                     for g in pair)
    if has_masked:
        h.append(Patch(facecolor="white", edgecolor="0.4", hatch="////"))
        l.append("water-undersaturated (masked)")
    fig.legend(h, l, loc="lower center", ncol=min(len(l), 5), fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    cb = fig.colorbar(mesh, cax=cax)
    cb.set_label("bound H$_2$O in solids (wt%)")
    out = os.path.join(args.output, f"pt_maps_f{f:.2f}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"\nSaved: {out} (+ .pdf)")
    print(pd.DataFrame(summary).round(2).to_string(index=False))
    if not sols:
        print("\nNo --solidus-csv given: no solidus drawn.")


if __name__ == "__main__":
    main()
