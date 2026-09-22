"""
make_figures.py
===============
Publication-track figures from the v3 ensemble, the real TerraFERMA slab
paths, and the per-run P-T tables. Reads only existing output -- no Perple_X.

  fig1_pt_map_<pair>.png    bound H2O over P-T for one pair's upper and lower
                            crust, with that layer's slab path drawn on it
                            (upper -> SlabMORB, lower -> SlabGabbro) and the
                            2 GPa / 600 C reference point marked
  fig2_profiles.png         bound H2O vs depth along the path, every pair,
                            coloured by MgO bin, split by Al type
  fig3_dehydration.png      dehydration depth vs MgO, and the release rate
                            -d(H2O)/d(depth) vs depth by MgO bin -- where the
                            water actually comes off
  fig4_layers_<pair>.png    upper / lower / mixed profiles for one pair

Not a phase-assemblage pseudosection: the run tables store h2o_solid,
fluid_wt, saturated and n_phases only. Assemblage fields need a rerun with
--keep-scratch and the phase-field classification.

Usage
-----
    python make_figures.py --manifest design_outputs/composition_manifest.csv \
        --runs h2o_runs/v3 --paths design_outputs/pt_paths \
        --output response_outputs/figures --f 0.35
"""

import argparse
import importlib.util
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors


def load_pp(path):
    spec = importlib.util.spec_from_file_location("pp", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_path(paths_dir, name):
    d = pd.read_csv(os.path.join(paths_dir, f"pt_path_{name}.csv"))
    d = d.sort_values("x_km")
    return {"P": d["P_GPa"].to_numpy(float), "T": d["T_K"].to_numpy(float),
            "x": d["x_km"].to_numpy(float), "depth": d["depth_km"].to_numpy(float)}


def pick_pair(pairs, mgo_target, al_type):
    sub = pairs[pairs["al_type"] == al_type] if al_type else pairs
    if sub.empty:
        sub = pairs
    return int(sub.iloc[(sub["MgO_pct"] - mgo_target).abs().argsort().iloc[0]]["pair_id"])


def fig_pt_map(pp, runs, pairs, pair_id, pu, pl, outdir, ref_p, ref_t):
    meta = pairs.set_index("pair_id").loc[pair_id]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, (layer, path, pname) in zip(axes, [("upper", pu, "SlabMORB"),
                                               ("lower", pl, "SlabGabbro")]):
        rid = f"p{pair_id:04d}_{layer}"
        if rid not in runs:
            continue
        P, T, G = pp.to_grid(runs[rid])
        m = ax.pcolormesh(T - 273.15, P, np.ma.masked_invalid(G),
                          shading="nearest", cmap="viridis")
        cs = ax.contour(T - 273.15, P, G, levels=[1, 2, 4, 6, 8, 10],
                        colors="w", linewidths=0.6, alpha=0.7)
        ax.clabel(cs, fmt="%g", fontsize=7)
        ax.plot(path["T"] - 273.15, path["P"], color="crimson", lw=2,
                label=f"{pname} path")
        step = max(1, len(path["depth"]) // 8)
        for i in range(0, len(path["depth"]), step):
            ax.annotate(f"{path['depth'][i]:.0f}", (path["T"][i] - 273.15, path["P"][i]),
                        fontsize=6, color="crimson",
                        xytext=(3, 3), textcoords="offset points")
        ax.plot([ref_t - 273.15], [ref_p], "o", ms=7, mfc="none", mec="k", mew=1.5)
        ax.set_xlabel("T (°C)")
        ax.set_title(f"{layer} crust — MgO {meta[f'MgO_pct'] if layer=='upper' else np.nan:.1f}"
                     if layer == "upper" else f"{layer} crust (cumulate)")
        ax.set_ylim(P.min(), min(P.max(), np.nanmax(path["P"]) * 1.05))
        ax.set_xlim(T.min() - 273.15, T.max() - 273.15)
        ax.legend(loc="upper left", fontsize=8)
        fig.colorbar(m, ax=ax, label="bound H$_2$O (wt%)")
    axes[0].set_ylabel("P (GPa)")
    fig.suptitle(f"Bound H$_2$O over P-T with the slab path — pair {pair_id} "
                 f"({meta['al_type']}, upper MgO {meta['MgO_pct']:.1f} wt%); "
                 "red labels = depth (km)", fontsize=10)
    fig.tight_layout()
    p = os.path.join(outdir, f"fig1_pt_map_p{pair_id:04d}.png")
    fig.savefig(p, dpi=200); plt.close(fig)
    print(f"  Saved: {p}")


def all_profiles(pp, runs, pairs, pu, pl, f):
    """Mixed bound-H2O profile on the upper path's abscissa, for every pair."""
    prof, meta = {}, {}
    for pid in pairs["pair_id"]:
        ru, rl = f"p{pid:04d}_upper", f"p{pid:04d}_lower"
        if ru not in runs or rl not in runs:
            continue
        Pu, Tu, Gu = pp.to_grid(runs[ru])
        Pl, Tl, Gl = pp.to_grid(runs[rl])
        a = pp.profile_on(pu, Pu, Tu, Gu, pu["x"])
        b = pp.profile_on(pl, Pl, Tl, Gl, pu["x"])
        prof[pid] = {"upper": a, "lower": b, "mixed": f * a + (1 - f) * b}
        meta[pid] = pairs.set_index("pair_id").loc[pid]
    return prof, meta


def fig_profiles(prof, meta, depth, outdir, f):
    bins = sorted({m["mgo_bin_center"] for m in meta.values()})
    norm = colors.Normalize(min(bins), max(bins))
    sm = cm.ScalarMappable(norm=norm, cmap="plasma")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharex=True, sharey=True)
    for ax, al in zip(axes, ["Al_depleted", "Al_undepleted"]):
        for pid, p in prof.items():
            if meta[pid]["al_type"] != al:
                continue
            ax.plot(depth, p["mixed"], color=sm.to_rgba(meta[pid]["mgo_bin_center"]),
                    lw=0.8, alpha=0.55)
        for b in bins:
            ids = [i for i in prof if meta[i]["al_type"] == al
                   and meta[i]["mgo_bin_center"] == b]
            if ids:
                ax.plot(depth, np.nanmean([prof[i]["mixed"] for i in ids], axis=0),
                        color=sm.to_rgba(b), lw=2.4)
        ax.set_title(al.replace("_", "-"))
        ax.set_xlabel("depth along slab (km)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(f"bound H$_2$O, mixed crust at f = {f:.2f} (wt%)")
    fig.colorbar(sm, ax=axes, label="upper crust MgO bin (wt%)")
    fig.suptitle("Bound H$_2$O along the slab path (thin = individual pairs, thick = bin mean)")
    p = os.path.join(outdir, "fig2_profiles.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {p}")


def fig_dehydration(prof, meta, depth, scalars, outdir, f):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    sub = scalars[np.isclose(scalars["f"], f)]
    cols = {"Al_depleted": "tab:red", "Al_undepleted": "tab:blue"}
    for al, c in cols.items():
        s = sub[sub["al_type"] == al]
        axes[0].scatter(s["MgO_pct"], s["dehyd_depth_km"], s=18, alpha=0.45, color=c)
        g = s.groupby("mgo_bin_center")["dehyd_depth_km"].agg(["mean", "std"]).reset_index()
        axes[0].errorbar(g["mgo_bin_center"], g["mean"], yerr=g["std"], color=c,
                         lw=2, marker="o", capsize=3, label=al.replace("_", "-"))
    axes[0].set_xlabel("upper crust MgO (wt%)")
    axes[0].set_ylabel("dehydration depth (km)")
    axes[0].set_title("Where the main dehydration step sits")
    axes[0].grid(alpha=0.25); axes[0].legend(fontsize=8)

    bins = sorted({m["mgo_bin_center"] for m in meta.values()})
    sm = cm.ScalarMappable(norm=colors.Normalize(min(bins), max(bins)), cmap="plasma")
    for b in bins:
        ids = [i for i in prof if meta[i]["mgo_bin_center"] == b]
        if not ids:
            continue
        mean = np.nanmean([prof[i]["mixed"] for i in ids], axis=0)
        rate = -np.gradient(mean, depth)
        axes[1].plot(depth, rate, color=sm.to_rgba(b), lw=1.8, label=f"MgO {b:.0f}")
    axes[1].set_xlabel("depth along slab (km)")
    axes[1].set_ylabel("release rate  $-\\mathrm{d}$H$_2$O/$\\mathrm{d}z$ (wt%/km)")
    axes[1].set_title("Dehydration band")
    axes[1].grid(alpha=0.25); axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    p = os.path.join(outdir, "fig3_dehydration.png")
    fig.savefig(p, dpi=200); plt.close(fig)
    print(f"  Saved: {p}")


def fig_layers(prof, meta, depth, pair_id, outdir, f):
    p_ = prof[pair_id]; m = meta[pair_id]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(depth, p_["upper"], lw=2, label="upper crust (SlabMORB path)")
    ax.plot(depth, p_["lower"], lw=2, label="lower crust (SlabGabbro path)")
    ax.plot(depth, p_["mixed"], lw=2.6, color="k", label=f"mixed, f = {f:.2f}")
    ax.set_xlabel("depth along slab (km)")
    ax.set_ylabel("bound H$_2$O (wt%)")
    ax.set_title(f"Layer contributions — pair {pair_id} "
                 f"({m['al_type']}, upper MgO {m['MgO_pct']:.1f} wt%)")
    ax.grid(alpha=0.25); ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(outdir, f"fig4_layers_p{pair_id:04d}.png")
    fig.savefig(p, dpi=200); plt.close(fig)
    print(f"  Saved: {p}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--paths", required=True, help="directory with pt_path_*.csv")
    ap.add_argument("--scalars", default="response_outputs/scalars.csv")
    ap.add_argument("--output", default="response_outputs/figures")
    ap.add_argument("--f", type=float, default=0.35)
    ap.add_argument("--pairs", default="",
                    help="comma-separated pair_ids for fig1/fig4; default picks "
                         "a low-MgO and a high-MgO Al-undepleted pair")
    ap.add_argument("--postprocess", default=os.path.join(here, "postprocess_response.py"))
    args = ap.parse_args()
    os.makedirs(args.output, exist_ok=True)

    pp = load_pp(args.postprocess)
    manifest = pd.read_csv(args.manifest)
    runs = pp.load_runs(args.runs)
    pairs = manifest[manifest["layer"] == "upper"].copy()
    pu, pl = read_path(args.paths, "SlabMORB"), read_path(args.paths, "SlabGabbro")
    scalars = pd.read_csv(args.scalars)

    if args.pairs:
        chosen = [int(x) for x in args.pairs.split(",")]
    else:
        chosen = [pick_pair(pairs, 7.0, "Al_undepleted"),
                  pick_pair(pairs, 17.0, "Al_undepleted")]
    print(f"  featured pairs: {chosen}")

    for pid in chosen:
        fig_pt_map(pp, runs, pairs, pid, pu, pl, args.output, pp.REF_P_GPA, pp.REF_T_K)

    prof, meta = all_profiles(pp, runs, pairs, pu, pl, args.f)
    fig_profiles(prof, meta, pu["depth"], args.output, args.f)
    fig_dehydration(prof, meta, pu["depth"], scalars, args.output, args.f)
    for pid in chosen:
        if pid in prof:
            fig_layers(prof, meta, pu["depth"], pid, args.output, args.f)


if __name__ == "__main__":
    main()
