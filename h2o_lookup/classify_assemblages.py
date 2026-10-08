"""
classify_assemblages.py
=======================
Turn a long-format phase-modes table into ~10 geologically meaningful
assemblage fields, map them over P-T with the slab path on top, and report
which field the main dehydration step sits in.

Input schema (one row per run / P-T point / phase), written by the
--keep-scratch rerun of the Julia driver:

    run_id, P_GPa, T_K, phase, wt_pct

Phase names are whatever Perple_X prints: Atg, B, Chl, T (or Tlc), cAmph,
O, Opx, Cpx, Gt, Fsp, Sp, law, zo, cz, ilm, q, coe, ...

Classification is priority-ordered: the first rule that matches wins, so the
carriers that dominate the H2O budget are tested before the anhydrous
framework silicates that are present nearly everywhere.

Usage
-----
    python classify_assemblages.py --phases h2o_runs/v3_phases/phases.csv \\
        --runs h2o_runs/v3 --paths design_outputs/pt_paths \\
        --output response_outputs/assemblages
"""

import argparse
import importlib.util
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors

# Perple_X abbreviations -> our groups. Case-insensitive, prefix match.
GROUPS = {
    "atg":   ["atg"],
    "brucite": ["b(", "br", "b"],
    "talc":  ["t(", "tlc", "ta", "t"],
    "chl":   ["chl", "clin"],
    "amph":  ["camph", "amph", "oamph", "tr", "gl"],
    "law":   ["law"],
    "zo":    ["zo", "cz", "ep"],
    "gt":    ["gt", "gr"],
    "cpx":   ["cpx", "di", "jd", "o(hgp)cpx"],
    "opx":   ["opx", "en"],
    "ol":    ["o(", "fo", "fa", "o"],
    "fsp":   ["fsp", "pl", "ab", "an"],
    "sp":    ["sp"],
    "qz":    ["q", "coe"],
    "oxide": ["ilm", "ru", "mt"],
}
HYDROUS = ["atg", "brucite", "talc", "chl", "amph", "law", "zo"]


def group_of(name):
    n = str(name).strip().lower()
    for g, prefixes in GROUPS.items():
        for p in prefixes:
            if n.startswith(p):
                return g
    return "other"


def classify(row, thresh):
    """Priority-ordered rules. `row` is a dict group -> wt%."""
    on = {g: row.get(g, 0.0) >= thresh for g in list(GROUPS) + ["other"]}
    hyd = sum(row.get(g, 0.0) for g in HYDROUS)

    if on["atg"] and on["brucite"]:
        return "Serpentinite + brucite"
    if on["atg"] and on["chl"]:
        return "Serpentinite + chlorite"
    if on["atg"]:
        return "Serpentinite"
    if on["talc"]:
        return "Talc-bearing"
    if on["chl"] and on["amph"]:
        return "Chlorite amphibolite"
    if on["amph"] and on["law"]:
        return "Lawsonite blueschist"
    if on["amph"] and on["gt"]:
        return "Garnet amphibolite"
    if on["amph"]:
        return "Amphibolite"
    if on["chl"]:
        return "Chlorite-bearing"
    if on["law"] or on["zo"]:
        return "Lawsonite / zoisite eclogite"
    if hyd < thresh and on["gt"] and on["cpx"]:
        return "Eclogite (dry)"
    if hyd < thresh:
        return "Anhydrous"
    return "Other hydrous"


ORDER = ["Serpentinite + brucite", "Serpentinite + chlorite", "Serpentinite",
         "Talc-bearing", "Chlorite amphibolite", "Garnet amphibolite",
         "Amphibolite", "Lawsonite blueschist", "Chlorite-bearing",
         "Lawsonite / zoisite eclogite", "Eclogite (dry)", "Anhydrous",
         "Other hydrous"]


def field_table(phases, thresh):
    p = phases.copy()
    p["group"] = p["phase"].map(group_of)
    wide = (p.groupby(["run_id", "P_GPa", "T_K", "group"])["wt_pct"].sum()
              .unstack("group").fillna(0.0).astype(float).reset_index())
    wide["field"] = [classify(r, thresh) for r in wide.to_dict("records")]
    return wide


def plot_fields(wide, run_id, path, outdir, h2o_grid=None):
    d = wide[wide["run_id"] == run_id]
    if d.empty:
        print(f"  no phase rows for {run_id}")
        return
    present = [f for f in ORDER if f in set(d["field"])]
    cmap = plt.get_cmap("tab20", max(len(present), 3))
    idx = {f: i for i, f in enumerate(present)}
    P = np.sort(d["P_GPa"].unique()); T = np.sort(d["T_K"].unique())
    M = np.full((len(P), len(T)), np.nan)
    pi = {v: i for i, v in enumerate(P)}; ti = {v: i for i, v in enumerate(T)}
    for _, r in d.iterrows():
        M[pi[r["P_GPa"]], ti[r["T_K"]]] = idx[r["field"]]

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.pcolormesh(T - 273.15, P, np.ma.masked_invalid(M), shading="nearest",
                  cmap=cmap, norm=colors.BoundaryNorm(np.arange(-0.5, len(present)), cmap.N))
    if h2o_grid is not None:
        Ph, Th, G = h2o_grid
        cs = ax.contour(Th - 273.15, Ph, G, levels=[1, 2, 4, 6, 8, 10],
                        colors="k", linewidths=0.7)
        ax.clabel(cs, fmt="%g", fontsize=7)
    if path is not None:
        ax.plot(path["T_K"] - 273.15, path["P_GPa"], color="crimson", lw=2.2)
        step = max(1, len(path) // 8)
        for i in range(0, len(path), step):
            ax.annotate(f"{path['depth_km'].iloc[i]:.0f}",
                        (path["T_K"].iloc[i] - 273.15, path["P_GPa"].iloc[i]),
                        fontsize=6, color="crimson", xytext=(3, 3),
                        textcoords="offset points")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=cmap(idx[f])) for f in present]
    ax.legend(handles, present, fontsize=7, loc="upper left",
              bbox_to_anchor=(1.02, 1.0))
    ax.set_xlabel("T (°C)"); ax.set_ylabel("P (GPa)")
    ax.set_title(f"Assemblage fields — {run_id}\n(black contours: bound H$_2$O wt%; "
                 "red: slab path, labels = depth km)", fontsize=9)
    fig.tight_layout()
    p = os.path.join(outdir, f"assemblage_{run_id}.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {p}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", required=True)
    ap.add_argument("--runs", default="", help="h2o run tables, for H2O contours")
    ap.add_argument("--paths", default="", help="dir with pt_path_*.csv")
    ap.add_argument("--output", default="response_outputs/assemblages")
    ap.add_argument("--threshold", type=float, default=2.0,
                    help="wt%% below which a phase is ignored (default 2)")
    ap.add_argument("--postprocess", default=os.path.join(here, "postprocess_response.py"))
    args = ap.parse_args()
    os.makedirs(args.output, exist_ok=True)

    phases = pd.read_csv(args.phases)
    raw = phases["phase"].nunique()
    wide = field_table(phases, args.threshold)
    wide.to_csv(os.path.join(args.output, "assemblage_fields.csv"), index=False)
    print(f"  {raw} raw phase names -> {wide['field'].nunique()} fields "
          f"at a {args.threshold} wt% threshold")
    print(wide.groupby("field").size().sort_values(ascending=False).to_string())

    unknown = sorted({p for p in phases["phase"].unique() if group_of(p) == "other"})
    if unknown:
        print(f"\n  unmapped phase names (add to GROUPS if any matter): {unknown}")

    runs = {}
    if args.runs:
        spec = importlib.util.spec_from_file_location("pp", args.postprocess)
        pp = importlib.util.module_from_spec(spec); spec.loader.exec_module(pp)
        runs = pp.load_runs(args.runs)

    for rid in sorted(wide["run_id"].unique()):
        layer = "SlabGabbro" if rid.endswith("lower") or "_lower" in rid else "SlabMORB"
        path = (pd.read_csv(os.path.join(args.paths, f"pt_path_{layer}.csv"))
                if args.paths else None)
        grid = None
        if rid in runs:
            Pg, Tg, G = pp.to_grid(runs[rid])
            grid = (Pg, Tg, G)
        plot_fields(wide, rid, path, args.output, grid)


if __name__ == "__main__":
    main()
