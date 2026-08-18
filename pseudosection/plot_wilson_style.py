"""
plot_wilson_style.py
====================
Render bound-H2O P-T grids in the style of Wilson et al. (2014) for direct
visual comparison:
  - pressure on the y-axis, INCREASING DOWNWARD (0 at top)
  - temperature on the TOP x-axis
  - depth z(km) on the right axis (lithostatic, ~30.9 km/GPa)
  - white -> blue -> cyan -> green -> yellow -> red -> magenta colormap

Reads the existing ladder CSVs (L0_v1_effective = v1, L1_plus_amph = v2).
Runs on your Mac with matplotlib.

Usage:
    python plot_wilson_style.py --results v2_results
    python plot_wilson_style.py --results v2_results --vmax 3.0 --fraction
"""

import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

# Wilson-like colormap: white at 0, then through the spectrum to magenta.
WILSON_CMAP = LinearSegmentedColormap.from_list("wilson", [
    "#ffffff", "#0000ff", "#00ffff", "#00ff00",
    "#ffff00", "#ff8000", "#ff0000", "#ff00ff",
])

KM_PER_GPA = 30.9   # lithostatic: 8 GPa -> ~247 km, matches Wilson right axis

# Reads percentile compute output: boundH2O_<scenario>_<set>_<pct>.csv
# Default = v2, p05/p50/p95 for both scenarios (Wilson-style rows).
PANELS = [
    ("homogeneous_crust",            "v2", "p05", "homogeneous — v2 p05"),
    ("homogeneous_crust",            "v2", "p50", "homogeneous — v2 p50"),
    ("homogeneous_crust",            "v2", "p95", "homogeneous — v2 p95"),
    ("layered_cumulate_lower_crust", "v2", "p05", "layered — v2 p05"),
    ("layered_cumulate_lower_crust", "v2", "p50", "layered — v2 p50"),
    ("layered_cumulate_lower_crust", "v2", "p95", "layered — v2 p95"),
]


def read_grid(path):
    df = pd.read_csv(path, index_col=0)
    T = df.columns.to_numpy(float)   # K
    P = df.index.to_numpy(float)     # GPa
    return T, P, df.to_numpy(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--vmax", type=float, default=None,
                    help="max of color scale (wt%, or fraction if --fraction)")
    ap.add_argument("--fraction", action="store_true",
                    help="plot bound-water FRACTION (wt%/100) to match Wilson's F")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    grids = []
    for scen, mset, pct, title in PANELS:
        p = os.path.join(args.results, f"boundH2O_{scen}_{mset}_{pct}.csv")
        if not os.path.isfile(p):
            print(f"[skip] missing {p}")
            continue
        T, P, Z = read_grid(p)
        if args.fraction:
            Z = Z / 100.0
        grids.append((title, T, P, Z))

    if not grids:
        raise SystemExit("No grids found — check --results path.")

    vmax = args.vmax or max(np.nanmax(Z) for _, _, _, Z in grids)
    clabel = "bound-water fraction, F" if args.fraction else "bound H$_2$O (wt%)"

    ncols = 3
    nrows = int(np.ceil(len(grids) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 5.2 * nrows),
                             constrained_layout=True, squeeze=False)
    flat = axes.ravel()

    pcm = None
    for ax, (title, T, P, Z) in zip(flat, grids):
        pcm = ax.pcolormesh(T, P, Z, cmap=WILSON_CMAP,
                            vmin=0, vmax=vmax, shading="auto")
        ax.set_ylim(P.max(), P.min())            # pressure increases downward
        ax.xaxis.set_ticks_position("top")        # T on the top axis
        ax.xaxis.set_label_position("top")
        ax.set_xlabel("T (K)")
        ax.set_title(title, pad=28, fontsize=11)
        ax.set_ylabel("$p_l$ (GPa)")
        sec = ax.secondary_yaxis(
            "right", functions=(lambda p: p * KM_PER_GPA,
                                lambda z: z / KM_PER_GPA))
        sec.set_ylabel("z (km)")
    for ax in flat[len(grids):]:
        ax.axis("off")

    fig.colorbar(pcm, ax=list(flat), orientation="horizontal",
                 fraction=0.05, pad=0.04, shrink=0.6, label=clabel)

    for ext in ("pdf", "png"):
        out = os.path.join(args.results, f"wilson_style_boundH2O.{ext}")
        fig.savefig(out, dpi=args.dpi)
        print("saved", out)


if __name__ == "__main__":
    main()

