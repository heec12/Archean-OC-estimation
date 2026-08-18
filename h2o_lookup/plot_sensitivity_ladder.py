"""
plot_sensitivity_ladder.py
==========================
Visualize bound-H2O P-T grids from sensitivity_ladder_v2.jl.

Per scenario, produces a 2-row panel figure:
  Row 1: absolute bound H2O (wt%) for each ladder rung, shared color scale
  Row 2: difference maps (rung - previous rung), diverging scale about 0

Usage:
    python plot_sensitivity_ladder.py --results perplex_sensitivity/v2_results
    python plot_sensitivity_ladder.py --results ... --celsius --geotherm slabtop.csv

Optional --geotherm CSV: two columns (T_K, P_GPa), no header requirement
strict -- overlaid as a dashed path on every panel (e.g., a slab-top
geotherm extracted from a TerraFERMA case).
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUNGS = [
    ("L0_v1_effective", "L0: v1 effective\n(no amphibole)"),
    ("L1_plus_amph",    "L1: + cAmph(G)"),
    ("L2_plus_talc",    "L2: + T (talc)"),
    ("L3_v2_full",      "L3: + Sp(WPC)  [v2]"),
]
DELTA_LABELS = ["\u0394 from cAmph(G)", "\u0394 from talc", "\u0394 from Sp(WPC)"]
SCENARIOS = ["homogeneous_crust", "layered_cumulate_lower_crust"]


def read_grid(path):
    """Read a ladder grid CSV: T(K) columns, P(GPa) row labels."""
    df = pd.read_csv(path, index_col=0)
    T = df.columns.to_numpy(dtype=float)   # K
    P = df.index.to_numpy(dtype=float)     # GPa
    Z = df.to_numpy(dtype=float)
    return T, P, Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True,
                    help="OUT_DIR of sensitivity_ladder (e.g. perplex_sensitivity/v2_results)")
    ap.add_argument("--celsius", action="store_true",
                    help="Plot temperature axis in Celsius instead of Kelvin")
    ap.add_argument("--geotherm", default=None,
                    help="Optional CSV with columns T_K,P_GPa to overlay as a path")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    geo = None
    if args.geotherm:
        g = pd.read_csv(args.geotherm)
        geo = (g.iloc[:, 0].to_numpy(float), g.iloc[:, 1].to_numpy(float))

    for scenario in SCENARIOS:
        paths = [os.path.join(args.results, f"boundH2O_{scenario}_{r}.csv")
                 for r, _ in RUNGS]
        if not all(os.path.isfile(p) for p in paths):
            print(f"[skip] missing grids for {scenario}")
            continue

        grids = [read_grid(p) for p in paths]
        T, P = grids[0][0], grids[0][1]
        Zs = [Z for _, _, Z in grids]
        Tplot = T - 273.15 if args.celsius else T
        tlabel = "T (\u00b0C)" if args.celsius else "T (K)"

        vmax = np.nanmax([np.nanmax(Z) for Z in Zs])
        deltas = [Zs[i] - Zs[i - 1] for i in range(1, len(Zs))]
        dmax = max(np.nanmax(np.abs(d)) for d in deltas)
        dmax = max(dmax, 1e-6)

        fig, axes = plt.subplots(
            2, 4, figsize=(16, 7.2), sharex=True, sharey=True,
            constrained_layout=True,
        )

        # Row 1: absolute bound H2O
        for j, ((rung, label), Z) in enumerate(zip(RUNGS, Zs)):
            ax = axes[0, j]
            pc = ax.pcolormesh(Tplot, P, Z, cmap="viridis",
                               vmin=0, vmax=vmax, shading="auto")
            cs = ax.contour(Tplot, P, Z, levels=np.arange(0.5, vmax, 1.0),
                            colors="w", linewidths=0.5, alpha=0.6)
            ax.clabel(cs, fontsize=6, fmt="%.1f")
            ax.set_title(label, fontsize=10)
        fig.colorbar(pc, ax=axes[0, :], shrink=0.9, pad=0.01,
                     label="bound H$_2$O (wt%, solid basis)")

        # Row 2: rung-to-rung differences
        axes[1, 0].axis("off")
        axes[1, 0].text(0.5, 0.5, "attribution \u2192", ha="center",
                        va="center", fontsize=11, transform=axes[1, 0].transAxes)
        for j, (d, dl) in enumerate(zip(deltas, DELTA_LABELS), start=1):
            ax = axes[1, j]
            pd_ = ax.pcolormesh(Tplot, P, d, cmap="RdBu_r",
                                vmin=-dmax, vmax=dmax, shading="auto")
            ax.set_title(dl, fontsize=10)
        fig.colorbar(pd_, ax=axes[1, :], shrink=0.9, pad=0.01,
                     label="\u0394 bound H$_2$O (wt%)")

        # Axes cosmetics + optional geotherm overlay
        for ax in axes.ravel():
            if not ax.has_data():
                continue
            ax.set_xlim(Tplot.min(), Tplot.max())
            ax.set_ylim(P.min(), P.max())
            if geo is not None:
                gT = geo[0] - 273.15 if args.celsius else geo[0]
                ax.plot(gT, geo[1], "k--", lw=1.2, alpha=0.8)
        for ax in axes[1, 1:]:
            ax.set_xlabel(tlabel)
        axes[0, 3].set_xlabel(tlabel)  # in case bottom-left is blank
        for ax in axes[:, 0]:
            ax.set_ylabel("P (GPa)")
        axes[1, 1].set_ylabel("P (GPa)")

        fig.suptitle(f"Solution-model ladder \u2014 {scenario.replace('_', ' ')}",
                     fontsize=13)
        for ext in ("png", "pdf"):
            out = os.path.join(args.results, f"ladder_{scenario}.{ext}")
            fig.savefig(out, dpi=args.dpi)
            print(f"saved {out}")
        plt.close(fig)


if __name__ == "__main__":
    main()

