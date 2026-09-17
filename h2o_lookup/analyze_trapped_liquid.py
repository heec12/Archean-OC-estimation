"""
analyze_trapped_liquid.py
=========================
Does the reference-point result survive a different trapped-liquid fraction?

Combines the production runs (upper crust + lower crust at 15 %) with the
extra lower-crust runs from build_trapped_liquid_manifest.py, then for each
trapped-liquid fraction reports, at 2 GPa / 600 C:

  * lower-crust bound H2O by MgO bin and Al type
  * mixed bound H2O at f (upper crust from production, lower at that fraction)
  * the design-axes model  h2o ~ MgO + Al_type + MgO x Al_type  on the mixed
    value, with a bootstrap by source analysis, and the same on the lower crust

Only the path-independent reference point is used, so this does not wait for
the real slab-top P-T path.

Usage
-----
    python analyze_trapped_liquid.py \
        --manifest    design_outputs/composition_manifest.csv \
        --tl-manifest design_outputs/manifest_trapped_liquid.csv \
        --runs h2o_runs/v3 --tl-runs h2o_runs/v3_trapped \
        --output response_outputs/trapped_liquid
"""

import argparse
import importlib.util
import os

import numpy as np
import pandas as pd


def load_pp(path):
    spec = importlib.util.spec_from_file_location("pp", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def axes_fit(sub, col, n_boot=1000, seed=0):
    y = sub[col].to_numpy(float)
    mgo = sub["MgO_upper"].to_numpy(float)
    mgo_c = mgo - mgo.mean()
    und = sub["al_type"].str.contains("undepleted", case=False).to_numpy(float)
    A = np.column_stack([np.ones_like(y), mgo_c, und, mgo_c * und])
    coef = np.linalg.lstsq(A, y, rcond=None)[0]

    codes = pd.factorize(sub["source_sample"].astype(str))[0]
    members = [np.flatnonzero(codes == c) for c in range(codes.max() + 1)]
    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, 4))
    for k in range(n_boot):
        i = np.concatenate([members[c] for c in rng.integers(0, len(members), len(members))])
        boot[k] = np.linalg.lstsq(A[i], y[i], rcond=None)[0]
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    r2 = 1 - np.sum((y - A @ coef) ** 2) / np.sum((y - y.mean()) ** 2)
    return coef, lo, hi, r2


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--tl-manifest", required=True)
    p.add_argument("--runs", required=True, help="production run tables (h2o_runs/v3)")
    p.add_argument("--tl-runs", required=True, help="trapped-liquid run tables")
    p.add_argument("--output", default="response_outputs/trapped_liquid")
    p.add_argument("--f", type=float, default=0.35)
    p.add_argument("--postprocess", default=os.path.join(here, "postprocess_response.py"))
    args = p.parse_args()
    os.makedirs(args.output, exist_ok=True)

    pp = load_pp(args.postprocess)
    man = pd.read_csv(args.manifest)
    tlm = pd.read_csv(args.tl_manifest)
    runs = pp.load_runs(args.runs)
    runs.update(pp.load_runs(args.tl_runs))

    missing = sorted(set(tlm["run_id"]) - set(runs))
    if missing:
        print(f"  WARNING: {len(missing)} trapped-liquid runs missing, e.g. {missing[:3]}")

    def ref(run_id):
        if run_id not in runs:
            return np.nan
        P, T, G = pp.to_grid(runs[run_id])
        return pp.value_at(P, T, G, pp.REF_P_GPA, pp.REF_T_K)

    upper = man[man["layer"] == "upper"].set_index("pair_id")
    prod_lower = man[man["layer"] == "lower"].copy()
    lowers = pd.concat([prod_lower, tlm])
    lowers = lowers[lowers["pair_id"].isin(tlm["pair_id"])]

    recs = []
    for _, r in lowers.iterrows():
        u = upper.loc[r["pair_id"]]
        h_u = ref(u["run_id"])
        h_l = ref(r["run_id"])
        recs.append({
            "pair_id": r["pair_id"], "trapped_liquid": r["trapped_liquid"],
            "mgo_bin_center": u["mgo_bin_center"], "al_type": u["al_type"],
            "source_sample": u["source_sample"], "source_terrane": u["source_terrane"],
            "MgO_upper": u["MgO_pct"], "Al2O3_lower": r["Al2O3_pct"],
            "h2o_upper_at_ref": h_u, "h2o_lower_at_ref": h_l,
            "h2o_at_ref": args.f * h_u + (1 - args.f) * h_l,
        })
    df = pd.DataFrame(recs).dropna(subset=["h2o_at_ref"])
    df.to_csv(os.path.join(args.output, "trapped_liquid_scalars.csv"), index=False)

    print(f"\n=== Bound H2O at {pp.REF_P_GPA} GPa / {pp.REF_T_K - 273:.0f} C, f = {args.f} ===")
    for col in ["h2o_lower_at_ref", "h2o_at_ref"]:
        tab = (df.groupby(["mgo_bin_center", "al_type", "trapped_liquid"])[col]
                 .mean().unstack("trapped_liquid").round(2))
        print(f"\n[{col}]  mean by cell; columns = trapped-liquid fraction")
        print(tab.to_string())

    print("\n=== Design-axes model per fraction (bootstrap by source analysis) ===")
    labels = ["intercept", "MgO slope", "Al_undepleted offset", "MgO x Al"]
    out = []
    for col in ["h2o_at_ref", "h2o_lower_at_ref"]:
        for tl, sub in df.groupby("trapped_liquid"):
            coef, lo, hi, r2 = axes_fit(sub, col)
            print(f"\n[{col}]  trapped liquid = {tl:.2f}   n = {len(sub)}   R^2 = {r2:.3f}")
            for name, c, a, b in zip(labels, coef, lo, hi):
                print(f"  {name:22s} {c:+.3f}   [{a:+.3f}, {b:+.3f}]")
                out.append(dict(scalar=col, trapped_liquid=tl, term=name,
                                coef=c, ci_lo=a, ci_hi=b, r2=r2, n=len(sub)))
    pd.DataFrame(out).round(4).to_csv(
        os.path.join(args.output, "trapped_liquid_axes.csv"), index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    styles = {"Al_depleted": ("tab:red", "o"), "Al_undepleted": ("tab:blue", "s")}
    for ax, col, title in zip(axes, ["h2o_lower_at_ref", "h2o_at_ref"],
                              ["lower crust", f"mixed, f = {args.f}"]):
        for (al, tl), g in df.groupby(["al_type", "trapped_liquid"]):
            m = g.groupby("mgo_bin_center")[col].agg(["mean", "std"])
            color, marker = styles.get(al, ("k", "o"))
            ls = {0: ":", 1: "-", 2: "--"}.get(sorted(df.trapped_liquid.unique()).index(tl), "-.")
            ax.errorbar(m.index, m["mean"], yerr=m["std"], color=color, marker=marker,
                        ls=ls, capsize=3, label=f"{al}, TL {tl:.2f}")
        ax.set_xlabel("upper crust MgO bin (wt%)")
        ax.set_ylabel("bound H$_2$O at 2 GPa, 600 °C (wt%)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output, "trapped_liquid_response.png"), dpi=200)
    print(f"\nSaved tables and figure to {args.output}/")


if __name__ == "__main__":
    main()
