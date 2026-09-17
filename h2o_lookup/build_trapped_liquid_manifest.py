"""
build_trapped_liquid_manifest.py
================================
Sensitivity test for the trapped intercumulus liquid fraction in the coupled
cumulate lower crust (TRAPPED_LIQUID_FRAC = 0.15 in the production design).

Keeps every upper-crust draw exactly as it is in the production manifest and
recomputes ONLY the lower crust at other trapped-liquid fractions, using the
same build_lower_crust() from build_composition_design.py. Upper-crust runs
and the production 15 % lower-crust runs are reused, so the only new Perple_X
work is the extra lower-crust rows.

Before writing anything, the production lower crust is recomputed at the
production fraction and compared with the manifest. If they disagree, the
cumulate constants changed after the manifest was built and the test would
not be comparing like with like -- the script stops.

Output rows keep the production pair_id (so they join back to their upper
crust) and get run_ids like p0072_lower_tl05. Run them into a SEPARATE output
directory (driver --version v3_trapped) so the production h2o_runs/v3 stays
exactly 192 files.

Usage
-----
    python build_trapped_liquid_manifest.py \
        --manifest design_outputs/composition_manifest.csv \
        --design-script build_composition_design.py \
        --fractions 0.05,0.30 --bins 0,4,5 \
        --output design_outputs/manifest_trapped_liquid.csv
"""

import argparse
import importlib.util
import os

import numpy as np
import pandas as pd


def load_design_module(path):
    spec = importlib.util.spec_from_file_location("bcd", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    p.add_argument("--manifest", required=True, help="production manifest")
    p.add_argument("--design-script",
                   default=os.path.join(here, "build_composition_design.py"))
    p.add_argument("--fractions", default="0.05,0.30",
                   help="trapped-liquid fractions to add (production value is skipped)")
    p.add_argument("--bins", default="0,4,5",
                   help="mgo_bin_index values to include (0 = MgO 6-8, 5 = 16-18)")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    bcd = load_design_module(args.design_script)
    oxides = bcd.OXIDES
    prod_tl = float(bcd.TRAPPED_LIQUID_FRAC)

    man = pd.read_csv(args.manifest)
    fractions = [float(x) for x in args.fractions.split(",")]
    fractions = [f for f in fractions if not np.isclose(f, prod_tl)]
    bins = [int(b) for b in args.bins.split(",")]

    upper = man[man["layer"] == "upper"].set_index("pair_id")
    lower = man[man["layer"] == "lower"].set_index("pair_id")

    # --- 1. Reproduce the production lower crust -------------------------------
    worst = 0.0
    for pid, u in upper.iterrows():
        x_lower, _, _ = bcd.build_lower_crust(u[oxides].to_numpy(float), prod_tl)
        worst = max(worst, float(np.max(np.abs(x_lower - lower.loc[pid, oxides].to_numpy(float)))))
    print(f"Reproduction check at trapped liquid = {prod_tl:.2f}: "
          f"max |diff| = {worst:.4f} wt% over {len(upper)} pairs")
    if worst > 0.01:   # manifest rounds oxides to 1e-4
        raise SystemExit("ERROR: cumulate constants no longer reproduce the production "
                         "lower crust. Do not run this test against this manifest.")

    # --- 2. New lower-crust rows ---------------------------------------------
    sel = upper[upper["mgo_bin_index"].isin(bins)]
    rows = []
    for tl in fractions:
        tag = f"tl{int(round(tl * 100)):02d}"
        for pid, u in sel.iterrows():
            x_lower, _, _ = bcd.build_lower_crust(u[oxides].to_numpy(float), tl)
            row = lower.loc[pid].to_dict()          # same metadata as production
            row["pair_id"] = pid
            row["trapped_liquid"] = tl
            row["run_id"] = f"p{pid:04d}_lower_{tag}"
            for name, val in zip(oxides, x_lower):
                row[name] = round(float(val), 4)
            rows.append(row)

    out = pd.DataFrame(rows)[man.columns]           # identical column order
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out.to_csv(args.output, index=False)

    # --- 3. Report -------------------------------------------------------------
    print(f"\nWrote {len(out)} lower-crust rows -> {args.output}")
    print(f"  pairs: {len(sel)}  x  fractions {fractions}")
    print(f"  unique source analyses: {sel['source_sample'].nunique()}")
    print("\nMean lower-crust Al2O3 / MgO (wt%) by bin and trapped-liquid fraction:")
    ref = lower.loc[sel.index].assign(trapped_liquid=prod_tl)
    both = pd.concat([ref.reset_index(), out])
    print(both.groupby(["mgo_bin_center", "trapped_liquid"])[["Al2O3_pct", "MgO_pct", "Na2O_pct"]]
              .mean().round(2).to_string())


if __name__ == "__main__":
    main()
