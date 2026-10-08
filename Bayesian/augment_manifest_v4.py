"""
augment_manifest_v4.py
======================
Stage 1b of the v4 pipeline. Takes the v3 design manifest (NCFMAST, 7 oxides)
and turns it into a KNCFMASHTO run manifest for MAGEMin.

It does NOT redraw compositions. The v3 design (MgO bin x Al type x replicate,
coupled cumulate lower crust) is kept exactly, so v3 and v4 results can be
compared row for row. What changes is the chemical system and the
database assignment:

  * Fe3+  -- FeOtot is split into FeO + Fe2O3 at a fixed molar
             Fe3+/sum(Fe) (default 0.10, MORB-glass value; Palin, pers. comm.
             Oct 2026). Applied to both layers.
  * K2O   -- NOT taken from the GSWA analyses by default. Measured K2O in the
             design's source samples has a median of ~1.2 wt% (range 0.01-4.3),
             roughly an order of magnitude above MORB: it is an alteration
             signal, not a protolith one. Instead the upper crust gets a fixed
             MORB-like value, and K2O is varied as a sensitivity on a pilot
             subset (levels below, plus the measured value for comparison).
             Lower crust K2O = trapped-liquid fraction x upper-crust K2O
             (K treated as perfectly incompatible in the cumulate).
  * DB    -- three scenarios (see RUN_PLAN):
               v4a  `all` database, v3-like phase list, ds62, NCFMAST
                    (K2O = 0, Fe3+ = 0): closest replica of v3 in MAGEMin
               v4b  `mb` (Green et al. 2016) for both layers
               v4   `mb` upper; `ume` lower (Evans & Frost 2021 + G16
                    amp/aug) as PRIMARY, `ig` (Green et al. 2025) as check.
                    The stitch rule (ume up to antigorite-out, ig beyond) is
                    applied in post-processing, never here.

H2O is NOT added here. Excess water is a run setting, applied by the driver,
so it can be changed without rebuilding the manifest.

Output columns: everything from the v3 manifest (renamed `v3_run_id`), plus

  run_id, layer, db, phase_set, scenarios, k2o_level, k2o_measured_wt,
  fe3_ratio, pilot,
  SiO2, TiO2, Al2O3, FeO, Fe2O3, MgO, CaO, Na2O, K2O   (wt%, anhydrous, sum 100)

Usage
-----
    python augment_manifest_v4.py \
        --manifest ../design_outputs/composition_manifest.csv \
        --data GSWA_Smithies2018_appendix1.xlsx --sheet Sheet0 \
        --output ../design_outputs/composition_manifest_v4.csv
"""

import argparse

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
FE3_RATIO_DEFAULT = 0.10      # molar Fe3+/sum(Fe)

# MORB-like baseline K2O for the upper crust, wt%.
# PLACEHOLDER: check against a MORB compilation (e.g. Gale et al. 2013, G3)
# before the production run. The sensitivity levels bracket it either way.
K2O_MORB = 0.15
K2O_SENSITIVITY_LEVELS = {"k005": 0.05, "k050": 0.50}   # plus "kmeas" (measured)

# Pilot subset: replicate 0 of these MgO bins, both Al types -> 6 pairs.
# Spans the bottom, middle and top of the MgO axis.
PILOT_MGO_BINS = (0, 2, 5)

# -----------------------------------------------------------------------------
# Scenario ladder (one change per step relative to the previous one)
#
#   v4a  MAGEMin `all`, v3-like phase list, ds62, NCFMAST (K2O = 0, Fe3+ = 0)
#        -> v3 vs v4a isolates the engine (as far as MAGEMin allows)
#   v4b  `mb` for BOTH layers, Fe3+/sumFe = 0.1, K2O MORB-like
#        -> self-consistent G16 set + Fe3+ + K. NB: mb has no antigorite,
#           brucite or talc, so the cumulate lower crust is expected to come
#           out too dry; that is the point of the comparison with v4.
#   v4   `mb` upper, `ume` lower (+ `ig` check), Fe3+ 0.1, K2O MORB-like
#
# Each run row is unique; `scenarios` lists every scenario that uses it
# (v4 and v4b share the upper-crust mb runs).
#
# (layer, db, phase_set, k2o tag, fe3 ratio or None for --fe3) -> scenarios
# -----------------------------------------------------------------------------
RUN_PLAN = [
    ("upper", "all", "v3like",  "k0", 0.0,  ["v4a"]),
    ("lower", "all", "v3like",  "k0", 0.0,  ["v4a"]),
    ("upper", "mb",  "default", "kmorb", None, ["v4b", "v4"]),
    ("lower", "mb",  "default", "kmorb", None, ["v4b"]),
    ("lower", "ume", "default", "kmorb", None, ["v4"]),
    ("lower", "ig",  "default", "kmorb", None, ["v4"]),
]
# K2O sensitivity (pilot pairs only), on the v4 databases that have K2O
K2O_SENS_DBS = {"upper": ["mb"], "lower": ["ig"]}

M_FEO = 71.844
M_FE2O3 = 159.688

V3_OXIDES = {   # v3 column -> v4 column
    "SiO2_pct": "SiO2", "TiO2_pct": "TiO2", "Al2O3_pct": "Al2O3",
    "MgO_pct": "MgO", "CaO_pct": "CaO", "Na2O_pct": "Na2O",
}
V4_OXIDES = ["SiO2", "TiO2", "Al2O3", "FeO", "Fe2O3",
             "MgO", "CaO", "Na2O", "K2O"]


def split_iron(feot_wt, fe3_ratio):
    """FeOtot (wt%) -> (FeO, Fe2O3) wt% at molar Fe3+/sum(Fe) = fe3_ratio.

    Mass is not conserved exactly: Fe2O3 carries extra oxygen. The final
    renormalisation to 100 absorbs it, as it would for a real analysis.
    """
    n_fe = feot_wt / M_FEO
    feo = (1.0 - fe3_ratio) * n_fe * M_FEO
    fe2o3 = 0.5 * fe3_ratio * n_fe * M_FE2O3
    return feo, fe2o3


def composition(row, k2o_wt, fe3_ratio):
    """Anhydrous KNCFMASTO composition, wt%, summing to 100.

    K2O is fixed at k2o_wt and the other oxides are scaled into the remaining
    100 - k2o_wt, so the requested K2O is what the driver actually receives.
    """
    x = {v4: float(row[v3]) for v3, v4 in V3_OXIDES.items()}
    x["FeO"], x["Fe2O3"] = split_iron(float(row["FeOtot_pct"]), fe3_ratio)
    s = sum(x.values())
    scale = (100.0 - k2o_wt) / s
    x = {k: v * scale for k, v in x.items()}
    x["K2O"] = k2o_wt
    return [x[o] for o in V4_OXIDES]


def load_measured_k2o(data_path, sheet):
    """Measured K2O per SampleID, renormalised anhydrous with the 7 majors."""
    g = pd.read_excel(data_path, sheet_name=sheet)
    g["FeOtot_pct"] = 0.8998 * g["Fe2O3T"]
    cols = ["SiO2_pct", "TiO2_pct", "Al2O3_pct", "FeOtot_pct",
            "MgO_pct", "CaO_pct", "Na2O_pct", "K2O_pct"]
    g = g.dropna(subset=cols)
    g["K2O_anh"] = 100.0 * g["K2O_pct"] / g[cols].sum(axis=1)
    g["SampleID"] = g["SampleID"].astype(str)
    return g.drop_duplicates("SampleID").set_index("SampleID")["K2O_anh"]


def build(manifest, k2o_meas, fe3_ratio):
    m = manifest.copy()
    m["source_sample"] = m["source_sample"].astype(str)
    m["k2o_measured_wt"] = m["source_sample"].map(k2o_meas)

    up = m[m.layer == "upper"].set_index("pair_id")
    pilot_pairs = set(
        up[(up.replicate == 0) & (up.mgo_bin_index.isin(PILOT_MGO_BINS))].index)

    meta_cols = [c for c in m.columns
                 if c not in list(V3_OXIDES) + ["FeOtot_pct", "run_id"]]
    out = []

    def emit(row, db, phase_set, ktag, k2o_upper, fe3, scenarios):
        trap = float(row["trapped_liquid"])
        k2o = k2o_upper if row["layer"] == "upper" else trap * k2o_upper
        dbtag = db if phase_set == "default" else f"{db}-{phase_set}"
        rec = {c: row[c] for c in meta_cols}
        rec.update(
            run_id=f"p{int(row['pair_id']):04d}_{row['layer']}_{dbtag}_{ktag}",
            v3_run_id=row["run_id"], db=db, phase_set=phase_set, k2o_level=ktag,
            fe3_ratio=fe3, scenarios=";".join(scenarios),
            pilot=int(row["pair_id"]) in pilot_pairs,
        )
        rec.update(dict(zip(V4_OXIDES, composition(row, k2o, fe3))))
        out.append(rec)

    for _, row in m.iterrows():
        pair = int(row["pair_id"])
        layer = row["layer"]
        # Measured K2O always refers to the upper-crust source sample.
        k_meas = float(up.loc[pair, "k2o_measured_wt"])

        for (lay, db, pset, ktag, fe3, scen) in RUN_PLAN:
            if lay != layer:
                continue
            k = 0.0 if ktag == "k0" else K2O_MORB
            emit(row, db, pset, ktag, k, fe3_ratio if fe3 is None else fe3, scen)

        if pair in pilot_pairs:
            levels = dict(K2O_SENSITIVITY_LEVELS)
            if np.isfinite(k_meas):
                levels["kmeas"] = k_meas
            for ktag, kval in levels.items():
                for db in K2O_SENS_DBS[layer]:
                    emit(row, db, "default", ktag, kval, fe3_ratio, ["v4-ksens"])

    df = pd.DataFrame(out)
    first = ["run_id", "v3_run_id", "pair_id", "layer", "db", "phase_set",
             "scenarios", "k2o_level", "pilot", "fe3_ratio", "k2o_measured_wt"]
    rest = [c for c in df.columns if c not in first + V4_OXIDES]
    return df[first + rest + V4_OXIDES]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--manifest", required=True, help="v3 composition_manifest.csv")
    p.add_argument("--data", required=True, help="GSWA Excel file (for measured K2O)")
    p.add_argument("--sheet", default="Sheet0")
    p.add_argument("--output", required=True)
    p.add_argument("--fe3", type=float, default=FE3_RATIO_DEFAULT,
                   help="molar Fe3+/sum(Fe) (default 0.10)")
    a = p.parse_args()

    v3 = pd.read_csv(a.manifest)
    df = build(v3, load_measured_k2o(a.data, a.sheet), a.fe3)

    sums = df[V4_OXIDES].sum(axis=1)
    assert np.allclose(sums, 100.0), "compositions do not close to 100"
    assert df.run_id.is_unique, "duplicate run_id"

    df.to_csv(a.output, index=False)
    print(f"wrote {len(df)} runs -> {a.output}")
    print(df.groupby(["scenarios", "layer", "db", "phase_set", "k2o_level"]).size().to_string())
    print(f"\nmeasured K2O of source samples (anhydrous wt%): "
          f"median {df.k2o_measured_wt.median():.2f}, "
          f"range {df.k2o_measured_wt.min():.2f}-{df.k2o_measured_wt.max():.2f}  "
          f"(baseline used: {K2O_MORB})")


if __name__ == "__main__":
    main()
