"""
postprocess_v4.py
=================
Stage 3 of the v4 pipeline (MAGEMin). Reads the per-run tables written by
run_magemin_manifest.jl and produces the reaction-resolved dehydration
metrics that replace v3's 50%-threshold "dehydration front".

What it does
------------
1. Lower crust, ume vs ig. `ume` is the primary database (it has antigorite
   and brucite; `ig` has neither). `ig` is carried along as the check:
     * beyond antigorite-out (in ume) the two should broadly agree; the
       difference is reported as `lower_ig_minus_ume`.
     * `ig` has melt, `ume` does not. Depths where ig is melt-bearing are
       flagged `ig_melt_present`. They are NOT masked by default: in the
       pilot, with 20 wt% excess H2O, ig produced a water-rich liquid
       (30-55 wt% H2O) from ~2.9 GPa on every path, at T as low as ~500 C.
       That is the liquid model standing in for the aqueous fluid, not a
       real solidus. Use --mask-lower-by-ig-melt to mask anyway; the proper
       solidus test is the minimum-water calculation (Palin).
2. Reaction-resolved release, per run and P-T path. Bound H2O is put on an
   ANHYDROUS-solid basis (g H2O per 100 g anhydrous rock) so that it is
   conserved as water leaves. Between consecutive path points, every phase
   whose H2O content falls is a donor; the net release is split among the
   donors in proportion to their losses (phases that grow, e.g. chlorite
   fed by antigorite breakdown, absorb part of it). Net uptake steps are
   charged to the phases that grew, so the per-phase totals sum exactly to
   the rock's net loss along the path. Integrating gives, per hydrous phase:
       released_wt   total H2O it released along the path
       frac_budget   share of the rock's initial bound H2O
       z10/z50/z90   depths (km) by which 10/50/90% of its release is done
   which is the "how much from mineral X, at what depth, what share of the
   budget" form Palin asked for.
3. Crust profile: upper (path_upper) and lower (path_lower) are mixed AFTER
   the phase equilibria, on the anhydrous basis, at common depths:
       H2O_crust(z) = f * H2O_upper(z) + (1 - f) * H2O_lower(z)
   f is swept (it costs nothing here).

Masking: points with MAGEMin status >= 3 (failed), no free fluid
(undersaturated: censored, not a capacity) or melt present are excluded
from the release accounting and flagged in the profile output.

Outputs (in --output):
    release_by_phase.csv    run_id x path x phase release summary
    release_steps.csv       per path step: depth, net release, donor phases
    lower_ume_vs_ig.csv     per pair x path: ume, ig, difference, flags
    crust_profiles.csv      per pair x f: mixed crust bound H2O vs depth
    pair_summary.csv        per pair: initial budget, main release depth/phase

Usage
-----
    python postprocess_v4.py --manifest design_outputs/composition_manifest_v4.csv \
        --runs h2o_runs/v4 --output response_v4 \
        --path-upper SlabSurface --path-lower SlabGabbro --f 0.2 0.3 0.4 0.5
"""

import argparse
import ast
import os

import numpy as np
import pandas as pd

# Scenario ladder: which run (run_id = p<pair>_<layer>_<tag>) is the upper
# crust, the lower crust, and (optionally) the lower-crust check database.
SCENARIOS = {
    "v4a": dict(upper="all-v3like_k0", lower="all-v3like_k0", check=None),
    "v4b": dict(upper="mb_kmorb",      lower="mb_kmorb",      check=None),
    "v4":  dict(upper="mb_kmorb",      lower="ume_kmorb",     check="ig_kmorb"),
}

# `ig` (Green et al. 2025, after Holland et al. 2018) is calibrated for
# T > 923 K; comparison points below that are flagged ig_in_range = False.
IG_MIN_T_K = 923.0

MIN_PHASE_WT = 1e-3         # wt% of solid; below this a phase is "absent"
HYDROUS_MIN = 0.05          # g/100 g: phases holding less than this and releasing
                            # nothing are left out of release_by_phase.csv


# -----------------------------------------------------------------------------
# I/O
# -----------------------------------------------------------------------------
def load_run(runs, run_id):
    fp = os.path.join(runs, f"pts_{run_id}.csv")
    fh = os.path.join(runs, f"phases_{run_id}.csv")
    if not (os.path.isfile(fp) and os.path.isfile(fh)):
        return None, None
    return pd.read_csv(fp), pd.read_csv(fh)


def valid_mask(pts):
    """Points usable as bound-H2O capacities."""
    return (pts["status"] < 3) & pts["saturated"].astype(bool) & ~pts["melt_present"].astype(bool)


def to_anhydrous(h2o_solid):
    """wt% H2O in solid -> g H2O per 100 g anhydrous solid."""
    h = np.asarray(h2o_solid, float)
    return 100.0 * h / (100.0 - h)


# -----------------------------------------------------------------------------
# Reaction-resolved release along one path
# -----------------------------------------------------------------------------
def phase_table(pts, phs, path):
    """Wide table: depth x phase -> H2O held (g / 100 g anhydrous solid)."""
    p = pts[(pts.kind == "path") & (pts.path == path)].sort_values("depth_km")
    if p.empty:
        return None, None
    p = p.assign(valid=valid_mask(p).values)
    h = phs[(phs.kind == "path") & (phs.path == path) & (phs.phase_kind == "solid")]
    # Solvus instances (two amp, two chl...) are summed per phase name.
    w = (h.groupby(["depth_km", "phase"])["h2o_contrib"].sum()
           .unstack(fill_value=0.0).reindex(p.depth_km.values, fill_value=0.0))
    w = w.div(1.0 - p.set_index("depth_km")["h2o_solid"] / 100.0, axis=0)  # anhydrous basis
    w = w.loc[:, (w.max() > 0.0)]                                           # hydrous phases only
    return p.reset_index(drop=True), w


def release_along_path(pts, phs, path):
    p, w = phase_table(pts, phs, path)
    if p is None or w.shape[1] == 0:
        return None, None
    ok = p.valid.values
    z = p.depth_km.values
    W = w.values
    steps, rel = [], np.zeros(W.shape[1])
    cum = np.zeros((len(z), W.shape[1]))
    last = None
    for i in range(len(z)):
        if not ok[i]:
            cum[i] = cum[i - 1] if i else 0.0
            continue
        if last is not None:
            d = W[i] - W[last]                         # per-phase change
            net = -d.sum()                             # >0 = water released
            loss = np.clip(-d, 0, None)
            gain = np.clip(d, 0, None)
            if net < -1e-9 and gain.sum() > 0:
                # net uptake (rehydration, or numerical wobble): charged to the
                # phases that grew, so per-phase totals close on the net loss.
                rel += net * gain / gain.sum()
            if net > 1e-9 and loss.sum() > 0:
                share = net * loss / loss.sum()
                rel += share
                donors = {str(w.columns[k]): round(float(share[k]), 4)
                          for k in np.argsort(-share) if share[k] > 1e-4}
                steps.append(dict(path=path, z_from=z[last], z_to=z[i],
                                  P_GPa=p.P_GPa.values[i], T_K=p.T_K.values[i],
                                  release=net, donors=donors))
        cum[i] = rel
        last = i

    budget0 = W[np.argmax(ok)].sum() if ok.any() else np.nan
    out = []
    for k, ph in enumerate(w.columns):
        tot = rel[k]
        if abs(tot) <= 1e-6 and w.iloc[:, k].max() < HYDROUS_MIN:
            continue
        frac = cum[:, k] / tot if tot > 1e-6 else np.zeros(len(z))
        zq = [float(np.interp(q, frac, z)) if frac[-1] >= q else np.nan
              for q in (0.1, 0.5, 0.9)]
        present = (w.iloc[:, k].values > 1e-4) & ok
        out.append(dict(path=path, phase=ph, released_wt=tot,
                        frac_budget=tot / budget0 if budget0 else np.nan,
                        z10_km=zq[0], z50_km=zq[1], z90_km=zq[2],
                        z_out_km=float(z[present][-1]) if present.any() else np.nan,
                        initial_wt=float(W[np.argmax(ok), k]) if ok.any() else np.nan))
    return pd.DataFrame(out), pd.DataFrame(steps)


# -----------------------------------------------------------------------------
# Profiles on a common depth axis
# -----------------------------------------------------------------------------
def profile(pts, path, z):
    p = pts[(pts.kind == "path") & (pts.path == path)].sort_values("depth_km")
    if p.empty:
        return np.full_like(z, np.nan), np.zeros_like(z, bool), np.zeros_like(z, bool)
    ok = valid_mask(p).values
    h = to_anhydrous(p.h2o_solid.values)
    h = np.where(ok, h, np.nan)
    hi = np.interp(z, p.depth_km.values, h, left=np.nan, right=np.nan)
    melt = np.interp(z, p.depth_km.values, p.melt_present.astype(float).values) > 0.5
    inval = np.interp(z, p.depth_km.values, (~ok).astype(float)) > 0.5
    return hi, melt, inval


def atg_out_depth(phs, path):
    h = phs[(phs.kind == "path") & (phs.path == path) & (phs.phase == "atg")
            & (phs.phase_wt_solid > MIN_PHASE_WT)]
    return float(h.depth_km.max()) if len(h) else np.nan


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="v4 reaction-resolved post-processing")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--output", default="response_v4")
    ap.add_argument("--path-upper", default="SlabSurface")
    ap.add_argument("--path-lower", default="SlabGabbro")
    ap.add_argument("--f", type=float, nargs="+", default=[0.2, 0.3, 0.4, 0.5],
                    help="upper-crust mass fraction(s)")
    ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS),
                    choices=list(SCENARIOS), help="scenario ladder steps to assemble")
    ap.add_argument("--dz", type=float, default=1.0, help="depth step for profiles, km")
    ap.add_argument("--mask-lower-by-ig-melt", action="store_true",
                    help="drop lower-crust (ume) values where ig is melt-bearing. OFF by "
                         "default: with excess H2O, ig produces a water-rich liquid "
                         "(30-55 wt%% H2O) from ~2.9 GPa even at ~500 C, so its melt is "
                         "not a usable solidus marker. See Methodology Updates.md.")
    a = ap.parse_args()
    os.makedirs(a.output, exist_ok=True)

    m = pd.read_csv(a.manifest)

    # ---- 1+2: release accounting for every run that exists ------------------
    rel_all, step_all = [], []
    cache = {}
    for _, r in m.iterrows():
        pts, phs = load_run(a.runs, r.run_id)
        if pts is None:
            continue
        cache[r.run_id] = (pts, phs)
        for path in sorted(pts.loc[pts.kind == "path", "path"].unique()):
            rel, st = release_along_path(pts, phs, path)
            if rel is None:
                continue
            meta = dict(run_id=r.run_id, pair_id=r.pair_id, layer=r.layer, db=r.db,
                        k2o_level=r.k2o_level, mgo_bin_center=r.mgo_bin_center,
                        al_type=r.al_type, MgO=r.MgO)
            rel_all.append(rel.assign(**meta))
            if len(st):
                step_all.append(st.assign(**meta))
    print(f"runs found: {len(cache)} of {len(m)}")
    if not cache:
        return
    rel_all = pd.concat(rel_all, ignore_index=True) if rel_all else pd.DataFrame()
    rel_all.to_csv(os.path.join(a.output, "release_by_phase.csv"), index=False)
    if step_all:
        st = pd.concat(step_all, ignore_index=True)
        st["donors"] = st["donors"].astype(str)
        st.to_csv(os.path.join(a.output, "release_steps.csv"), index=False)

    # ---- 3: per scenario: lower ume vs ig (v4 only), crust mixing, summary ----
    zmax = max(p[p.kind == "path"].depth_km.max() for p, _ in cache.values())
    z = np.arange(0.0, zmax + a.dz, a.dz)
    comp_rows, prof_rows, summ_rows = [], [], []
    pairs = m.drop_duplicates("pair_id").set_index("pair_id")

    def main_step(run_id, path):
        """Largest release over a 10 km window, and its dominant donor."""
        if not step_all:
            return (np.nan,) * 3
        s_ = st[(st.run_id == run_id) & (st.path == path)]
        if s_.empty:
            return (np.nan,) * 3
        zc = 0.5 * (s_.z_from.values + s_.z_to.values)
        best = max(zc, key=lambda c: s_.release[(zc >= c - 5) & (zc < c + 5)].sum())
        win = s_[(zc >= best - 5) & (zc < best + 5)]
        don = {}
        for d in win.donors:
            for kk, vv in ast.literal_eval(d).items():
                don[kk] = don.get(kk, 0) + vv
        return best, win.release.sum(), max(don, key=don.get) if don else ""

    for scen in a.scenarios:
        spec = SCENARIOS[scen]
        for pair, meta in pairs.iterrows():
            rid = {k: (f"p{int(pair):04d}_{k if k != 'check' else 'lower'}_{v}" if v else None)
                   for k, v in spec.items()}
            up = cache.get(rid["upper"])
            lo = cache.get(rid["lower"])
            ck = cache.get(rid["check"]) if rid["check"] else None

            # lower crust primary vs check db (v4: ume vs ig), every shared path
            if lo is not None and ck is not None:
                for path in sorted(lo[0].loc[lo[0].kind == "path", "path"].unique()):
                    hu, _, _ = profile(lo[0], path, z)
                    hi, melt_i, _ = profile(ck[0], path, z)
                    zatg = atg_out_depth(lo[1], path)
                    Ti = np.interp(z, *(lambda q: (q.depth_km.values, q.T_K.values))(
                        ck[0][(ck[0].kind == "path") & (ck[0].path == path)].sort_values("depth_km")))
                    for k in range(len(z)):
                        if np.isnan(hu[k]) and np.isnan(hi[k]):
                            continue
                        comp_rows.append(dict(scenario=scen, pair_id=pair, path=path,
                                              depth_km=z[k], T_K=Ti[k],
                                              lower_ume=hu[k], lower_ig=hi[k],
                                              lower_ig_minus_ume=hi[k] - hu[k],
                                              beyond_atg_out=bool(z[k] > zatg) if np.isfinite(zatg) else True,
                                              ig_in_range=bool(Ti[k] > IG_MIN_T_K),
                                              ig_melt_present=bool(melt_i[k]),
                                              atg_out_km=zatg))

            if up is None or lo is None:
                continue
            hu_up, _, _ = profile(up[0], a.path_upper, z)
            hl, _, _ = profile(lo[0], a.path_lower, z)
            if ck is not None and a.mask_lower_by_ig_melt:
                _, melt_lo, _ = profile(ck[0], a.path_lower, z)
                hl = np.where(melt_lo, np.nan, hl)
            for f in a.f:
                hc = f * hu_up + (1 - f) * hl
                for k in range(len(z)):
                    if np.isnan(hc[k]):
                        continue
                    prof_rows.append(dict(scenario=scen, pair_id=pair, f=f, depth_km=z[k],
                                          upper=hu_up[k], lower=hl[k], crust=hc[k],
                                          mgo_bin_center=meta.mgo_bin_center,
                                          al_type=meta.al_type))

            for layer, path in (("upper", a.path_upper), ("lower", a.path_lower)):
                r_id = rid[layer]
                pts = cache[r_id][0]
                pp = pts[(pts.kind == "path") & (pts.path == path)].sort_values("depth_km")
                okk = valid_mask(pp).values
                b0 = float(to_anhydrous(pp.h2o_solid.values[okk][:1])[0]) if okk.any() else np.nan
                zb, amt, ph = main_step(r_id, path)
                summ_rows.append(dict(scenario=scen, pair_id=pair, layer=layer,
                                      run_id=r_id, path=path,
                                      mgo_bin_center=meta.mgo_bin_center, al_type=meta.al_type,
                                      MgO=float(m.loc[m.run_id == r_id, "MgO"].iloc[0]),
                                      initial_bound_anh=b0, main_release_depth_km=zb,
                                      main_release_wt=amt, main_release_phase=ph))

    pd.DataFrame(comp_rows).to_csv(os.path.join(a.output, "lower_ume_vs_ig.csv"), index=False)
    pd.DataFrame(prof_rows).to_csv(os.path.join(a.output, "crust_profiles.csv"), index=False)
    summ = pd.DataFrame(summ_rows)
    summ.to_csv(os.path.join(a.output, "pair_summary.csv"), index=False)
    print(f"wrote outputs to {a.output}/")
    if len(summ):
        print(summ.drop(columns=["run_id"]).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
