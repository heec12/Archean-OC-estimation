# v4 pipeline: bound H₂O with MAGEMin

v4 computes how much water Archean oceanic crust can carry and where it releases it along slab P-T paths.
The thermodynamics run in MAGEMin. The composition design from v3 is reused unchanged: MgO bin × Al₂O₃/TiO₂ type × 8 replicates, with a cumulate lower crust coupled to each upper crust (96 pairs).
v3 (Perple_X, NCFMAST) is kept at tag `v3-final`.

## Scenarios

The three scenarios are a ladder. Each step changes one thing relative to the one before, so each difference can be attributed to that change.

| scenario | upper crust | lower crust | system | what the comparison isolates |
|---|---|---|---|---|
| **v4a** | MAGEMin `all`, v3-like phase list, ds62 | same | NCFMAST (K₂O = 0, Fe³⁺ = 0) | v3 → v4a: switching engine (Perple_X → MAGEMin), kept as close to v3 as MAGEMin allows |
| **v4b** | `mb` (Green et al. 2016) | `mb` | + Fe³⁺/ΣFe = 0.1, K₂O = 0.15 wt% | v4a → v4b: self-consistent G16 set, plus Fe³⁺ and K |
| **v4** | `mb` | `ume` (Evans & Frost 2021 + G16 amp/aug), with `ig` (Green et al. 2025) run as a check | same as v4b | v4b → v4: the a-X set for the ultramafic lower crust |

**v4a phase list** (`all` database, `select_phases`). These are the closest MAGEMin equivalents of the v3 Perple_X list:
`amp_G16, ol_H18, cpx_T21, opx_T21, g_T21, fsp_H22, spl_W02, chl_W14, atg_EF21, ta_EF21, br_E13`, with pure H₂O as the fluid.
Antigorite and talc come from different models than in v3. The amphibole keeps all its end-members, whereas v3's `excludes` list likely removed ts, parg and gl. On ds636 instead of ds62, this list loses amphibole in basalt, so v4a stays on ds62.

**Known limits:**
- `ume` has no lawsonite, zoisite, epidote or melt.
- `ume` has only a Ca-free (pyrope–almandine) garnet.
- `ume` drops TiO₂ and K₂O.
- `mb` has no antigorite, brucite or talc.
- `ig` has no antigorite or brucite, and is calibrated only for T > 923 K.

With 20 wt% excess H₂O, `ig` forms a spurious water-rich liquid at about 3 GPa. Melt-bearing points are flagged and excluded from all release accounting.

## Files

| file | role |
|---|---|
| `Bayesian/augment_manifest_v4.py` | Turns the v3 manifest into v4 runs: splits Fe³⁺, sets K₂O, assigns a database per row, and lists the scenarios |
| `design_outputs/composition_manifest_v4.csv` | 612 unique runs. The `scenarios` column records which scenarios use each row; v4 and v4b share the upper-crust `mb` runs |
| `h2o_lookup/check_magemin_setup.jl` | Checks phase lists and 3 reference points. Run once before any job |
| `h2o_lookup/run_magemin_manifest.jl` | Driver: for each row, computes the P-T paths and the 40×40 grid, with no statistics |
| `h2o_lookup/submit_magemin_array.sbatch` | SLURM array job (16 tasks × 8 threads) |
| `h2o_lookup/postprocess_v4.py` | Release attributed to each mineral, ume vs ig, crust mixing, per-scenario summaries |
| `h2o_lookup/plot_v4.py` | Figures comparing the scenarios |
| `envs/v4/` | Julia environment for v4 (MAGEMin_C ≥ 2.4.0), separate from the v3/StatGeochem environment |

## Workflow (PACE)

```bash
cd /storage/scratch1/7/hchoi342/Archean-OC-estimation
git switch v4-magemin
```

**0. Rebuild the manifest.** This is only needed if the design, K₂O or Fe³⁺ changes. It needs `(pyenv)` active.
```bash
cd Bayesian
python augment_manifest_v4.py --manifest ../design_outputs/composition_manifest.csv \
  --data GSWA_Smithies2018_appendix1.xlsx --output ../design_outputs/composition_manifest_v4.csv
cd ..
```

**1. Set up and check.** Run this once, interactively (use the interactive node).

```bash
module load julia/1.12.5
export JULIA_DEPOT_PATH=/storage/scratch1/7/hchoi342/.julia
julia --project=envs/v4 -e 'using Pkg; Pkg.add(["CSV","DataFrames"]); Pkg.add(name="MAGEMin_C", version="2.4.1")'
julia --project=envs/v4 h2o_lookup/check_magemin_setup.jl     # all three must print OK
```

**2. Pilot.** 6 pairs, paths only, a few minutes.
```bash
julia --project=envs/v4 --threads 8 h2o_lookup/run_magemin_manifest.jl \
  --manifest design_outputs/composition_manifest_v4.csv \
  --paths design_outputs/pt_paths --outdir h2o_runs/v4_pilot \
  --pilot true --grid false
```
Other driver options:
- `--scenario v4a|v4b|v4` runs only the rows for one scenario.
- `--db mb|ume|ig|all` runs only the rows for one database.
- `--rows 1:10` runs only those manifest rows.

**3. Full ensemble.**
```bash
exit
sbatch h2o_lookup/submit_magemin_array.sbatch
ls h2o_runs/v4/pts_*.csv | wc -l        # target: 612
```
Runs that already finished are skipped, so if some fail, resubmitting the same job fills the gaps.

**4. Post-process and plot.** Run these on the login node with `(pyenv)` active.
```bash
python h2o_lookup/postprocess_v4.py --manifest design_outputs/composition_manifest_v4.csv \
  --runs h2o_runs/v4 --output response_v4
python h2o_lookup/plot_v4.py --response response_v4 --output figures_v4 --f 0.3
```

**5. Back up.** Scratch is purged, so copy the results to project storage.
```bash
cp -r h2o_runs/v4 response_v4 figures_v4 /storage/project/r-ssim33-0/hchoi342/
```

## Outputs

**Per run, written by the driver** (`h2o_runs/v4/`):
- `pts_<run_id>.csv`: one row per P-T point. Columns include bound H₂O in the solid (`h2o_solid`, wt%), the fluid and melt fractions, the `saturated` and `melt_present` flags, the assemblage, and `kind` (`path` or `grid`).
- `phases_<run_id>.csv`: one row per stable phase at each point. Gives the phase's wt% of the solid, its own H₂O content, and its contribution to bound H₂O (`h2o_contrib`).

**Post-processing** (`response_v4/`). All amounts are in g H₂O per 100 g of anhydrous rock.
- `release_by_phase.csv`: for each mineral along each path: total released, share of the initial budget, and the depths by which 10, 50 and 90 % of its release is done.
- `release_steps.csv`: release between each pair of consecutive path points, with the donor phases.
- `pair_summary.csv`: for each scenario × pair × layer: initial bound H₂O, plus the 10 km window with the largest release (its depth, amount and dominant mineral).
- `crust_profiles.csv`: crust bound H₂O against depth, computed as f · upper + (1 − f) · lower, for f = 0.2 to 0.5.
- `lower_ume_vs_ig.csv`: v4 only. Gives `ume` minus `ig` at each depth, flagged for beyond antigorite-out, `ig` in its calibrated range, and `ig` melt.

## Run IDs

Run IDs have the form `p<pair>_<layer>_<db>[-<phase_set>]_<k2o_level>`, for example:
- `p0088_lower_ume_kmorb`: pair 88, lower crust, `ume`, K₂O at the MORB-like value
- `p0088_upper_all-v3like_k0`: pair 88, upper crust, `all` with the v3-like phase list, no K₂O

The K₂O sensitivity rows (`k005`, `k050`, `kmeas`) exist for the 6 pilot pairs only. Their `scenarios` value is `v4-ksens`.

## Open items

- The 0.15 wt% K₂O baseline is a placeholder. Check it against a MORB compilation.
- The TerraFERMA lookup-table export from the grid points isn't written yet.
- The pilot showed a numerical wobble in v4a between 25 and 75 km, which makes its shallow release depths unreliable.
- The P-T paths are modern. The Archean paths still need to be added; drop them into `design_outputs/pt_paths/` as `pt_path_<name>.csv`.
- The questions for the MAGEMin developer (Discord) and for Palin are in `Methodology Updates.md`.
