# v4 pipeline: bound H₂O in Archean oceanic crust (MAGEMin)

This pipeline computes how much water Archean oceanic crust can hold and where along a subducting slab it releases it, using MAGEMin phase-equilibrium calculations.
This README walks through reproducing the published v4 results on Georgia Tech's PACE Phoenix cluster, from a fresh clone to the figures.

- Repository: `heec12/Archean-OC-estimation`, branch `v4-magemin`
- Previous version (Perple_X): tag `v3-final`
- Contact: Hee Choi (hchoi342@gatech.edu)

---

## 1. What the model does

**Compositions.** The input is a designed set of crustal compositions rather than one average rock. Real analyses from the GSWA greenstone compilation (Smithies et al. 2018) are sampled across:

- **6 MgO bins** (6–18 wt%),
- **2 Al₂O₃/TiO₂ types**: Al-depleted and Al-undepleted,
- **8 replicates** per cell,

which gives **96 crustal pairs**. In each pair, the upper crust is a basalt and the lower crust is its cumulate complement: olivine ± pyroxene ± plagioclase plus 15 % trapped liquid, reaching up to about 35 wt% MgO.

**Calculation.** Each layer is equilibrated with 20 wt% excess H₂O, so the result is the layer's water-*holding capacity*. It is evaluated along slab P-T paths from a TerraFERMA thermal model and on a 40 × 40 P-T grid. The upper and lower crust are run separately and their bound water is mixed afterwards:

  crust = f · upper + (1 − f) · lower

**Accounting.** Each step of water release along a path is attributed to the mineral that releases it, such as lawsonite, chlorite or antigorite.

## 2. Scenarios

The three scenarios form a ladder. Each one changes a single thing relative to the previous step, so each difference can be attributed to that change.

| scenario | upper crust | lower crust | chemical system | difference from the previous step |
|---|---|---|---|---|
| **v4a** | MAGEMin `all` with a v3-like phase list, dataset ds62 | same | NCFMASTH (K₂O = 0, Fe³⁺ = 0) | v3 → v4a: Perple_X replaced by MAGEMin, kept as close to v3 as MAGEMin allows |
| **v4b** | `mb` = Green et al. (2016) metabasite set | `mb` | KNCFMASTOH (Fe³⁺/ΣFe = 0.1, K₂O = 0.15 wt%) | v4a → v4b: a self-consistent a-X set, plus Fe³⁺ and K |
| **v4** | `mb` | `ume` = Evans & Frost (2021) ultramafic set + G16 amphibole and augite. `ig` (Green et al. 2025) is also run as a check | same as v4b | v4b → v4: a-X set suited to serpentinised ultramafic rock in the lower crust |

**v4a phase list.** These are the closest MAGEMin equivalents of the v3 Perple_X models:
`amp_G16, ol_H18, cpx_T21, opx_T21, g_T21, fsp_H22, spl_W02, chl_W14, atg_EF21, ta_EF21, br_E13`, with pure H₂O as the fluid.

**Known model limits.** These are deliberate trade-offs, documented in `Methodology Updates.md`:

| database | limits |
|---|---|
| `ume` | no lawsonite, zoisite, epidote or melt; pyrope–almandine garnet only (no Ca); TiO₂ and K₂O dropped |
| `mb` | no antigorite, brucite or talc |
| `ig` | no antigorite or brucite; calibrated only above 923 K |

Points with melt present are flagged and excluded from all water-release accounting.

## 3. Prerequisites

You need:
- A PACE Phoenix account and a charge account for SLURM.
- About 2 GB of scratch space.

On the login node:

```bash
# 1. Clone the repo into scratch
cd ~/scratch
git clone https://github.com/heec12/Archean-OC-estimation.git
cd Archean-OC-estimation
git switch v4-magemin
REPO=$(pwd)

# 2. Python (for post-processing and plotting)
python3 -m venv ~/venvs/archean
source ~/venvs/archean/bin/activate
pip install numpy pandas matplotlib openpyxl
```

**Julia.** The Julia environment is pinned in `envs/v4/Project.toml` and `envs/v4/Manifest.toml` (MAGEMin_C 2.4.1). Install it inside an interactive allocation; precompiling on the login node is slow.

```bash
salloc -A <your-account> -N1 -c8 --mem-per-cpu=4G -t 1:00:00
module load julia/1.12.5
export JULIA_DEPOT_PATH=$HOME/scratch/.julia          # keep packages off /home
cd $REPO
julia --project=envs/v4 -e 'using Pkg; Pkg.instantiate(); using MAGEMin_C; println(pkgversion(MAGEMin_C))'
```

That should print `2.4.1`. MAGEMin_C must be **2.4.0 or newer**: older versions don't have the `all` database, and the driver refuses to run without it.

**Do not use the repo-root Julia environment for v4.** That one is for v3 (StatGeochem/Perple_X), and its dependencies pull MAGEMin_C back to an old version.

## 4. Run

### Step 1: setup check (≈1 min, in the `salloc` session)

```bash
julia --project=envs/v4 h2o_lookup/check_magemin_setup.jl
```

This lists the phases in each database and recomputes three reference points. **All three must print `OK`.** The expected bound H₂O values are:

- 9.957 wt% (`ume`)
- 10.012 wt% (`ig`)
- 6.855 wt% (`mb`)

### Step 2: pilot (≈5–10 min, same session)

The pilot covers 6 pairs across all scenarios, on the paths only:

```bash
julia --project=envs/v4 --threads 8 h2o_lookup/run_magemin_manifest.jl \
  --manifest design_outputs/composition_manifest_v4.csv \
  --paths design_outputs/pt_paths --outdir h2o_runs/v4_pilot \
  --pilot true --grid false
```

The run should end with `ok 72   skipped 0   failed 0`.

### Step 3: full ensemble (batch, about 1–3 h wall time)

1. Edit `#SBATCH --account=` in `h2o_lookup/submit_magemin_array.sbatch` to your charge account.
2. Check that `--partition` exists for you.
3. Submit **from the repo root**, because the script uses the submit directory as the repo path:

```bash
exit                                    # leave salloc
cd $REPO
mkdir -p logs                           # SLURM needs this before the job starts
sbatch h2o_lookup/submit_magemin_array.sbatch
```

The job is 16 array tasks × 8 threads covering 612 runs (all three scenarios, plus K₂O sensitivity runs on the 6 pilot pairs).

To monitor it:

```bash
squeue -u $USER
tail -f logs/magemin_<jobid>_0.out          # one [run]/[done] pair per row
ls h2o_runs/v4/pts_*.csv | wc -l            # target: 612
```

If some tasks fail, resubmit the same command. Runs that already finished are skipped.

If your Julia depot isn't at `~/scratch/.julia`, pass its location to the job:

```bash
sbatch --export=ALL,JULIA_DEPOT_PATH=/path/to/depot h2o_lookup/submit_magemin_array.sbatch
```

### Step 4: post-process (login node, Python environment active)

```bash
source ~/venvs/archean/bin/activate
python h2o_lookup/postprocess_v4.py --manifest design_outputs/composition_manifest_v4.csv \
  --runs h2o_runs/v4 --output response_v4
```

### Step 5: keep the results

PACE scratch is purged periodically. Copy `h2o_runs/v4`, `response_v4` and `figures_v4` to project storage.

### Optional: rebuild the manifest

`design_outputs/composition_manifest_v4.csv` is committed, so you only need this if you change the design, K₂O or Fe³⁺:

```bash
cd Bayesian
python augment_manifest_v4.py --manifest ../design_outputs/composition_manifest.csv \
  --data GSWA_Smithies2018_appendix1.xlsx --output ../design_outputs/composition_manifest_v4.csv
```

The base design (`composition_manifest.csv`) comes from `Bayesian/build_composition_design.py`; see `Methodology Updates.md`.

### Driver options

| option | effect |
|---|---|
| `--scenario v4a\|v4b\|v4` | only rows used by one scenario |
| `--db mb\|ume\|ig\|all` | only rows for one database |
| `--rows 1:10` | a range of manifest rows |
| `--pilot true` | the 6 pilot pairs only |
| `--grid false` | P-T paths only; skip the 40 × 40 lookup grid |

## 5. Files

| file | role |
|---|---|
| `Bayesian/augment_manifest_v4.py` | builds the v4 run list from the v3 design (Fe³⁺, K₂O, database per row, scenarios) |
| `design_outputs/composition_manifest_v4.csv` | 612 unique runs; the `scenarios` column says which scenarios use each run |
| `design_outputs/pt_paths/pt_path_*.csv` | slab P-T paths (`depth_km, P_GPa, T_K`); every `pt_path_*.csv` in this folder is run |
| `envs/v4/` | pinned Julia environment |
| `h2o_lookup/check_magemin_setup.jl` | installation check |
| `h2o_lookup/run_magemin_manifest.jl` | MAGEMin driver; no statistics, one output pair per run |
| `h2o_lookup/submit_magemin_array.sbatch` | SLURM array job |
| `h2o_lookup/postprocess_v4.py` | release by mineral, scenario assembly, crust mixing |
| `h2o_lookup/plot_v4.py` | comparison figures |

## 6. Outputs

All water amounts in the post-processed files are in **g H₂O per 100 g of anhydrous rock**.

**`h2o_runs/v4/`** (one pair of files per run)

| file | contents |
|---|---|
| `pts_<run_id>.csv` | one row per P-T point: bound H₂O in the solid (wt%), fluid and melt fractions, `saturated` / `melt_present` flags, assemblage, and `kind` (`path` or `grid`) |
| `phases_<run_id>.csv` | one row per phase per point: phase wt% of the solid, H₂O in the phase, and its contribution to bound H₂O |

**`response_v4/`**

| file | contents |
|---|---|
| `release_by_phase.csv` | per mineral and path: total released, share of the initial budget, and depths at which 10 / 50 / 90 % of its water is released |
| `release_steps.csv` | release between consecutive path points, with the donor minerals |
| `pair_summary.csv` | per scenario × pair × layer: initial bound H₂O, and the 10 km window with the largest release (its depth, amount and main mineral) |
| `crust_profiles.csv` | mixed-crust bound H₂O vs depth, for f = 0.2–0.5 |
| `lower_ume_vs_ig.csv` | v4 only: `ume` vs `ig` comparison with validity flags |

**Run IDs** follow `p<pair>_<layer>_<db>[-<phase_set>]_<k2o_level>`. For example, `p0088_lower_ume_kmorb` is pair 88, lower crust, `ume`, MORB-like K₂O.

## 7. Troubleshooting

| symptom | cause / fix |
|---|---|
| `Database not implemented... using default mp` | MAGEMin_C older than 2.4.0. Use `--project=envs/v4` and run `Pkg.instantiate()` |
| `ModuleNotFoundError: pandas` | Python environment not active; it doesn't carry into `salloc` or batch shells |
| job dies in ~1 s with `module: command not found` | the sbatch must keep `#!/bin/bash -l` |
| `.out` files empty while the job runs | Julia output buffering; check the `pts_*.csv` count instead |
| many `undersaturated` warnings | `H2O_EXCESS` (20 wt%) too low for that composition; bound H₂O there is censored, not a capacity |
