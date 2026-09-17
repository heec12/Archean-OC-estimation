#!/bin/bash -l
#SBATCH --job-name=archean_tl_py
#SBATCH --account=gts-ssim33-atlas
#SBATCH --qos=inferno
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/tl_py_%j.out
#SBATCH --error=logs/tl_py_%j.err
#
# Trapped-liquid sensitivity test, Python steps.
#   STEP=build    -> write design_outputs/manifest_trapped_liquid.csv
#   STEP=analyze  -> read h2o_runs/v3 + h2o_runs/v3_trapped, write results
# Submit with:  sbatch --export=ALL,STEP=build   h2o_lookup/submit_trapped_liquid_python.sbatch

set -euo pipefail

BASE=/storage/home/hcoda1/7/hchoi342/scratch/Archean-OC-estimation
DESIGN_SCRIPT=${BASE}/Bayesian/build_composition_design.py      # <-- confirm with find
cd "${BASE}"

module load anaconda3
eval "$(conda shell.bash hook)"
set +u; conda activate my_python; set -u
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg

echo "step    : ${STEP:?set STEP=build or STEP=analyze}"
echo "python  : $(which python)"
echo "started : $(date)"

case "${STEP}" in
  build)
    python h2o_lookup/build_trapped_liquid_manifest.py \
        --manifest      design_outputs/composition_manifest.csv \
        --design-script "${DESIGN_SCRIPT}" \
        --fractions     0.05,0.30 \
        --bins          0,4,5 \
        --output        design_outputs/manifest_trapped_liquid.csv
    ;;
  analyze)
    python h2o_lookup/analyze_trapped_liquid.py \
        --manifest    design_outputs/composition_manifest.csv \
        --tl-manifest design_outputs/manifest_trapped_liquid.csv \
        --runs        h2o_runs/v3 \
        --tl-runs     h2o_runs/v3_trapped \
        --output      response_outputs/trapped_liquid
    ;;
  *) echo "unknown STEP=${STEP}"; exit 1 ;;
esac

echo "finished: $(date)"
