#!/bin/bash
#SBATCH --job-name=archean_h2o
#SBATCH --account=gts-ssim33-atlas
#SBATCH --partition=inferno
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00
#SBATCH --array=0-15
#SBATCH --output=logs/perplex_%A_%a.out
#SBATCH --error=logs/perplex_%A_%a.err

# set -euo pipefail must come near the TOP, right after the shebang block --
# not after the Julia call, or failures partway through go unnoticed.
set -euo pipefail

BASE=/storage/home/hcoda1/7/hchoi342/scratch/Archean-OC-estimation
MANIFEST=${BASE}/design_outputs/composition_manifest.csv
SCRIPT=${BASE}/h2o_lookup/run_perplex_manifest.jl

mkdir -p logs

# Julia buffers stdout on SLURM, which is why .out files come back empty while
# a job is clearly running. Unbuffer at the shell level -- there is no
# JULIA_STDOUT_BUFFER variable inside Julia; the script also calls
# flush(stdout) after each run.
export JULIA_DEPOT_PATH=/storage/scratch1/7/hchoi342/.julia
export PYTHONUNBUFFERED=1
stdbuf -oL -eL true 2>/dev/null || true

echo "host        : $(hostname)"
echo "array task  : ${SLURM_ARRAY_TASK_ID} / ${SLURM_ARRAY_TASK_COUNT}"
echo "manifest    : ${MANIFEST}"
echo "started     : $(date)"

# --project points at the h2o_lookup environment
stdbuf -oL -eL julia --project="${BASE}/h2o_lookup" \
    --threads "${SLURM_CPUS_PER_TASK}" \
    "${SCRIPT}" \
    --manifest "${MANIFEST}" \
    --scratch  "${BASE}/perplex_ensemble" \
    --version  v3

echo "finished    : $(date)"

# Completed runs are skipped on restart, so resubmitting this exact job fills
# any holes left by failures:
#     sbatch submit_perplex_array.sbatch
#
# To watch progress while .out files are quiet, ssh to the compute node and:
#     ps aux | grep hchoi342
# looking for active vertex / werami processes.
