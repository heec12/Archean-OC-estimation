#!/bin/bash -l
# The -l (login shell) matters: SLURM runs batch scripts as a NON-login shell
# by default, so /etc/profile.d is never sourced and `module` is undefined.
# Without it, `module load julia` fails with "command not found" and the job
# dies in about one second.
#SBATCH --job-name=archean_h2o_v4
#SBATCH --account=gts-ssim33-atlas
#SBATCH --partition=cpu-medium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=24:00:00
#SBATCH --array=0-15
#SBATCH --output=logs/magemin_%A_%a.out
#SBATCH --error=logs/magemin_%A_%a.err

# set -euo pipefail must come near the TOP, right after the shebang block --
# not after the Julia call, or failures partway through go unnoticed.
set -euo pipefail

BASE=/storage/home/hcoda1/7/hchoi342/scratch/Archean-OC-estimation
MANIFEST=${BASE}/design_outputs/composition_manifest_v4.csv
SCRIPT=${BASE}/h2o_lookup/run_magemin_manifest.jl

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

# Julia lives at the REPO ROOT environment (Project.toml + Manifest.toml are
# there, and that is the one with StatGeochem). h2o_lookup/Project.toml exists
# too but does not resolve StatGeochem -- using it gives
# "Package StatGeochem not found in current path".
#
# Precompile interactively before submitting:
#     julia --project=. -e 'using Pkg; Pkg.add("MAGEMin_C"); using MAGEMin_C'
#     julia --project=. h2o_lookup/check_magemin_setup.jl
# The first precompile would otherwise happen inside the first array task,
# where it looks indistinguishable from a hang. The check script must print OK
# for all three reference points before you submit.
#
# TODO: replace with whatever makes julia available on a compute node in your
# setup -- a `module load julia`, or a PATH export for a local install. A
# fresh compute-node shell does not have it.
# Pinned rather than bare `module load julia`: the Phoenix default is marked
# (D) and can move between 1.10.1 / 1.11.3 / 1.12.5. Manifest.toml was
# resolved under a specific version, and a silent switch means a full
# recompile at best and a resolution failure at worst. Change this line only
# together with a deliberate Manifest.toml update.
module load julia/1.12.5
command -v julia >/dev/null || { echo "ERROR: julia not on PATH after module load"; exit 1; }
echo "julia       : $(julia --version)  ($(command -v julia))"
module list 2>&1 | sed 's/^/  module: /'

stdbuf -oL -eL julia --project="${BASE}" \
    --threads "${SLURM_CPUS_PER_TASK}" \
    "${SCRIPT}" \
    --manifest "${MANIFEST}" \
    --paths    "${BASE}/design_outputs/pt_paths" \
    --outdir   "${BASE}/h2o_runs/v4" \
    --version  v4
# Pilot first (6 pairs, paths only, a few minutes):
#     sbatch --array=0-1 submit_magemin_array.sbatch   with  --pilot true --grid false
# added to the julia line above. One scenario only: add  --scenario v4a  (or v4b, v4).

echo "finished    : $(date)"

# Completed runs are skipped on restart, so resubmitting this exact job fills
# any holes left by failures:
#     sbatch submit_perplex_array.sbatch
#
# MAGEMin runs in-process (no vertex/werami children), so progress shows in the
# .out file: one [run ] / [done] line pair per manifest row.
