#!/bin/bash -l
#SBATCH --job-name=archean_post
#SBATCH --account=gts-ssim33-atlas
#SBATCH --qos=inferno
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/post_%j.out
#SBATCH --error=logs/post_%j.err

set -euo pipefail

BASE=/storage/home/hcoda1/7/hchoi342/scratch/Archean-OC-estimation
cd "${BASE}"

module load anaconda3
# conda activate <your_env>    # uncomment if pandas/etc. live in a named env
export PYTHONUNBUFFERED=1

echo "started : $(date)"
echo "python  : $(which python)"

echo "=== sanity check ==="
python h2o_lookup/check_h2o_runs.py      # set -e stops the job here if it fails

echo "=== post-processing ==="
python h2o_lookup/postprocess_response.py \
    --manifest design_outputs/composition_manifest.csv \
    --runs     h2o_runs/v3 \
    --output   response_outputs

echo "finished: $(date)"
