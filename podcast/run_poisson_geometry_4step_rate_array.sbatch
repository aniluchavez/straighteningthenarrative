#!/bin/bash
#SBATCH --job-name=poisson_4step_rate
#SBATCH --partition=guppy
#SBATCH --array=0-11
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/aniluchavez/straighteningthenarrative/podcast/logs/poisson_4step_rate_%a.out
#SBATCH --error=/scratch/aniluchavez/straighteningthenarrative/podcast/logs/poisson_4step_rate_%a.err

set -euo pipefail
cd /scratch/aniluchavez/straighteningthenarrative/podcast
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
MODES=(perpatient pseudopop)
KS=(8 16 32 64 128 full)
i=$SLURM_ARRAY_TASK_ID
PY=/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3
$PY -u compute_poisson_normalized_geometry.py ${MODES[$((i % 2))]} dur8040 ${KS[$((i / 2))]} four rate
