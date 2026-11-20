#!/bin/bash
#SBATCH --job-name=poisson_geom
#SBATCH --partition=guppy
#SBATCH --array=0-3
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/aniluchavez/straighteningthenarrative/podcast/logs/poisson_geom_%a.out
#SBATCH --error=/scratch/aniluchavez/straighteningthenarrative/podcast/logs/poisson_geom_%a.err

set -euo pipefail
cd /scratch/aniluchavez/straighteningthenarrative/podcast
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
MODES=(perpatient perpatient pseudopop pseudopop)
WINS=(dur8040 fixed250 dur8040 fixed250)
PY=/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3
$PY -u compute_poisson_normalized_geometry.py ${MODES[$SLURM_ARRAY_TASK_ID]} ${WINS[$SLURM_ARRAY_TASK_ID]}
