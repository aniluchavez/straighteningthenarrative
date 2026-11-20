#!/bin/bash
#SBATCH --job-name=poisson_geom_pc
#SBATCH --partition=guppy
#SBATCH --array=0-19
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/aniluchavez/semantic-curvature/podcast/logs/poisson_geom_pc_%a.out
#SBATCH --error=/scratch/aniluchavez/semantic-curvature/podcast/logs/poisson_geom_pc_%a.err

set -euo pipefail
cd /scratch/aniluchavez/semantic-curvature/podcast
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
MODES=(perpatient perpatient pseudopop pseudopop)
WINS=(dur8040 fixed250 dur8040 fixed250)
KS=(8 16 32 64 128)
i=$SLURM_ARRAY_TASK_ID
c=$((i % 4)); k=${KS[$((i / 4))]}
PY=/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3
$PY -u compute_poisson_normalized_geometry.py ${MODES[$c]} ${WINS[$c]} $k
