#!/bin/bash
#SBATCH --job-name=geom_sem_neur
#SBATCH --partition=guppy
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/aniluchavez/semantic-curvature/podcast/logs/geometry_by_semantic_neurons.out
#SBATCH --error=/scratch/aniluchavez/semantic-curvature/podcast/logs/geometry_by_semantic_neurons.err

set -euo pipefail
cd /scratch/aniluchavez/semantic-curvature/podcast
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
PY=/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3
$PY -u compute_geometry_by_semantic_neurons.py ${1:-poisson} ${2:-shuffle}
