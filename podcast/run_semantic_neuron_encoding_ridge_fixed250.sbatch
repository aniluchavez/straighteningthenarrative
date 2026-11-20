#!/bin/bash
#SBATCH --job-name=sem_ridge_enc
#SBATCH --partition=guppy
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/aniluchavez/semantic-curvature/podcast/logs/semantic_neuron_encoding_ridge_fixed250.out
#SBATCH --error=/scratch/aniluchavez/semantic-curvature/podcast/logs/semantic_neuron_encoding_ridge_fixed250.err

set -euo pipefail
cd /scratch/aniluchavez/semantic-curvature/podcast
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
PY=/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3
$PY -u compute_semantic_neuron_encoding_ridge_fixed250.py
