#!/bin/bash
#SBATCH --job-name=sem_neuron_enc
#SBATCH --partition=guppy
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/aniluchavez/semantic-curvature/podcast/logs/semantic_neuron_encoding_fixed250.out
#SBATCH --error=/scratch/aniluchavez/semantic-curvature/podcast/logs/semantic_neuron_encoding_fixed250.err

set -euo pipefail
cd /scratch/aniluchavez/semantic-curvature/podcast
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3
$PY -u compute_semantic_neuron_encoding_fixed250.py
