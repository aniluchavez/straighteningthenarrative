#!/bin/bash
#SBATCH --job-name=static_curv_surp
#SBATCH --partition=guppy
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/aniluchavez/straighteningthenarrative/podcast/logs/static_curv_surp.out
#SBATCH --error=/scratch/aniluchavez/straighteningthenarrative/podcast/logs/static_curv_surp.err

set -euo pipefail
cd /scratch/aniluchavez/straighteningthenarrative/podcast
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3 -u compute_static_embedding_curvature_surprisal.py
