#!/bin/bash
#SBATCH --job-name=curv_hilo_all
#SBATCH --partition=mammoth
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/aniluchavez/straighteningthenarrative/podcast/logs/curv_hilo_all.out
#SBATCH --error=/scratch/aniluchavez/straighteningthenarrative/podcast/logs/curv_hilo_all.err

set -euo pipefail
cd /scratch/aniluchavez/straighteningthenarrative/podcast
PY=/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3
$PY -u compute_curvature_hilo_all_models_all_layers.py --models llama-2-7b --suffix _llama2
