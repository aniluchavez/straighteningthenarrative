#!/bin/bash
#SBATCH --job-name=scatter_llama2
#SBATCH --partition=mammoth
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/aniluchavez/straighteningthenarrative/podcast/logs/scatter_llama2.out
#SBATCH --error=/scratch/aniluchavez/straighteningthenarrative/podcast/logs/scatter_llama2.err

set -euo pipefail
cd /scratch/aniluchavez/straighteningthenarrative/podcast
PY=/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3
$PY -u compute_scatter_data_podcast_pca20_llama2.py
