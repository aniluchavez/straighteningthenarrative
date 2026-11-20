#!/bin/bash
#SBATCH --job-name=scat_pca20_podcast
#SBATCH --partition=guppy
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/aniluchavez/straighteningthenarrative/podcast/logs/scat_pca20_podcast.out
#SBATCH --error=/scratch/aniluchavez/straighteningthenarrative/podcast/logs/scat_pca20_podcast.err

set -euo pipefail
cd /scratch/aniluchavez/straighteningthenarrative/podcast
mkdir -p logs
PY=/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3
$PY -u compute_scatter_data_podcast_pca20.py
