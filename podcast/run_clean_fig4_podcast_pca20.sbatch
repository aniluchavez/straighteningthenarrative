#!/bin/bash
#SBATCH --job-name=fig4_pca20_podcast
#SBATCH --partition=guppy
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/aniluchavez/straighteningthenarrative/podcast/logs/fig4_pca20_podcast.out
#SBATCH --error=/scratch/aniluchavez/straighteningthenarrative/podcast/logs/fig4_pca20_podcast.err

set -euo pipefail
cd /scratch/aniluchavez/straighteningthenarrative/podcast
mkdir -p logs
PY=/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3
$PY -u run_clean_fig4_tests_podcast_pca20.py --model llama-3.1-8b --surprisal_lag 0 --causal
