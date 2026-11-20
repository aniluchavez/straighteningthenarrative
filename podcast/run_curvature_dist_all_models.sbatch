#!/bin/bash
#SBATCH --job-name=curv_dist_all
#SBATCH --partition=guppy
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/aniluchavez/semantic-curvature/podcast/logs/curv_dist_all.out
#SBATCH --error=/scratch/aniluchavez/semantic-curvature/podcast/logs/curv_dist_all.err

set -euo pipefail
cd /scratch/aniluchavez/semantic-curvature/podcast
PY=/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3
$PY -u compute_curvature_dist_all_models_all_layers.py --models llama-2-7b --suffix _llama2
