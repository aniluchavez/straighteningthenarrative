#!/bin/bash
# PCA(20)-reduced-trajectory geometry pipeline, PODCAST, all 9 models.
# One array task per model.
#
# Submit with: sbatch run_geometry_pca20_podcast_array.sbatch

#SBATCH --job-name=geom_pca20_podcast
#SBATCH --partition=guppy
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-8
#SBATCH --output=/scratch/aniluchavez/semantic-curvature/podcast/logs/geom_pca20_podcast_%a.out
#SBATCH --error=/scratch/aniluchavez/semantic-curvature/podcast/logs/geom_pca20_podcast_%a.err

set -euo pipefail
cd /scratch/aniluchavez/semantic-curvature/podcast
mkdir -p logs

MODELS=(gpt2 gpt2-medium bert-base roberta-base opt-350m llama-2-7b llama-3.1-8b gemma-2-9b mistral-7b)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

PY=/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3
$PY -u run_podcast_geometry_analysis_pca20.py --models "$MODEL" --regions hippocampus
